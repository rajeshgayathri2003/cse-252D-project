import os
import shutil
import torch
import prior
from ai2thor.controller import Controller
from PIL import Image
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForCausalLM
from IPython.display import display, clear_output
import time

try:
    import ai2thor_colab
except ImportError:
    ai2thor_colab = None

# Import the patching tools needed for the fix
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Florence-2 commonly misnames certain objects; remap to the canonical label.
LABEL_ALIASES: dict[str, str] = {
    "drawer": "desk",
    "chest of drawers": "desk",
    "backpack": "garbage bag",
}

def workaround_fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Intercepts the dependency check and removes flash_attn."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class FlorencePerceptionAgent:
    def __init__(
        self, 
        florence_model="microsoft/Florence-2-base", 
        sam_weights="sam2_b.pt", 
        save_dir="saved_agent_data", 
        headless=True,
        device=None # Added device parameter
    ):
        if headless:
            if ai2thor_colab is not None:
                ai2thor_colab.start_xserver()
            else:
                print(
                    "[PerceptionAgent] ai2thor_colab unavailable; skipping start_xserver(). "
                    "Use CloudRendering or set headless=False if you need an X server."
                )
        self.save_dir = save_dir
        
        # Set device based on user input or auto-detect (prefer cuda > mps > cpu).
        if device and device != "auto":
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"[PerceptionAgent] Initializing with device: {self.device}")
        
        # 1. Load Florence-2 (The Open-Vocabulary Detector)
        print("Loading Florence-2... (This may take a moment)")
        self.processor = AutoProcessor.from_pretrained(florence_model, trust_remote_code=True)

        # --- THE FIX ---
        # We wrap the model loading step in the patch context manager.
        # This forces the loader to use our modified dependency list.
        with patch("transformers.dynamic_module_utils.get_imports", workaround_fixed_get_imports):
            self.florence = AutoModelForCausalLM.from_pretrained(
                florence_model, 
                trust_remote_code=True,
                attn_implementation="sdpa" # Fall back to standard PyTorch attention
            ).to(self.device)
        
        # 2. Load SAM2 (The Segmenter)
        print("Loading SAM2...")
        self.sam = SAM(sam_weights)
        self.sam.to(self.device)
        
        # 3. Dynamic Class Dictionary for Dataset Labeling
        self.class_map = {}
        self.next_class_id = 0
        
        # Clean workspace
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_class_id(self, text_label: str) -> int:
        """Dynamically assigns an integer ID to new object strings discovered by Florence."""
        clean_label = text_label.lower().strip()
        if clean_label not in self.class_map:
            self.class_map[clean_label] = self.next_class_id
            self.next_class_id += 1
            
            # Save/Update the global class dictionary for the dataset
            dict_path = os.path.join(self.save_dir, "classes.txt")
            with open(dict_path, "w") as f:
                for name, cid in self.class_map.items():
                    f.write(f"{cid}: {name}\n")
                    
        return self.class_map[clean_label]

    def _get_spatial_descriptor(self, normalized_polygon) -> str:
        """Translates normalized polygon points into spatial text."""
        x_coords = [pt[0] for pt in normalized_polygon]
        y_coords = [pt[1] for pt in normalized_polygon]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        horizontal = "center"
        if center_x < 0.33: horizontal = "left"
        elif center_x > 0.66: horizontal = "right"
            
        vertical = "center"
        if center_y < 0.33: vertical = "top"
        elif center_y > 0.66: vertical = "bottom"
            
        if horizontal == "center" and vertical == "center":
            return "in the center"
        return f"in the {vertical}-{horizontal}"

    def _region_category(self, image: Image.Image, bbox: list) -> str:
        """Ask Florence what object is in the given bbox region."""
        x1, y1, x2, y2 = bbox
        prompt = "<REGION_TO_CATEGORY>"
        region_str = f"<loc_{int(x1/image.width*999)}><loc_{int(y1/image.height*999)}><loc_{int(x2/image.width*999)}><loc_{int(y2/image.height*999)}>"
        inputs = self.processor(text=prompt + region_str, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        # Strip any residual <loc_XXXX> tokens from the category string
        import re as _re
        raw = str(parsed.get(prompt, ""))
        return _re.sub(r"<[^>]+>", "", raw).strip().lower()

    def _florence_ovd_pass(self, image: Image.Image, target_label: str, max_area_frac: float = 0.30):
        """Run OPEN_VOCABULARY_DETECTION for a target label Florence's OD missed.

        Each candidate bbox is validated with REGION_TO_CATEGORY — only kept if
        Florence's region classifier agrees it plausibly matches the target.
        """
        # Keywords that count as a match for each target label
        VALIDATION_KEYWORDS = {
            "plunger": ["plunger", "toilet brush", "cleaning tool"],
            "clothes dryer": ["dryer", "washer", "laundry", "washing machine", "appliance"],
            "desk": ["desk", "shelf", "drawer", "cabinet", "furniture", "table"],
            "television": ["television", "tv", "monitor", "screen", "display"],
            "refrigerator": ["refrigerator", "fridge", "freezer", "appliance"],
            "trash bag": ["bag", "trash", "garbage", "sack", "backpack"],
        }
        keywords = VALIDATION_KEYWORDS.get(target_label.lower(), [target_label.lower()])

        prompt = "<OPEN_VOCABULARY_DETECTION>"
        inputs = self.processor(text=prompt + target_label, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        result = parsed.get(prompt, {})
        if not isinstance(result, dict):
            return [], []
        img_area = image.width * image.height
        bboxes, labels = [], []
        for bbox in result.get("bboxes", []):
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area / img_area > max_area_frac:
                continue
            # Validate with region classification
            region_cat = self._region_category(image, bbox)
            if any(kw in region_cat for kw in keywords):
                print(f"\n[DEBUG] OVD region validated as '{region_cat}' for target '{target_label}'")
                bboxes.append(bbox)
                labels.append(target_label)
            else:
                print(f"\n[DEBUG] OVD bbox rejected by region check: '{region_cat}' != '{target_label}'")
        return bboxes, labels

    def perceive(self, image: Image.Image, frame_name: str, target_label: str = None) -> str:
        step_dir = os.path.join(self.save_dir, frame_name)
        os.makedirs(step_dir, exist_ok=True)
        
        # Save raw image
        img_path = os.path.join(step_dir, "frame.jpg")
        image.save(img_path)
        label_file = os.path.join(step_dir, "frame.txt")
        
        # --- STEP 1: Florence-2 Open Vocabulary Detection ---
        prompt = "<OD>" # Switched to standard Object Detection
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        
        # Print exactly what Florence sees to the console so we aren't flying blind
        print(f"\n[DEBUG] Florence Output: {parsed_answer}")
        
        detections = parsed_answer.get(prompt, {})
        
        # --- THE FIX ---
        # Check if Florence returned a proper dictionary. 
        # If it panicked and returned a string, default to empty lists.
        if isinstance(detections, dict):
            bboxes = detections.get('bboxes', [])
            labels = detections.get('labels', [])
        else:
            bboxes = []
            labels = []
            
        # OVD supplement: only for objects Florence's generic OD can't reliably detect.
        # OVD can hallucinate small bboxes for absent objects, so we restrict it to a
        # known list rather than running it for every target.
        # Map camelCase target_type keys to human-readable OVD query strings.
        OVD_NEEDED = {
            "plunger": "plunger",
            "clothesdryer": "clothes dryer",
            "studydeskwithshelf": "desk",
            "television": "television",
            "fridge": "refrigerator",
            "refrigerator": "refrigerator",
            "garbagebag": "trash bag",
        }
        target_key = target_label.lower().replace(" ", "") if target_label else None
        ovd_query = OVD_NEEDED.get(target_key)
        if ovd_query and not any(ovd_query in l.lower() for l in labels):
            extra_bboxes, extra_labels = self._florence_ovd_pass(image, ovd_query)
            if extra_bboxes:
                print(f"\n[DEBUG] OVD found '{ovd_query}': {extra_bboxes}")
                bboxes = list(bboxes) + extra_bboxes
                labels = list(labels) + extra_labels
            else:
                print(f"\n[DEBUG] OVD: '{ovd_query}' not in frame.")

        # Handle Empty Detections
        if len(bboxes) == 0:
            open(label_file, 'w').close()
            return "No distinct objects are visible in the current view."

        # --- STEP 2: SAM2 Segmentation from Florence Boxes ---
        # We pass the Florence bounding boxes directly to SAM as prompts
        sam_results = self.sam(image, bboxes=bboxes, verbose=False)[0]
        
        description_lines = ["Visible Objects (Segmented):"]
        
        # Open file to write dataset labels
        with open(label_file, "w") as f:
            # Check if masks were successfully generated
            if sam_results.masks is not None:
                # Iterate through every object detected
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    label = LABEL_ALIASES.get(label.lower().strip(), label)
                    # Get normalized polygon coordinates from SAM2
                    polygon = sam_results.masks.xyn[i]

                    if len(polygon) < 3:
                        continue # Skip invalid masks

                    # 1. Write to dataset txt file
                    class_id = self._get_class_id(label)
                    flat_coords = " ".join([f"{pt[0]:.5f} {pt[1]:.5f}" for pt in polygon])
                    f.write(f"{class_id} {flat_coords}\n")
                    
                    # 2. Build Text Description for LLM
                    spatial_loc = self._get_spatial_descriptor(polygon)
                    
                    # Calculate Area
                    width = max([p[0] for p in polygon]) - min([p[0] for p in polygon])
                    height = max([p[1] for p in polygon]) - min([p[1] for p in polygon])
                    area_ratio = width * height
                    
                    description_lines.append(f"- {label} located {spatial_loc}, covering roughly {area_ratio:.1%} of the view.")

        # Updated to safely check string prefix for CUDA devices (e.g. "cuda:0")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return "\n".join(description_lines)

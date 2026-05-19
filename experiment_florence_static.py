"""Florence-2 experimentation on static images.

Same model + same tasks as experiment_florence.py, but reads images
from disk instead of from a live AI2-THOR controller. Lets us validate
Florence inference on DSMLP without needing Xvfb or libvulkan
(neither is installed on the cuda128 pod image).

Where the images come from:
- Drop any .jpg/.png files into test_images/ and they'll be processed.
- If test_images/ is empty (or missing), the script downloads a single
  Florence demo image (a car photo) so you have something to run on.

Run on DSMLP after `uv sync --extra sim --extra perception` and the
opencv-python-headless swap:
    uv run python experiment_florence_static.py

Outputs land in florence_static_outputs/<image_name>/:
    results.txt            — printed task outputs for that image
    annotated.jpg          — the image with <OD> bboxes drawn on
"""

import os
import urllib.request
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "microsoft/Florence-2-base"
INPUT_DIR = "test_images"
SAVE_DIR = "florence_static_outputs"

# Florence-2 task prompts to try on each image.
TASKS = [
    "<DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
]

# Fallback if the input directory is empty.
DEMO_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/transformers/tasks/car.jpg"
)


def load_florence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor, device, dtype


def run_task(model, processor, device, dtype, image, task_prompt):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed


def annotate_od(image, od_result, save_path):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    data = od_result.get("<OD>", {})
    for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), str(label), fill="red")
    img.save(save_path)


def gather_input_paths():
    os.makedirs(INPUT_DIR, exist_ok=True)
    paths = []
    for name in sorted(os.listdir(INPUT_DIR)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(INPUT_DIR, name))
    if paths:
        return paths

    # Empty dir — pull the Florence demo image so we have something.
    demo_path = os.path.join(INPUT_DIR, "florence_demo_car.jpg")
    print(f"{INPUT_DIR}/ is empty — downloading demo image to {demo_path}")
    urllib.request.urlretrieve(DEMO_IMAGE_URL, demo_path)
    return [demo_path]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    image_paths = gather_input_paths()
    print(f"Found {len(image_paths)} input image(s): "
          f"{[os.path.basename(p) for p in image_paths]}")

    model, processor, device, dtype = load_florence()

    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")
        out_dir = os.path.join(SAVE_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        header = f"\n{'=' * 60}\n  Image: {name}  ({image.size[0]}x{image.size[1]})\n{'=' * 60}"
        print(header)
        log_lines = [header]

        results = {}
        for task in TASKS:
            print(f"\n--- Task: {task} ---")
            log_lines.append(f"\n--- Task: {task} ---")
            result = run_task(model, processor, device, dtype, image, task)
            results[task] = result
            print(result)
            log_lines.append(str(result))

        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write("\n".join(log_lines))

        if "<OD>" in results:
            annotate_od(image, results["<OD>"], os.path.join(out_dir, "annotated.jpg"))

    print(f"\n[Done. Outputs in {SAVE_DIR}/]")


if __name__ == "__main__":
    main()

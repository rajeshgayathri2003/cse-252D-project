"""Florence-2 experimentation on ProcTHOR frames.

Microsoft's Florence-2 is open-vocab and trained on much richer data
than YOLO/COCO, so it actually produces useful detections and captions
on ProcTHOR. This script runs it on 5 frames and dumps three different
task outputs per frame so you can see what Florence sees vs what YOLO
returned (nothing).

Run on DSMLP after `uv sync --extra sim --extra perception`:
    uv run python experiment_florence.py

Outputs land in florence_outputs/step_N/:
    frame.jpg              — raw ai2thor RGB
    frame_annotated.jpg    — frame with <OD> bboxes drawn on
    results.txt            — printed task outputs

Switch MODEL_ID to "microsoft/Florence-2-large" for stronger quality
(2-3x slower per inference, fits comfortably in 24GB VRAM).
"""

import os
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from ai2thor.controller import Controller
import prior


PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"
MODEL_ID = "microsoft/Florence-2-base"
SAVE_DIR = "florence_outputs"
TOTAL_STEPS = 5
DEGREES_PER_STEP = 30

# Florence-2 task prompts to try on each frame. See model card for full list.
TASKS = [
    "<DETAILED_CAPTION>",       # natural-language scene description
    "<OD>",                     # open-vocab object detection (bboxes + labels)
    "<DENSE_REGION_CAPTION>",   # bbox per region + a short caption each
]


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
    """Draw bboxes from <OD> result onto a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    data = od_result.get("<OD>", {})
    bboxes = data.get("bboxes", [])
    labels = data.get("labels", [])
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), str(label), fill="red")
    img.save(save_path)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # AI2-THOR's Linux build needs a GLX-capable display. ai2thor_colab.start_xserver()
    # boots xvfb (no sudo needed) so the simulator can render headlessly on DSMLP.
    import ai2thor_colab
    print("Starting headless X server (xvfb)...")
    ai2thor_colab.start_xserver()

    model, processor, device, dtype = load_florence()

    print("Loading procthor-10k...")
    dataset = prior.load_dataset("procthor-10k", revision=PROCTHOR_REVISION)
    house = dataset["train"][0]

    print("Starting ai2thor controller...")
    controller = Controller(scene=house)

    try:
        for step in range(TOTAL_STEPS):
            if step > 0:
                event = controller.step(action="RotateRight", degrees=DEGREES_PER_STEP)
            else:
                event = controller.last_event

            frame = Image.fromarray(event.frame)
            step_dir = os.path.join(SAVE_DIR, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            frame.save(os.path.join(step_dir, "frame.jpg"))

            header = (f"\n{'=' * 60}\n"
                      f"  Step {step + 1}/{TOTAL_STEPS} "
                      f"(rotated {step * DEGREES_PER_STEP}°)\n"
                      f"{'=' * 60}")
            print(header)
            log_lines = [header]

            results = {}
            for task in TASKS:
                print(f"\n--- Task: {task} ---")
                log_lines.append(f"\n--- Task: {task} ---")
                result = run_task(model, processor, device, dtype, frame, task)
                results[task] = result
                print(result)
                log_lines.append(str(result))

            with open(os.path.join(step_dir, "results.txt"), "w") as f:
                f.write("\n".join(log_lines))

            if "<OD>" in results:
                annotate_od(frame, results["<OD>"],
                            os.path.join(step_dir, "frame_annotated.jpg"))
    finally:
        controller.stop()

    print(f"\n[Florence experiment complete. Outputs in {SAVE_DIR}/]")


if __name__ == "__main__":
    main()

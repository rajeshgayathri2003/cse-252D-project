"""Smoke test for the perception agent.

Verifies DSMLP can run the existing YOLO11m + SAM2-Base pipeline against
a ProcTHOR scene end-to-end. Detection quality is NOT what this checks
(YOLO is COCO-trained and largely misses ProcTHOR objects — expected).
What this checks: GPU is detected, ai2thor boots on Linux, procthor
scene loads, YOLO + SAM2 inference run without error, frames + labels
land in saved_agent_data/.

Run on DSMLP after:
    uv sync --extra sim --extra perception
    uv run python run_perception_smoke.py
"""

from PIL import Image
from ai2thor.controller import Controller
import prior

from agents.perception.agent import PerceptionAgent

# Match the revision pinned in test_2.py and plan/base.py.
PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"

TOTAL_STEPS = 5
DEGREES_PER_STEP = 30


def main():
    print("Loading procthor-10k...")
    dataset = prior.load_dataset("procthor-10k", revision=PROCTHOR_REVISION)
    house = dataset["train"][0]

    print("Starting ai2thor controller...")
    controller = Controller(scene=house)

    print("Initializing perception agent (first run downloads YOLO11m + SAM2-Base)...")
    perception_agent = PerceptionAgent(
        yolo_weights="yolo11m.pt",
        sam_weights="sam2_b.pt",
        headless=True,
    )

    try:
        for step in range(TOTAL_STEPS):
            if step > 0:
                event = controller.step(action="RotateRight", degrees=DEGREES_PER_STEP)
            else:
                event = controller.last_event

            frame = Image.fromarray(event.frame)
            description = perception_agent.perceive(frame, frame_name=f"step_{step}")

            print(f"\n=== Step {step + 1}/{TOTAL_STEPS} "
                  f"(rotated {step * DEGREES_PER_STEP}°) ===")
            print(description)
    finally:
        controller.stop()

    print("\n[Smoke test complete. Frames + labels saved under saved_agent_data/.]")


if __name__ == "__main__":
    main()

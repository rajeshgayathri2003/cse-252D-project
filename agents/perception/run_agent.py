# Install the core simulator, dataset libraries, and the Colab display utility
!pip install --upgrade ai2thor prior ai2thor_colab

!pip install -q --upgrade ai2thor ai2thor_colab prior
# Install the vision models required by the Perception Agent
!pip install --upgrade ultralytics transformers

from ai2thor.controller import Controller
import prior

dataset = prior.load_dataset("procthor-10k")
house = dataset["train"][0]
controller = Controller(scene=house)

perception_agent = PerceptionAgent(yolo_weights="yolo11m.pt", sam_weights="sam2_b.pt",headless = True)

total_steps = 10
degrees_per_step = 15

for step in range(total_steps):
    if step > 0:
        event = controller.step(action="RotateRight", degrees=degrees_per_step)
    else:
        event = controller.last_event
    
    current_frame = Image.fromarray(event.frame)
    
    # Pass a unique identifier (e.g., "step_0") to save the data permanently
    step_id = f"step_{step}"
    scene_description = perception_agent.perceive(current_frame, frame_name=step_id)
    
    clear_output(wait=True)
    print(f"======================================")
    print(f"  Agent Step {step + 1}/{total_steps} (Turned {step * degrees_per_step}° total)  ")
    print(f"======================================")
    print(f"Data saved to: /saved_agent_data/{step_id}/")
    
    display(current_frame)
    print("\n" + scene_description)
    time.sleep(0.5)

print("\n[Sequence Complete. All masks and boxes are saved in the 'saved_agent_data' folder.]")

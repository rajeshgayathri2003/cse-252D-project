"""Integration pipeline for Florence-2 perception, mapping, planning, and critic.

This script wires together the agents that exist today:

1. Florence-2 perception reads the current ProcTHOR RGB frame.
2. MappingAgent stores the current pose and Florence object labels.
3. PlanningAgent proposes the next sub-goal/action.
4. CriticAgent reviews the planner output before any execution.
5. ActionAgent translates the approved subgoal into a concrete AI2-THOR action and executes it.

Run from the repo root, after installing sim/perception dependencies:
    uv run python run_navigation_pipeline.py --task "find a flower vase"
"""

import argparse
import os
from dataclasses import dataclass

import prior
from ai2thor.controller import Controller
from PIL import Image

from act.base import ActionAgent
from agents.critic_agent import CriticAgent
from agents.mapping_agent import MappingAgent
try:
    from agents.perception.agent import FlorencePerceptionAgent
except ImportError as e:
    print(f"Error importing FlorencePerceptionAgent: {e}")
    FlorencePerceptionAgent = None
from plan.base import PlanningAgent, encode_image


FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
SAM_WEIGHTS = "sam2_b.pt"


@dataclass
class PerceptionResult:
    description: str
    objects: list[dict]
    frame_path: str


def build_controller(scene_index):
    print("[Sim] Loading ProcTHOR house...")
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][scene_index]
    print("[Sim] Starting AI2-THOR controller with CloudRendering...")
    try:
        return Controller(scene=house, platform="CloudRendering")
    except:
        return Controller(scene=house)


def run_perception(perception_agent, image, frame_name):
    description = perception_agent.perceive(image, frame_name)
    step_dir = os.path.join(perception_agent.save_dir, frame_name)
    frame_path = os.path.join(step_dir, "frame.jpg")
    label_path = os.path.join(step_dir, "frame.txt")
    objects = objects_from_label_file(label_path, perception_agent.class_map)

    with open(os.path.join(step_dir, "perception.txt"), "w") as f:
        f.write(description)

    return PerceptionResult(
        description=description,
        objects=objects,
        frame_path=frame_path,
    )


def objects_from_label_file(label_path, class_map):
    if not os.path.exists(label_path):
        return []

    id_to_label = {class_id: label for label, class_id in class_map.items()}
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            objects.append({"label": id_to_label.get(class_id, f"class_{class_id}")})

    return objects


def run_pipeline(args):
    controller = build_controller(args.scene_index)
    mapping_agent = MappingAgent()
    perception_agent = FlorencePerceptionAgent(
        florence_model=args.florence_model,
        sam_weights=args.sam_weights,
        save_dir=args.output_dir,
        headless=args.headless_perception,
        device=args.device,
    )
    if args.trionai:
        planner = PlanningAgent.from_tritonai(
            name="Planner",
            role="navigation planner",
            controller=controller,
            mapping_agent=mapping_agent,
            save_dir=None,
        )
        critic = CriticAgent.from_tritonai()
        action_agent = ActionAgent.from_tritonai(
            name="ActionAgent",
            role="action executor",
            controller=controller,
            planning_agent=planner,
            save_dir=args.output_dir,
        )
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(project_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Created temp folder at: {temp_dir}")

        planner = PlanningAgent(
            name="Planner",
            role="Planning",
            controller=controller,
            mapping_agent=None,
            perception_agent=perception_agent,
            save_dir=temp_dir,
        )
        critic = CriticAgent.from_openai()
        action_agent = ActionAgent(
            name="ActionAgent",
            role="User",
            controller=controller,
            planning_agent=planner,
        )

    action_history = []
    event = controller.step("Pass")  # force a fresh render before reading frames

    try:
        for step in range(args.steps):
            print(f"\n{'=' * 72}\nCycle {step + 1}/{args.steps}: observe -> map -> plan -> critique\n{'=' * 72}")

            frame_name = f"step_{step:02d}"
            step_dir = os.path.join(args.output_dir, frame_name)
            frame = Image.fromarray(event.frame)
            perception = run_perception(perception_agent, frame, frame_name)

            mapping_agent.update(
                event,
                perception_output=perception.objects,
                action=event.metadata.get("lastAction"),
            )
            map_summary = mapping_agent.get_context_string()

            plan = planner.generate_plan(
                task=args.task,
                visual_input=encode_image(perception.frame_path),
                perception_description=perception.description,
                map_summary=map_summary,
            )

            verdict = critic.review(
                task=args.task,
                proposed_subgoal=plan,
                map_summary=map_summary,
                action_history=action_history,
                perception_description=perception.description,
            )
            selected_subgoal = plan if verdict["approved"] else verdict.get("revised_subgoal") or plan

            print("\n[Perception]\n" + perception.description)
            print("\n[Map]\n" + map_summary)
            print("\n[Planner proposed]\n" + plan)
            print("\n[Critic verdict]\n" + str(verdict))
            print("\n[Selected subgoal]\n" + selected_subgoal)

            action_command = action_agent.choose_action(selected_subgoal)
            print(f"\n[ActionAgent] Executing: {action_command}")
            action_agent.perform_action(action_command)
            event = controller.last_event
            action_history.append(action_command)

            with open(os.path.join(step_dir, "pipeline_result.txt"), "w") as f:
                f.write(f"Task:\n{args.task}\n\n")
                f.write(f"Perception:\n{perception.description}\n\n")
                f.write(f"Map:\n{map_summary}\n\n")
                f.write(f"Planner proposed:\n{plan}\n\n")
                f.write(f"Critic verdict:\n{verdict}\n\n")
                f.write(f"Selected subgoal:\n{selected_subgoal}\n\n")
                f.write(f"Action executed:\n{action_command}\n")
    finally:
        controller.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current navigation agent integration pipeline.")
    parser.add_argument("--task", default="find a flower vase")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--scene-index", type=int, default=5)
    parser.add_argument("--output-dir", default="pipeline_outputs")
    parser.add_argument("--florence-model", default=FLORENCE_MODEL_ID)
    parser.add_argument("--sam-weights", default=SAM_WEIGHTS)
    parser.add_argument("--trionai", action="store_true", help="Use TritonAI-hosted models for planner and critic.")
    parser.add_argument(
        "--headless-perception",
        action="store_true",
        help="Start ai2thor_colab X server from the perception agent.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Torch device for Florence-2 and SAM2. 'auto' picks cuda > mps > cpu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

import argparse
from dataclasses import dataclass

from agents.mapping_agent import MappingAgent


@dataclass
class MockEvent:
    metadata: dict


class MockController:
    """Tiny AI2-THOR-like controller for testing agent handoffs without ProcTHOR."""

    def __init__(self):
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.last_event = self._make_event("Initialize", True)

    def step(self, action):
        if action == "GetReachablePositions":
            event = self._make_event(action, True)
            event.metadata["actionReturn"] = [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.25},
                {"x": 0.25, "y": 0.0, "z": 0.25},
                {"x": 0.25, "y": 0.0, "z": 0.0},
            ]
            return event

        if action == "MoveAhead":
            self.z = round(self.z + 0.25, 2)
        elif action == "MoveBack":
            self.z = round(self.z - 0.25, 2)
        elif action == "MoveRight":
            self.x = round(self.x + 0.25, 2)
        elif action == "MoveLeft":
            self.x = round(self.x - 0.25, 2)
        elif action == "RotateRight":
            self.yaw = (self.yaw + 90.0) % 360.0
        elif action == "RotateLeft":
            self.yaw = (self.yaw - 90.0) % 360.0

        self.last_event = self._make_event(action, True)
        return self.last_event

    def _make_event(self, last_action, success):
        return MockEvent(
            metadata={
                "agent": {
                    "position": {"x": self.x, "y": 0.0, "z": self.z},
                    "rotation": {"x": 0.0, "y": self.yaw, "z": 0.0},
                },
                "lastAction": last_action,
                "lastActionSuccess": success,
            }
        )


class MockPerceptionAgent:
    """Returns the same structured contract as PerceptionAgent.perceive(..., return_structured=True)."""

    def __init__(self):
        self.scene_objects = {
            (0.0, 0.0): [
                {"label": "Sofa", "screen_location": "in the center", "area_ratio": 0.18},
                {"label": "TV", "screen_location": "in the top-center", "area_ratio": 0.10},
            ],
            (0.0, 0.25): [
                {"label": "Table", "screen_location": "in the bottom-center", "area_ratio": 0.21},
                {"label": "Apple", "screen_location": "in the center-right", "area_ratio": 0.03},
            ],
            (0.25, 0.25): [
                {"label": "Fridge", "screen_location": "in the left-center", "area_ratio": 0.16},
            ],
        }

    def perceive(self, event, frame_name):
        position = event.metadata["agent"]["position"]
        key = (round(position["x"], 2), round(position["z"], 2))
        objects = self.scene_objects.get(key, [])

        if objects:
            lines = ["Visible Objects (Segmented):"]
            for obj in objects:
                lines.append(
                    f"- {obj['label']} located {obj['screen_location']}, "
                    f"covering roughly {obj['area_ratio']:.1%} of the view."
                )
            description = "\n".join(lines)
        else:
            description = "No distinct objects are visible in the current view."

        return {
            "description": description,
            "objects": objects,
            "frame_path": None,
            "label_path": None,
            "frame_name": frame_name,
        }


class RuleBasedPlanningAgent:
    """Planner stand-in that consumes task, perception text, and map summary."""

    def __init__(self, mapping_agent):
        self.mapping_agent = mapping_agent

    def generate_plan(self, task, perception_description, map_summary):
        target = self._extract_target(task)
        if target and target in perception_description.lower():
            return f"Stop: {target} is visible"

        if "No distinct objects" in perception_description:
            return "RotateRight"
        return "MoveAhead"

    def _extract_target(self, task):
        words_to_drop = ["find", "go", "to", "the", "a", "an", "object"]
        words = [word.strip(" .?!,").lower() for word in task.split()]
        target_words = [word for word in words if word and word not in words_to_drop]
        return " ".join(target_words) or task


class FakeLLMResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakePlannerLLM:
    """Test LLM callable for exercising PlanningAgent without an API call."""

    def __call__(self, model, input):
        text = self._first_text(input)
        lowered = text.lower()
        if "tv" in lowered and "tv:" in lowered:
            return FakeLLMResponse("Stop: already near tv")
        if "no distinct objects" in lowered:
            return FakeLLMResponse("RotateRight")
        return FakeLLMResponse("MoveAhead")

    def _first_text(self, input):
        for message in input:
            for content in message.get("content", []):
                if content.get("type") == "input_text":
                    return content.get("text", "")
        return ""


class FakePlannerClient:
    def __init__(self):
        self.responses = self
        self._llm = FakePlannerLLM()

    def create(self, model, input):
        return self._llm(model=model, input=input)


class RealPlanningAgentAdapter:
    """Uses plan.base.PlanningAgent with TritonAI."""

    def __init__(self, controller, mapping_agent):
        from agents.plan.agent import PlanningAgent

        self.planner = PlanningAgent.from_tritonai(
            name="Planner",
            role="Planning",
            controller=controller,
            mapping_agent=mapping_agent,
            save_dir=None,
        )

    def generate_plan(self, task, perception_description, map_summary):
        return self.planner.generate_plan(
            task=task,
            perception_description=perception_description,
            map_summary=map_summary,
        )


class RuleBasedCriticAgent:
    def review(self, task, proposed_subgoal, map_summary="", action_history=None, perception_description=""):
        action_history = action_history or []
        if action_history[-3:] == [proposed_subgoal] * 3:
            return {
                "approved": False,
                "reason": "The proposed action repeats too many recent steps.",
                "revised_subgoal": "RotateRight",
            }
        return {
            "approved": True,
            "reason": "The proposed action is consistent with the current map and recent history.",
            "revised_subgoal": None,
        }


def make_planner(mode, controller, mapping_agent):
    if mode == "real":
        return RealPlanningAgentAdapter(controller, mapping_agent)
    return RuleBasedPlanningAgent(mapping_agent)


def make_critic(mode):
    if mode == "real":
        from agents.critic_agent import CriticAgent

        return CriticAgent.from_tritonai()
    return RuleBasedCriticAgent()


def run_loop(task, steps, planner_mode, critic_mode, attach_semantics_to_map=False):
    controller = MockController()
    perception_agent = MockPerceptionAgent()
    mapping_agent = MappingAgent(grid_size=0.25)
    planner = make_planner(planner_mode, controller, mapping_agent)
    critic = make_critic(critic_mode)

    event = controller.last_event
    last_action = "Initialize"
    action_history = []

    print(f"Task: {task}")
    print(f"Planner: {planner_mode}")
    print(f"Critic: {critic_mode}")
    print("=" * 72)

    for step_idx in range(steps):
        mapping_agent.update(event=event, action=last_action)
        perception = perception_agent.perceive(event, frame_name=f"step_{step_idx}")
        if attach_semantics_to_map:
            mapping_agent.update(event=event, perception_output=perception["objects"], action=None)
        map_summary = mapping_agent.get_context_string()

        proposed = planner.generate_plan(
            task=task,
            perception_description=perception["description"],
            map_summary=map_summary,
        )
        verdict = critic.review(
            task=task,
            proposed_subgoal=proposed,
            map_summary=map_summary,
            action_history=action_history,
            perception_description=perception["description"],
        )
        action = proposed if verdict["approved"] else verdict["revised_subgoal"]

        print(f"\nStep {step_idx}")
        print(f"Perception:\n{perception['description']}")
        print(f"Map:\n{map_summary}")
        print(f"Planner proposed: {proposed}")
        print(f"Critic verdict: {verdict}")
        print(f"Executing: {action}")

        if isinstance(action, str) and (action == "Stop" or action.startswith("Stop:")):
            print("\nLoop finished: planner says the target is already known/current.")
            break

        event = controller.step(action)
        last_action = action
        action_history.append(action)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test simulator -> mapping and perception -> planner -> critic loop.")
    parser.add_argument("--task", default="Find TV")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--planner",
        choices=["mock", "real"],
        default="mock",
        help="Use mock planner or plan.base.PlanningAgent with TritonAI.",
    )
    parser.add_argument(
        "--critic",
        choices=["mock", "real"],
        default="mock",
        help="Use mock critic or agents.critic_agent.CriticAgent.from_tritonai().",
    )
    parser.add_argument(
        "--real-critic",
        action="store_true",
        help="Backward-compatible alias for --critic real.",
    )
    parser.add_argument(
        "--attach-semantics-to-map",
        action="store_true",
        help="Optionally annotate map nodes with perception objects; topology still comes from pose.",
    )
    args = parser.parse_args()
    critic_mode = "real" if args.real_critic else args.critic
    run_loop(
        task=args.task,
        steps=args.steps,
        planner_mode=args.planner,
        critic_mode=critic_mode,
        attach_semantics_to_map=args.attach_semantics_to_map,
    )


if __name__ == "__main__":
    main()

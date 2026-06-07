from ai2thor.controller import Controller
from openai import OpenAI
import prior
import os
import re
from PIL import Image
from dotenv import load_dotenv
import argparse
try:
    from agents.perception.agent import FlorencePerceptionAgent
except ImportError:
    FlorencePerceptionAgent = None
from agents.plan.agent import PlanningAgent
from agents.mapping_agent import MappingAgent

# Pinned to the pre-5.0-compatible revision; works with ai2thor==5.0.0 from PyPI.
PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"
TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TRITONAI_MODEL = "api-gemma-4-26b"

ACTION_TYPE_PROMPT = """You are given a description of the action that needs to be performed by the agent. \n
Action Description: {} \n
This action can broadly be classified into navigation, object interaction, and object state change.
1. Navigation: Actions that involve moving the agent through the environment, such as "MoveAhead", "RotateLeft", etc.
2. Object Interaction: Actions that involve interacting with objects in the environment, such as "PickupObject", "PushObject", etc.
3. Object State Change: Actions that involve changing the state of objects, such as "OpenObject", "CleanObject", etc.

Based on the description of the action, classify it into one of the three categories mentioned above and return the category name as the output.
Your response should be in the following format:\n
Action Type: [Category Name]
Explanation: [A brief explanation of why you classified the action into this category, referencing specific keywords or aspects of the action description that led to your decision.]
Where [Category Name] is one of "Navigation", "Object Interaction", or "Object State Change".\n 
"""

ACTION_CHOOSE_PROMPT = """
The agent's goal is: {}.
Possible actions: {}.
Available objects with their IDs and locations: {}
Visible objects with their IDs and locations: {}

Note that you can interact with an object only if it is visible. If you can not interact with the object, return a message saying "Object not visible" instead of a controller command.

Note that the action you return MUST be one of the possible actions listed above, and if it requires an objectId, you should use one from the available objects list. The agent will execute exactly the command you return, so it must be a valid controller.step(...) call that can be executed in Python.

IMPORTANT — exact API names only. The action name passed to controller.step(action=...) must match one of the listed actions EXACTLY, character-for-character. Do NOT substitute English variants. Specifically:
- Use 'RotateLeft' / 'RotateRight', NOT 'TurnLeft' / 'TurnRight'.
- Use 'MoveAhead', NOT 'MoveForward' / 'GoForward' / 'WalkForward'.
- Use 'MoveBack', NOT 'MoveBackward' / 'StepBack'.
Invalid action names crash the simulator and waste a cycle.

For navigation actions use: controller.step(action='MoveAhead', moveMagnitude=<magnitude>) or any other action from the navigation category.
For object interactions use the exact objectId from the list above: controller.step(action='OpenObject', objectId='Fridge|1|2|3')

IMPORTANT: Always use y=0.9 for the Teleport action. Using y=0.0 will fail silently.

ROTATION INCREMENT: The default rotation step is 90 degrees, which over-corrects for a target that is only slightly off-center. Prefer fine rotations: controller.step(action='RotateLeft', degrees=30) or controller.step(action='RotateRight', degrees=30). Use larger degrees (60-90) only when you need to scan/search for an out-of-view target, not when correcting heading toward a target already visible in the frame.
"""

class ActionAgent:
    def __init__(self,
                 name,
                 role,
                 controller: Controller,
                 planning_agent: PlanningAgent = None,
                 save_dir: str = None,
                 in_context_example: bool = True,
                 client=None,
                 model=DEFAULT_TRITONAI_MODEL):
        self.name = name
        self.role = role
        self.controller = controller
        self.planning_agent = planning_agent
        self.model = model
        self.in_context_example = in_context_example

        if client is not None:
            self.client = client
        else:
            load_dotenv(self._repo_env_path())
            tritonai_key = os.environ.get("TRITONAI_API_KEY")
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not fallback_key:
                with open("key.txt", "r") as f:
                    fallback_key = f.read().strip()
            if tritonai_key:
                self.client = OpenAI(base_url=TRITONAI_BASE_URL, api_key=tritonai_key)
            else:
                self.client = OpenAI(api_key=openai_key)

        self.LLM = self.client.responses.create

        if save_dir is not None:
            self.save_dir = save_dir
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_dir = os.path.join(project_dir, "..", "..", "action_outputs")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.navigation = [
            "MoveAhead",
            "MoveBack",
            "MoveLeft",
            "MoveRight",
            "RotateLeft",
            "RotateRight",
            "LookUp",
            "LookDown",
            "Crouch",
            "Stand",
            "Teleport",
        ]
        
        self.object_interaction = [
            "PickupObject",
            "PutObject",
            "DropHandObject",
            "ThrowObject",
            "MoveHeldObjectAhead",
            "MoveHeldObjectBack",
            "MoveHeldObjectLeft",
            "MoveHeldObjectRight",
            "MoveHeldObjectUp",
            "MoveHeldObjectDown",
            "PushObject",
            "PullObject",
            "TouchThenApplyForce",
            "PlaceObjectAtPoint",
            "GetSpawnCoordinatesAboveReceptacle"
        ]
        
        self.object_state_change = [
            "OpenObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "DirtyObject",
            "CleanObject",
            "FillObjectWithLiquid",
            "SliceObject",
            "CleanObject",
            "EmptyLiquidFromObject",
            "UseUpObject",
        ]
        
    @staticmethod
    def _repo_env_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    @classmethod
    def from_tritonai(
        cls,
        name,
        role,
        controller: Controller,
        planning_agent: PlanningAgent = None,
        save_dir: str = None,
        in_context_example: bool = True,
        model=DEFAULT_TRITONAI_MODEL,
        api_key=None,
    ):
        if OpenAI is None:
            raise RuntimeError("The openai package is required for TritonAI planner calls.")

        load_dotenv(cls._repo_env_path())
        key = api_key or os.environ.get("TRITONAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "TritonAI API key not found. Set TRITONAI_API_KEY (or OPENAI_API_KEY) in .env."
            )

        client = OpenAI(base_url=TRITONAI_BASE_URL, api_key=key)
        return cls(
            name=name,
            role=role,
            controller=controller,
            planning_agent=planning_agent,
            save_dir=save_dir,
            in_context_example=in_context_example,
            client=client,
            model=model,
        )

    def __call__(self, text_input):
        response = self.LLM(model=self.model, input=[{
            "role": "user",
            "content": [
                { "type": "input_text", "text": text_input },

            ],
        }
        ],)
        response = response.output_text
        return response
    
    def get_current_visual_output(self, frame_id=None):
        curr = Image.fromarray(self.controller.last_event.frame)
        
        if self.save_dir:
            if frame_id is not None:
                frame_path = os.path.join(self.save_dir, f"visual_output_{frame_id}.jpg")
            else:
                frame_path = os.path.join(self.save_dir, "visual_output.jpg")
            curr.save(frame_path)
            print(f"Visual output saved to {frame_path}")
            return frame_path
        return None
    
    def generate_incontext_example(self):
       
        incontext_example = {"Navigation": 
                                {"action_description": "Move towards the table on the right",
                                "possible_actions": self.navigation,
                                "check_visibility": "No need to check visibility for navigation actions",
                                "action_command": "controller.step(action='MoveRight')"},
                             
                             "Object Interaction": 
                                {"action_description": "Pick up the apple on the table",
                                "possible_actions": self.object_interaction,
                                "check_visibility": "Check if the apple is visible in the current view. If it is not visible, return 'Object not visible'.",
                                "action_command": "controller.step(action='PickupObject', objectId='Apple|1|2|3')"},
                                
                             "Object State Change": 
                                {"action_description": "Open the fridge",
                                "possible_actions": self.object_state_change,
                                "check_visibility": "Check if the fridge is visible in the current view. If it is not visible, return 'Object not visible'.",
                                "action_command": "controller.step(action='OpenObject', objectId='Fridge|1|2|3')"}
                            }
        
        return incontext_example
        
    def choose_action_type(self, action_description):
        response = self.__call__(ACTION_TYPE_PROMPT.format(action_description))
        text = response.strip()
        
        print(f"Action type classification response:\n{text}\n")

        # Extract the category that appears after the literal "Action Type" label
        m = re.search(r"Action\s*Type\s*[:\-]\s*['\"]?\s*([^\n\r]+)", text, re.IGNORECASE)
        if m:
            cat = m.group(1).strip().strip('"').strip("'")
            # Normalize common variants to the expected category names
            if re.search(r"navigation", cat, re.IGNORECASE):
                return "Navigation"
            if re.search(r"object\s*interaction", cat, re.IGNORECASE):
                return "Object Interaction"
            if re.search(r"object\s*state|state\s*change", cat, re.IGNORECASE):
                return "Object State Change"
            # If it doesn't match expected labels, return the raw captured text
            return cat

        # If label not found, default to Navigation
        return "Navigation"
    
    def choose_action(self, task_description, failure_msg = None, perception_description=None):
        action_description = self.planning_agent.generate_plan(task_description)
        # The inner re-plan above is called with task_description only — no
        # perception, no map. Any "turn"/"rotate" verbs it emits are uninformed
        # and were observed biasing the action LLM into rotations even when
        # perception said target was top-center. Neutralize those verbs so the
        # action LLM chooses Rotate-vs-Move purely from the perception block.
        def _neutralize(m):
            return "Move" if m.group(0)[0].isupper() else "move"
        action_description = re.sub(
            r"\b(?:turn|rotate)\b",
            _neutralize,
            action_description,
            flags=re.IGNORECASE,
        )
        print(f"Generated plan (neutralized): {action_description}")
        type = self.choose_action_type(action_description)
        if type == "Navigation":
            possible_actions = self.navigation
        elif type == "Object Interaction":
            possible_actions = self.object_interaction
        else:            
            possible_actions = self.object_state_change    
        
        # Pass objectId alongside type so the LLM can use real IDs
        available_objects = [
            {"objectType": obj["objectType"], "objectId": obj["objectId"], "Location": (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])}
            for obj in self.controller.last_event.metadata["objects"]
        ]
        
        visible_objects = [
            {"objectType": obj["objectType"], "objectId": obj["objectId"], "Location": (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])} for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] 
        ]

        action_choose_input = ACTION_CHOOSE_PROMPT.format(
                                action_description,
                                possible_actions,
                                available_objects,
                                visible_objects
                            )
        
        if perception_description:
            action_choose_input += (
                "\n\nCurrent visual perception (camera-frame spatial language). "
                "PERCEPTION IS AUTHORITATIVE for choosing direction: if the "
                "plan's prose contains directional words ('to the left', 'on "
                "the right side', etc.) that conflict with the perception "
                "label below, TRUST THE PERCEPTION LABEL. The plan describes "
                "intent (move toward X); perception tells you where X actually "
                "is in the camera frame right now.\n"
                "Spatial label -> action mapping for the target object:\n"
                "  - ANY label containing 'center' ('top-center', 'center', "
                "'center-center', 'bottom-center', 'center-left', "
                "'center-right'): target is in the central column of the "
                "view, approximately ahead -> MoveAhead. Do NOT rotate to "
                "fine-align a target whose bbox is in the central column; at "
                "close range a small lateral bbox offset from optical-axis "
                "center is normal and a 30deg rotation will over-shoot to "
                "the opposite side, causing a stuck rotate-left/rotate-right "
                "oscillation.\n"
                "  - 'top-left' (peripheral, not central column): use a SMALL "
                "rotation: controller.step(action='RotateLeft', degrees=30)\n"
                "  - 'top-right' (peripheral, not central column): use a SMALL "
                "rotation: controller.step(action='RotateRight', degrees=30)\n"
                "  - 'bottom-left' / 'bottom-right': target is below and to the "
                "side; usually MoveAhead first, then rotate as it shifts up\n"
                "If perception does NOT list the target at all, the target is "
                "OUT OF VIEW — use a LARGER rotation to scan/search: "
                "controller.step(action='RotateLeft', degrees=90) or "
                "controller.step(action='RotateRight', degrees=90).\n"
                "Rationale: small rotations (~30 deg) when target is visible "
                "let the agent approximate a smooth curved path toward the "
                "target; large rotations (~90 deg) are only for searching "
                "when the target is lost.\n\n"
                f"{perception_description}\n"
            )

        if self.in_context_example:
            example = self.generate_incontext_example()
            action_choose_input += "\n\nHere are some examples of how to choose actions based on the type of action description:\n"
            for cat, ex in example.items():
                action_choose_input += f"\nCategory: {cat}\nAction Description: {ex['action_description']}\nPossible Actions: {ex['possible_actions']}\nVisibility Check: {ex['check_visibility']}\nExample Action Command: {ex['action_command']}\n"

        if failure_msg:
            action_choose_input += f"\n\nPrevious action failed with message: {failure_msg}\nUse this information to inform your next action choice.\n"
            
        response = self.__call__(action_choose_input)
        response = response.strip()

        # Strip tool-call wrapper tokens that some LLMs (e.g. Gemma) emit
        # instead of clean text. Observed:
        #   <|tool_call>call:controller.step(action='RotateLeft')<tool_call|>
        # Without this, the parser below sees a line starting with
        # `<|tool_call>` (not `controller.step(`), fails, and the agent
        # silently defaults to MoveAhead — masking correct rotation commands.
        response = response.replace("<|tool_call>", "").replace("<tool_call|>", "")
        response = re.sub(r"\bcall:", "", response)
        response = response.strip()

        # Extract a complete controller.step(...) call, which may span multiple lines
        def extract_step_call(text):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip().lstrip("`").rstrip("`").strip()
                if stripped.startswith("controller.step("):
                    collected = stripped
                    depth = collected.count("(") - collected.count(")")
                    j = i + 1
                    while depth > 0 and j < len(lines):
                        next_line = lines[j].strip().rstrip("`").strip()
                        collected += "\n" + next_line
                        depth += next_line.count("(") - next_line.count(")")
                        j += 1
                    return collected
            return None

        command = extract_step_call(response)
        if command:
            return command

        # Fallback: strip markdown code block and retry
        if "```" in response:
            inner = response.split("```")[1]
            if inner.startswith("python"):
                inner = inner[6:]
            command = extract_step_call(inner)
            if command:
                return command

        print(f"[ActionAgent] WARNING: could not extract controller.step() from response, defaulting to MoveAhead.\nRaw response: {response}")
        return "controller.step(action='MoveAhead')"
    
    def perform_action(self, action_command):
        try:
            exec(action_command, {"controller": self.controller})
            event = self.controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                print(f"Action failed: {event.metadata.get('errorMessage', 'unknown error')}")
                return False, event.metadata.get("errorMessage", "unknown error")
            return True, None
        except StopIteration:
            print(f"Error: Object not found in scene. Available objects: {[obj['objectType'] for obj in self.controller.last_event.metadata['objects']]}")
            raise
        except (ValueError, KeyError) as e:
            # The action LLM occasionally hallucinates English-y variants
            # like 'TurnLeft' / 'MoveForward' that AI2-THOR rejects with
            # ValueError before lastActionSuccess is ever set. Treat these
            # as a failed action so the pipeline's [FAILED] tagging,
            # intra-cycle retry with failure_msg, and wedge guard handle
            # recovery — rather than crashing the whole run.
            print(f"Invalid action command: {e}")
            return False, f"Invalid action: {e}"
        except Exception as e:
            print(f"Error executing action: {e}")
            raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the action agent with optional TritonAI models.")
    parser.add_argument("--tritonai", action="store_true", help="Use TritonAI-hosted models for planner and critic.")
    parser.add_argument("--task", type=str, default="Find the table and move towards it.", help="The task description for the agent.")
    args = parser.parse_args()
    
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][280]
    controller = Controller(scene=house)
    
    task = args.task
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "..", "..", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temp folder at: {temp_dir}")
    
    # Optional testing wrapper if running script locally:
    perception_module = None
    if FlorencePerceptionAgent is not None:
        print("Instantiating FlorencePerceptionAgent...")
        perception_module = FlorencePerceptionAgent(
            florence_model="microsoft/Florence-2-base",
            sam_weights="sam2_b.pt",
            save_dir=os.path.join(temp_dir, "perception_logs"),
            headless=True
        )

    if not args.tritonai:
        planning_agent = PlanningAgent(
            name="Planner",
            role="Planning",
            controller=controller,
            mapping_agent=None,
            perception_agent=perception_module,
            save_dir=temp_dir
        )
        agent = ActionAgent(name="ActionAgent", role="User", controller=controller, planning_agent=planning_agent)
    
    else:
        planner = PlanningAgent.from_tritonai(
                name="Planner",
                role="navigation planner",
                controller=controller,
                mapping_agent=None,
                perception_agent=perception_module,
                save_dir=temp_dir,
            )
        agent = ActionAgent.from_tritonai(
                name="ActionAgent",
                role="action executor",
                controller=controller,
                planning_agent=planner,
                in_context_example=False,
            )
    
    done = False
    
    while not done:
        action_command = agent.choose_action(task)
        print(f"Action Command: {action_command}")
        
        agent_meta = controller.last_event.metadata["agent"]
        start = (round(agent_meta["position"]["x"], 2),
                    round(agent_meta["position"]["z"], 2))
        
        print(f"Agent starting position (x, z): {start}")
        
        done = agent.perform_action(action_command)
        
        agent_meta = controller.last_event.metadata["agent"]
        final = (round(agent_meta["position"]["x"], 2),
                    round(agent_meta["position"]["z"], 2))
        
        print(f"Agent final position (x, z): {final}")
        agent.get_current_visual_output()
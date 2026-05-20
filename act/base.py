from ai2thor.controller import Controller
from openai import OpenAI
import prior
import os
from PIL import Image
from dotenv import load_dotenv
try:
    from agents.perception.agent import FlorencePerceptionAgent
except ImportError:
    FlorencePerceptionAgent = None
from plan.base import PlanningAgent
from agents.mapping_agent import MappingAgent

with open("key.txt", "r") as f:
    key = f.read().strip()

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
Your response should be only one of the following: "Navigation", "Object Interaction", or "Object State Change". Make sure to choose the category that best fits the action description provided.
"""

ACTION_CHOOSE_PROMPT ="""
The goal of the agent is to perform {}. The possible actions that can be taken are as follows: {}. \n
Available objects in the scene: {}

Based on the goal of the agent return the most appropriate action to take from the list of possible actions. Make sure the action you choose directly contributes to accomplishing the goal.

If there is object interacttion involved, identify the object and obtain the object ID from the object metadata. The object metadata can be obtained using controller.last_event.metadata["objects"]. 

Use the object ID in the command. For example,
controller.step(
    action="PickupObject",
    objectId="Apple|1|1|1",
    forceAction=False,
    manualInteract=False
)
\nMake sure to replace object_id with the actual ID of the object you want to interact with.

You will be using the Controller from ai2thor to perform the actions. Return the command to take the given action using the Controller. For example, if the action is "MoveAhead", you would return "controller.step(action='MoveAhead')". Make sure to replace "MoveAhead" with the actual action you choose.
"""

class ActionAgent:
    def __init__(self, 
                 name, 
                 role, 
                 controller: Controller, 
                 planning_agent: PlanningAgent = None):
        self.name = name
        self.role = role
        self.controller = controller
        self.planning_agent = planning_agent

        self.client = OpenAI(api_key=key)

        self.LLM = self.client.responses.create
        
        project_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(project_dir, "action_outputs")
        os.makedirs(temp_dir, exist_ok=True)
        self.save_dir = temp_dir
        
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
        
    @classmethod
    def from_tritonai(
        cls,
        name,
        role,
        controller: Controller,
        planning_agent: PlanningAgent = None,
        save_dir: str = None,
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
            client=client,
            model=model,
        )

    def __call__(self, text_input):
        response = self.LLM(model="gpt-5.5", input=[{
            "role": "user",
            "content": [
                { "type": "input_text", "text": text_input },

            ],
        }
        ],)
        response = response.output_text
        return response
    
    def get_current_visual_output(self):
        curr = Image.fromarray(self.controller.last_event.frame)
        
        if self.save_dir:
            frame_path = os.path.join(self.save_dir, "visual_output.jpg")
            curr.save(frame_path)
            print(f"Visual output saved to {frame_path}")
            return frame_path
        return None
        
    def choose_action_type(self, action_description):
        response = self.__call__(ACTION_TYPE_PROMPT.format(action_description))
        return response.strip()
    
    def choose_action(self, task_description):
        action_description = self.planning_agent.generate_plan(task_description)
        print(f"Generated plan: {action_description}")
        type = self.choose_action_type(action_description)
        if type == "Navigation":
            possible_actions = self.navigation
        elif type == "Object Interaction":
            possible_actions = self.object_interaction
        else:            
            possible_actions = self.object_state_change    
        
        # Get available objects in the scene
        available_objects = [obj["objectType"] for obj in self.controller.last_event.metadata["objects"]]
        available_objects = list(set(available_objects))  # Remove duplicates
        
        response = self.__call__(ACTION_CHOOSE_PROMPT.format(action_description, possible_actions, available_objects))
        response = response.strip()
        
        # Remove markdown code block formatting if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("python"):
                response = response[6:]  # Remove "python" language specifier
        response = response.strip()
        
        return response
    
    def perform_action(self, action_command):
        try:
            exec(action_command, {"controller": self.controller})
        except StopIteration:
            print(f"Error: Object not found in scene. Available objects: {[obj['objectType'] for obj in self.controller.last_event.metadata['objects']]}")
            raise
        except Exception as e:
            print(f"Error executing action: {e}")
            raise
    
if __name__ == "__main__":
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][5]
    controller = Controller(scene=house)
    
    task = "Find the table and move towards it."
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp")
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

    planning_agent = PlanningAgent(
        name="Planner",
        role="Planning",
        controller=controller,
        mapping_agent=None,
        perception_agent=perception_module,
        save_dir=temp_dir
    )
    
    agent = ActionAgent(name="ActionAgent", role="User", controller=controller, planning_agent=planning_agent)
    action_command = agent.choose_action(task)
    print(f"Action Command: {action_command}")
    agent.perform_action(action_command)
    agent.get_current_visual_output()
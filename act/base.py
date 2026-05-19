from ai2thor.controller import Controller
from openai import OpenAI
import prior

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
    def __init__(self, name, role, controller: Controller):
        self.name = name
        self.role = role
        self.controller = controller
        
        self.client = OpenAI(api_key="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")

        self.LLM = self.client.responses.create
        
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
        
    def choose_action_type(self, action_description):
        response = self.__call__(ACTION_TYPE_PROMPT.format(action_description))
        return response.strip()
    
    def choose_action(self, action_description):
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
    house = dataset["train"][0]
    controller = Controller(scene=house)
    
    action = "Move towards the wall"
    agent = ActionAgent(name="ActionAgent", role="User", controller=controller)
    action_command = agent.choose_action(action)
    print(f"Action Command: {action_command}")
    agent.perform_action(action_command)
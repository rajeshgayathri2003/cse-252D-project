import copy

from openai import OpenAI
from ai2thor.controller import Controller
from PIL import Image
import networkx as nx
import prior

PROMPT = """
You are a helpful planning agent. Your role is to decide the action to be taken based on the task provided, visual input, map and top down view of the current location.
Here is a detailed description of the inputs you will receive:
1. Task: A natural language description of the task that needs to be accomplished. For example, "Go to the kitchen and pick up the apple on the table."
2. Visual Input: A description of the current visual scene from the agent's perspective. This may include objects in view, their locations, and any relevant details. For example, "You see a table with an apple on it, a chair, and a door leading to the kitchen."
3. Graph Representation: A graph representation of the environment, where nodes represent reachable positions and edges represent possible movements between those positions. 
"""

class PlanningAgent:
    def __init__(self, name, role, controller: Controller):
        self.name = name
        self.role = role
        self.controller = controller

        self.client = OpenAI()
        
        self.LLM = self.client.responses.create
        
        self.task_terminate = "Call to critic"


    def get_reachable_positions(self):
        reachable = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        return reachable
    
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        return Image.fromarray(top_down_frame)
    
    def create_graph(self, reachable_positions):
        reachable = self.get_reachable_positions()
        G = nx.Graph()

        STEP = 0.25  # AI2-THOR default grid size

        # Add a node for every reachable position
        for pos in reachable:
            node = (round(pos["x"], 2), round(pos["z"], 2))   # use (x, z) — y is height
            G.add_node(node, y=pos["y"])

        # Connect nodes that are exactly one grid step apart (4-connectivity)
        nodes = list(G.nodes)
        node_set = set(nodes)

        for (x, z) in nodes:
            for dx, dz in [(STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)]:
                neighbor = (round(x + dx, 2), round(z + dz, 2))
                if neighbor in node_set:
                    G.add_edge((x, z), neighbor)

        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        
    def __call__(self, input):
        response = self.LLM(model="gpt-5.5", input=input)
        response = response.output_text
        return response
    
    def generate_plan(self, task, visual_input, map_view, top_down_view):
        # Generate a plan based on the task and the provided inputs
        plan = self.__call__(f"{PROMPT}\nTask: {task}\nVisual Input: {visual_input}\nGraph View: {map_view}\nTop Down View: {top_down_view}")
        return plan
    
        
    def should_terminate(self, task):
        return self.task_terminate(task)["completed"]
    

if __name__ == "__main__":
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][0]
    controller = Controller(scene=house)
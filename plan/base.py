import base64
import copy

from openai import OpenAI
from ai2thor.controller import Controller
from PIL import Image
import networkx as nx
import prior
import os
import matplotlib.pyplot as plt
from agents.mapping_agent import MappingAgent

# Pinned to the pre-5.0-compatible revision; works with ai2thor==5.0.0 from PyPI.
PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"

PROMPT = """
You are a helpful planning agent. Your role is to decide the action to be taken based on the task provided, visual input, map and top down view of the current location.
Here is a detailed description of the inputs you will receive:
1. Task: A natural language description of the task that needs to be accomplished. For example, "Go to the kitchen and pick up the apple on the table."
2. Visual Input: A description of the current visual scene from the agent's perspective. This may include objects in view, their locations, and any relevant details. For example, "You see a table with an apple on it, a chair, and a door leading to the kitchen."
3. Map Summary: The Mapping Agent's spatial memory, including visited nodes, known objects, and current location.
4. Graph Representation: A graph representation of the environment, where nodes represent reachable positions and edges represent possible movements between those positions. 

Return a clear instruction for the next action to take, such as "Move forward", "Turn left" etc. Make sure the instruction is actionable and directly contributes to accomplishing the task.
"""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class PlanningAgent:
    def __init__(self, name, role, controller: Controller, mapping_agent: MappingAgent = None, save_dir: str = None):
        self.name = name
        self.role = role
        self.controller = controller
        self.mapping_agent = mapping_agent or MappingAgent()
        self.save_dir = save_dir

        self.client = OpenAI(api_key="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")

        self.LLM = self.client.responses.create

        self.task_terminate = "Call to critic"


    def get_reachable_positions(self):
        reachable = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        return reachable
    
    def get_current_visual_input(self):
        curr = Image.fromarray(self.controller.last_event.frame)
        
        if self.save_dir:
            frame_path = os.path.join(self.save_dir, "visual_input.jpg")
            curr.save(frame_path)
            print(f"Visual input saved to {frame_path}")
        
        return frame_path
    
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
        frame_image = Image.fromarray(top_down_frame)
        
        # Save the frame if save_dir is provided
        if self.save_dir:
            frame_path = os.path.join(self.save_dir, "top_down.jpg")
            frame_image.save(frame_path)
            print(f"Top-down view saved to {frame_path}")
        
        return frame_path
    
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
        
        # Save the graph if save_dir is provided
        if self.save_dir:
            return self._save_graph(G)

        return None
    
    def _save_graph(self, G):
        """Save the graph visualization to a JPG file."""
        pos_2d = {node: node for node in G.nodes}   # (x, z) doubles as plot coords
        
        plt.figure(figsize=(10, 10))
        nx.draw(
            G, pos=pos_2d,
            node_size=30, node_color="steelblue",
            edge_color="gray", width=0.8, with_labels=False
        )
        
        # Mark the agent's starting position
        agent_meta = self.controller.last_event.metadata["agent"]
        start = (round(agent_meta["position"]["x"], 2),
                 round(agent_meta["position"]["z"], 2))
        
        nx.draw_networkx_nodes(G, pos=pos_2d, nodelist=[start],
                               node_color="red", node_size=120)
        
        plt.title("Reachable Positions Graph")
        plt.axis("equal")
        plt.tight_layout()
        
        graph_path = os.path.join(self.save_dir, "reachable_pos_graph.jpg")
        plt.savefig(graph_path, dpi=150, format="jpg")
        plt.close()
        print(f"Graph saved to {graph_path}")
        
        return graph_path
        
        
    def __call__(self, text_input, visual_input=None, graph_view=None, top_down_view=None):
        content = [{"type": "input_text", "text": text_input}]

        if visual_input:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{visual_input}",
                }
            )

        if graph_view:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{graph_view}",
                }
            )

        if top_down_view:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{top_down_view}",
                }
            )

        response = self.LLM(model="gpt-5.5", input=[{
            "role": "user",
            "content": content,
        }
        ],)
        response = response.output_text
        return response
    
    def generate_plan(self, task, visual_input=None, perception_description=None, map_summary=None):
        map_summary = map_summary or self.mapping_agent.get_context_string()
        
        if perception_description is None:
            perception_description = "No perception description was provided."

        if visual_input is None and self.save_dir:
            visual_input_path = self.get_current_visual_input()
            visual_input = encode_image(visual_input_path)
        
        graph_path = self.create_graph(self.get_reachable_positions())
        graph_view = encode_image(graph_path) if graph_path else None

        top_down_frame_path = self.get_top_down_frame() if self.save_dir else None
        top_down_view = encode_image(top_down_frame_path) if top_down_frame_path else None
        
        planner_input = (
            f"{PROMPT}\n\n"
            f"Task:\n{task}\n\n"
            f"Visual Input:\n{perception_description}\n\n"
            f"Map Summary:\n{map_summary}\n"
        )
        plan = self.__call__(planner_input, visual_input, graph_view, top_down_view)
        return plan
    
        
    def should_terminate(self, task):
        return self.task_terminate(task)["completed"]
    

if __name__ == "__main__":
    dataset = prior.load_dataset("procthor-10k", revision=PROCTHOR_REVISION)
    house = dataset["train"][0]
    controller = Controller(scene=house)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temp folder at: {temp_dir}")
    
    planning_agent = PlanningAgent(
        name="Planner",
        role="Planning",
        controller=controller,
        save_dir=temp_dir
    )
    
    plan = planning_agent.generate_plan("Find a flower vase")
    print("Generated Plan:", plan)

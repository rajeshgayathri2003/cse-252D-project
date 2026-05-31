import base64
import copy
import os

from ai2thor.controller import Controller
from dotenv import load_dotenv
from PIL import Image
import networkx as nx
import prior
import matplotlib.pyplot as plt
import argparse

from agents.mapping_agent import MappingAgent

# Import the perception agent. Wrapping in a try/except to maintain modularity.
try:
    from perception.agent import FlorencePerceptionAgent
except ImportError:
    FlorencePerceptionAgent = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

# Pinned to the pre-5.0-compatible revision; works with ai2thor==5.0.0 from PyPI.
PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"
TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TRITONAI_MODEL = "api-gemma-4-26b"

PROMPT = """
You are a helpful planning agent. Your role is to decide the action to be taken based on the task provided, visual input, map and top down view of the current location.
Here is a detailed description of the inputs you will receive:
1. Task: A natural language description of the task that needs to be accomplished. For example, "Go to the kitchen and pick up the apple on the table."
2. Visual Input: A description of the current visual scene from the agent's perspective. This may include objects in view, their locations, and any relevant details. For example, "You see a table with an apple on it, a chair, and a door leading to the kitchen."
3. Map Summary: The Mapping Agent's spatial memory, including visited nodes, known objects, and current location.
4. Graph Representation: A graph representation of the environment, where nodes represent reachable positions and edges represent possible movements between those positions. 

Give your output in the following format:
Plan: <Your plan here based on the task and inputs>
"""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
#read key
with open("key.txt", "r") as f:
    key = f.read().strip()

class PlanningAgent:
    def __init__(
        self,
        name,
        role,
        controller: Controller,
        mapping_agent: MappingAgent = None,
        perception_agent: FlorencePerceptionAgent = None,
        save_dir: str = None,
        incontext_dir: str = None,
        use_incontext_example: bool = True,
        client=None,
        model=DEFAULT_OPENAI_MODEL,
    ):
        self.name = name
        self.role = role
        self.controller = controller
        self.mapping_agent = mapping_agent or MappingAgent()
        self.perception_agent = perception_agent
        self.save_dir = save_dir
        self.incontext_dir = incontext_dir
        self.use_incontext_example = use_incontext_example
        self.model = model
        
        # Tracks steps to pass unique sequential frame identifiers to the perception agent
        self.step_counter = 0

        if client is not None:
            self.client = client
        elif OpenAI is not None:
            self.client = OpenAI(api_key=key)
        else:
            self.client = None

        self.LLM = self.client.responses.create if self.client is not None else None
        self.task_terminate = "Call to critic"

    @classmethod
    def from_tritonai(
        cls,
        name,
        role,
        controller: Controller,
        mapping_agent: MappingAgent = None,
        perception_agent: FlorencePerceptionAgent = None,
        save_dir: str = None,
        incontext_dir: str = None,
        use_incontext_example: bool = True,
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
            mapping_agent=mapping_agent,
            perception_agent=perception_agent,
            save_dir=save_dir,
            incontext_dir=incontext_dir,
            use_incontext_example=use_incontext_example,
            client=client,
            model=model,
        )

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
        return None
    
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 90
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # Add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        frame_image = Image.fromarray(top_down_frame)
        if frame_image.mode == "RGBA":
            frame_image = frame_image.convert("RGB")

        if self.save_dir:
            frame_path = os.path.join(self.save_dir, "top_down.jpg")
            frame_image.save(frame_path)
            print(f"Top-down view saved to {frame_path}")
            return frame_path
        
        return None
    
    def create_graph(self, reachable_positions):
        if not reachable_positions:
            return None
        reachable = reachable_positions
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

        if self.LLM is None:
            raise RuntimeError("PlanningAgent needs an OpenAI client or a test LLM callable.")

        return self._create_response(content)

    def _create_response(self, content):
        chat_content = []
        for item in content:
            if item.get("type") == "input_text":
                chat_content.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "input_image":
                chat_content.append({
                    "type": "image_url",
                    "image_url": {"url": item["image_url"]},
                })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": chat_content}],
        )
        return response.choices[0].message.content

    def _format_incontext_prompt_block(self, incontext_example):
        if not incontext_example:
            return ""

        warnings = incontext_example.get("warnings") or []
        warning_text = "None" if not warnings else "; ".join(warnings)

        return (
            "In-Context Example:\n"
            f"Example Task:\n{incontext_example['example_task']}\n\n"
            f"Example Visual Input:\n{incontext_example['example_perception']}\n\n"
            f"Example Map Summary:\n{incontext_example['example_map_summary']}\n\n"
            f"Expected Plan:\n{incontext_example['example_plan']}\n\n"
            f"Example Asset Warnings:\n{warning_text}\n"
        )
    
    def generate_incontext_example(self):
        warnings = []
        if not self.incontext_dir:
            warnings.append("incontext_dir is not configured.")
            return {
                "example_task": "Find the blue trash can in this scene",
                "example_perception": "No in-context visual assets were provided.",
                "example_map_summary": "No in-context map summary available.",
                "example_plan": "Move ahead towards the blue trash can located in the middle of the room",
                "visual_input": None,
                "graph_view": None,
                "top_down_view": None,
                "warnings": warnings,
            }

        visual_input_path = os.path.join(self.incontext_dir, "visual_input.jpg")
        graph_view_path = os.path.join(self.incontext_dir, "reachable_pos_graph.jpg")
        top_down_view_path = os.path.join(self.incontext_dir, "top_down.jpg")

        visual_input = None
        graph_view = None
        top_down_view = None

        if os.path.exists(visual_input_path):
            visual_input = encode_image(visual_input_path)
        else:
            warnings.append(f"Missing in-context visual input: {visual_input_path}")

        if os.path.exists(graph_view_path):
            graph_view = encode_image(graph_view_path)
        else:
            warnings.append(f"Missing in-context graph view: {graph_view_path}")

        if os.path.exists(top_down_view_path):
            top_down_view = encode_image(top_down_view_path)
        else:
            warnings.append(f"Missing in-context top-down view: {top_down_view_path}")

        perception_description = "No in-context perception description available."
        if visual_input and self.perception_agent:
            frame_identity = "incontext_example"
            with Image.open(visual_input_path) as pil_image:
                perception_description = self.perception_agent.perceive(
                    image=pil_image,
                    frame_name=frame_identity,
                )
        elif visual_input:
            perception_description = "In-context image is available, but no perception agent is configured."
        else:
            warnings.append("No in-context visual input available to run perception.")

        return {
            "example_task": "Find the blue trash can in this scene",
            "example_perception": perception_description,
            "example_map_summary": "Reachable map is shown via graph and top-down assets.",
            "example_plan": "Move ahead towards the blue trash can located in the middle of the room",
            "visual_input": visual_input,
            "graph_view": graph_view,
            "top_down_view": top_down_view,
            "warnings": warnings,
        }

    def generate_plan(self, task, visual_input=None, perception_description=None, map_summary=None, critic_feedback=None):
        map_summary = map_summary or self.mapping_agent.get_context_string()
        
        # Convert simulator RGB frame to PIL for the perception agent
        raw_frame_array = self.controller.last_event.frame
        pil_image = Image.fromarray(raw_frame_array)
        
        if visual_input is None and self.save_dir:
            visual_input_path = self.get_current_visual_input()
            if visual_input_path:
                visual_input = encode_image(visual_input_path)
        
        # Intercept and fetch open-vocabulary + SAM segmentations from FlorencePerceptionAgent
        if perception_description is None:
            if self.perception_agent is not None:
                frame_identity = f"frame_{self.step_counter}"
                perception_description = self.perception_agent.perceive(
                    image=pil_image, 
                    frame_name=frame_identity
                )
                self.step_counter += 1
            else:
                perception_description = "No perception description was provided."

        graph_path = self.create_graph(self.get_reachable_positions()) if self.save_dir else None
        graph_view = encode_image(graph_path) if graph_path else None

        top_down_frame_path = self.get_top_down_frame() if self.save_dir else None
        top_down_view = encode_image(top_down_frame_path) if top_down_frame_path else None

        incontext_block = ""
        if self.use_incontext_example:
            incontext_example = self.generate_incontext_example()
            incontext_block = self._format_incontext_prompt_block(incontext_example)
        
        
        planner_input = (
            f"{PROMPT}\n\n"
            f"{incontext_block}\n"
            f"Task:\n{task}\n\n"
            f"Visual Input:\n{perception_description}\n\n"
            f"Map Summary:\n{map_summary}\n"
        )
        
        if critic_feedback is not None:
            planner_input += f"\nCritic Feedback on Previous Plan:\n{critic_feedback}\nEnsure that the new plan addresses the critic's concerns.\n"
        plan = self.__call__(planner_input, visual_input, graph_view, top_down_view)
        return plan
        
    def should_terminate(self, task):
        return self.task_terminate(task)["completed"]

    @staticmethod
    def _repo_env_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the action agent with optional TritonAI models.")
    parser.add_argument("--tritonai", action="store_true", help="Use TritonAI-hosted models for planner and critic.")
    parser.add_argument("--task", type=str, default="Find the table and move towards it.", help="The task description for the agent.")
    args = parser.parse_args()
    
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][0]
    controller = Controller(scene=house)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp")
    incontext_dir = os.path.join(project_dir, "..", "incontext")
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

    if args.tritonai:
        print("Using TritonAI-hosted models for the planning agent.")
        planning_agent = PlanningAgent.from_tritonai(
            name="TritonAI Planner",
            role="Planning",
            controller=controller,
            mapping_agent=None,
            perception_agent=perception_module,
            save_dir=temp_dir,
            incontext_dir=incontext_dir,
        )
        
    else:
        planning_agent = PlanningAgent(
        name="Planner",
        role="Planning",
        controller=controller,
        mapping_agent=None,
        perception_agent=perception_module,
        save_dir=temp_dir,
        incontext_dir=incontext_dir,
        use_incontext_example=False,
        )
    
    plan = planning_agent.generate_plan(args.task)
    print("Generated Plan:", plan)

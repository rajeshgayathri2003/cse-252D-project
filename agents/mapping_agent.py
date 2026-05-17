import ast
import json
import math

import networkx as nx

from agents.base import AgentBase, BaseAgentCore


class SpatialGraph:
    def __init__(self, grid_size=0.25, rotation_bins=4, bidirectional_edges=True):
        self.graph = nx.DiGraph()
        self.grid_size = grid_size
        self.rotation_bins = rotation_bins
        self.bidirectional_edges = bidirectional_edges
        self.step_count = 0
        self.reverse_actions = {
            "MoveAhead": "MoveBack",
            "MoveBack": "MoveAhead",
            "MoveRight": "MoveLeft",
            "MoveLeft": "MoveRight",
            "RotateRight": "RotateLeft",
            "RotateLeft": "RotateRight",
        }

    def add_or_update_node(self, position, rotation, objects_seen=None):
        self.step_count += 1
        node_id = self.make_node_id(position, rotation)
        objects_seen = self._normalize_objects(objects_seen)

        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                position=dict(position),
                snapped_position=self._snapped_position(position),
                rotation=dict(rotation),
                rotation_bin=node_id[2],
                objects_seen=[],
                visit_count=0,
                timestamp=self.step_count,
            )

        node_data = self.graph.nodes[node_id]
        node_data["position"] = dict(position)
        node_data["rotation"] = dict(rotation)
        node_data["visit_count"] += 1
        node_data["timestamp"] = self.step_count

        for obj in objects_seen:
            if obj not in node_data["objects_seen"]:
                node_data["objects_seen"].append(obj)

        return node_id

    def add_edge(self, from_id, to_id, action):
        if from_id is None or to_id is None or action is None or from_id == to_id:
            return

        self.graph.add_edge(from_id, to_id, action=action, timestamp=self.step_count)

        reverse_action = self.reverse_actions.get(action)
        if self.bidirectional_edges and reverse_action is not None:
            self.graph.add_edge(
                to_id,
                from_id,
                action=reverse_action,
                timestamp=self.step_count,
                inferred_reverse=True,
            )

    def get_shortest_path(self, from_id, to_id):
        from_id = self._coerce_node_id(from_id)
        to_id = self._coerce_node_id(to_id)
        try:
            return nx.shortest_path(self.graph, source=from_id, target=to_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def path_to_actions(self, path):
        actions = []
        for from_id, to_id in zip(path, path[1:]):
            edge_data = self.graph.get_edge_data(from_id, to_id, default={})
            actions.append(edge_data.get("action"))
        return actions

    def find_nodes_with_object(self, label):
        matches = []
        label = label.lower().strip()
        for node_id, node_data in self.graph.nodes(data=True):
            for obj in node_data.get("objects_seen", []):
                object_label = self._object_label(obj).lower()
                if label and label in object_label:
                    matches.append(node_id)
                    break
        return matches

    def nearest_node(self, position):
        if not self.graph.nodes:
            return None

        target_x = float(position.get("x", 0.0))
        target_z = float(position.get("z", 0.0))
        best_node = None
        best_distance = None

        for node_id, node_data in self.graph.nodes(data=True):
            node_position = node_data.get("position", {})
            dx = float(node_position.get("x", 0.0)) - target_x
            dz = float(node_position.get("z", 0.0)) - target_z
            distance = math.sqrt(dx * dx + dz * dz)
            if best_distance is None or distance < best_distance:
                best_node = node_id
                best_distance = distance

        return best_node

    def get_summary(self, current_node_id=None):
        known_objects = {}
        for node_id, node_data in self.graph.nodes(data=True):
            for obj in node_data.get("objects_seen", []):
                label = self._object_label(obj)
                if not label:
                    continue
                known_objects.setdefault(label, []).append(str(node_id))

        lines = [
            "Spatial map summary:",
            f"- nodes: {self.graph.number_of_nodes()}",
            f"- edges: {self.graph.number_of_edges()}",
            f"- current_node: {current_node_id}",
            "- known_objects:",
        ]

        if known_objects:
            for label, node_ids in sorted(known_objects.items()):
                lines.append(f"  - {label}: {', '.join(node_ids)}")
        else:
            lines.append("  - none")

        return "\n".join(lines)

    def make_node_id(self, position, rotation):
        snapped = self._snapped_position(position)
        return (snapped["x"], snapped["z"], self._rotation_bin(rotation))

    def _snapped_position(self, position):
        return {
            "x": self._snap(float(position.get("x", 0.0))),
            "y": round(float(position.get("y", 0.0)), 2),
            "z": self._snap(float(position.get("z", 0.0))),
        }

    def _snap(self, value):
        return round(round(value / self.grid_size) * self.grid_size, 2)

    def _rotation_bin(self, rotation):
        yaw = float(rotation.get("y", 0.0)) % 360.0
        bin_size = 360.0 / self.rotation_bins
        return int(round(yaw / bin_size)) % self.rotation_bins

    def _normalize_objects(self, objects_seen):
        if objects_seen is None:
            return []
        if isinstance(objects_seen, list):
            return [dict(obj) if isinstance(obj, dict) else {"label": str(obj)} for obj in objects_seen]
        if isinstance(objects_seen, dict):
            for key in ("objects", "detections", "objects_seen"):
                if isinstance(objects_seen.get(key), list):
                    return self._normalize_objects(objects_seen[key])
            return [dict(objects_seen)]
        return [{"label": str(objects_seen)}]

    def _object_label(self, obj):
        if isinstance(obj, dict):
            return str(obj.get("label") or obj.get("name") or obj.get("objectType") or "")
        return str(obj)

    def _coerce_node_id(self, node_id):
        if isinstance(node_id, tuple):
            return node_id
        if isinstance(node_id, str):
            return tuple(ast.literal_eval(node_id))
        return node_id


class MappingAgentCore(BaseAgentCore):
    def __init__(self, name="MappingAgentCore", role="spatial memory service", grid_size=0.25, rotation_bins=4):
        super().__init__(name, role, llm=None)
        self.spatial_graph = SpatialGraph(grid_size=grid_size, rotation_bins=rotation_bins)
        self.current_node_id = None
        self.previous_node_id = None

    def update_map(self, event, perception_output=None, action=None):
        metadata = self._event_metadata(event)
        agent_metadata = metadata.get("agent", {})
        position = agent_metadata.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        rotation = agent_metadata.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})

        if action is None:
            action = metadata.get("lastAction")

        action_success = metadata.get("lastActionSuccess", True)
        node_id = self.spatial_graph.add_or_update_node(position, rotation, perception_output)

        if action_success:
            self.spatial_graph.add_edge(self.current_node_id, node_id, action)

        self.previous_node_id = self.current_node_id
        self.current_node_id = node_id
        return node_id

    def find_object(self, label):
        node_ids = self.spatial_graph.find_nodes_with_object(label)
        results = []
        for node_id in node_ids:
            node_data = self.spatial_graph.graph.nodes[node_id]
            results.append(
                {
                    "node_id": node_id,
                    "position": node_data.get("position"),
                    "objects_seen": node_data.get("objects_seen", []),
                }
            )
        return results

    def get_path(self, target_node_id=None, target_position=None):
        if self.current_node_id is None:
            return {"path": [], "actions": []}

        if target_node_id is None and target_position is not None:
            target_node_id = self.spatial_graph.nearest_node(target_position)

        if target_node_id is None:
            return {"path": [], "actions": []}

        path = self.spatial_graph.get_shortest_path(self.current_node_id, target_node_id)
        return {"path": path, "actions": self.spatial_graph.path_to_actions(path)}

    def get_map_summary(self):
        return self.spatial_graph.get_summary(self.current_node_id)

    def should_terminate(self, user_input):
        return False

    def summarize(self, user_input=None, response=None):
        labels = set()
        for _, node_data in self.spatial_graph.graph.nodes(data=True):
            for obj in node_data.get("objects_seen", []):
                label = self.spatial_graph._object_label(obj)
                if label:
                    labels.add(label)

        snapshot = {
            "nodes": self.spatial_graph.graph.number_of_nodes(),
            "edges": self.spatial_graph.graph.number_of_edges(),
            "current_node": self.current_node_id,
            "known_objects": sorted(labels),
        }
        return json.dumps(snapshot, default=str)

    def _event_metadata(self, event):
        if isinstance(event, dict):
            return event.get("metadata", event)
        return getattr(event, "metadata", {})


class MappingAgent(AgentBase):
    def __init__(self, name="MappingAgent", role="spatial memory and navigation helper", grid_size=0.25, rotation_bins=4):
        self.name = name
        self.role = role
        self.core = MappingAgentCore(name=f"{name}Core", role=role, grid_size=grid_size, rotation_bins=rotation_bins)

    def update(self, event, perception_output=None, action=None):
        return self.core.update_map(event, perception_output=perception_output, action=action)

    def find_object(self, label):
        return self.core.find_object(label)

    def get_path(self, target_node_id=None, target_position=None):
        return self.core.get_path(target_node_id=target_node_id, target_position=target_position)

    def get_context_string(self):
        return self.core.get_map_summary()

    def query(self, user_input):
        text = str(user_input).strip()
        lowered = text.lower()

        if any(word in lowered for word in ("path", "route", "go to", "navigate")):
            target = self._extract_target(text)
            node_id = self._parse_node_id(target)
            if node_id is not None:
                return self.get_path(target_node_id=node_id)

            object_matches = self.find_object(target)
            if object_matches:
                return self.get_path(target_node_id=object_matches[0]["node_id"])
            return {"path": [], "actions": [], "message": f"No known target for '{target}'."}

        if any(word in lowered for word in ("find", "where", "locate", "object")):
            label = self._extract_target(text)
            return self.find_object(label)

        if any(word in lowered for word in ("summary", "map", "context")):
            return self.get_context_string()

        return self.get_context_string()

    def build_context(self, user_input):
        return f"""
Mapping Agent context:
{self.get_context_string()}

Request:
{user_input}
"""

    def generate_response(self, user_input):
        return self.query(user_input), None

    def __call__(self, user_input, caller="PlanningAgent"):
        return self.query(user_input)

    def _extract_target(self, text):
        lowered = text.lower()
        for phrase in ("path to", "route to", "go to", "navigate to", "find object", "find", "where is", "where are", "locate"):
            if phrase in lowered:
                index = lowered.find(phrase) + len(phrase)
                return text[index:].strip(" :?.'")
        return text.strip(" :?.'")

    def _parse_node_id(self, text):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        if isinstance(value, tuple):
            return value
        return None

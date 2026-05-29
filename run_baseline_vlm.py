"""Baseline single-VLM agent loop for ProcTHOR navigation.

A single VLM receives the raw scene image every step and directly outputs the
next AI2-THOR action (plan + translate in one shot), replacing the full
perception → mapping → planning → critic → action pipeline.

Run from the repo root:
    uv run python run_baseline_vlm.py --task "find a flower vase"
    uv run python run_baseline_vlm.py --task "find a flower vase" --tritonai
"""

import argparse
import base64
import math
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO

import prior
from ai2thor.controller import Controller
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_TRITONAI_MODEL = "api-gemma-4-26b"

PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"

NAVIGATION_ACTIONS = [
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight", "LookUp", "LookDown",
    "Crouch", "Stand",
]

OBJECT_INTERACTION_ACTIONS = [
    "PickupObject", "PutObject", "DropHandObject", "ThrowObject",
    "MoveHeldObjectAhead", "MoveHeldObjectBack", "MoveHeldObjectLeft",
    "MoveHeldObjectRight", "MoveHeldObjectUp", "MoveHeldObjectDown",
    "PushObject", "PullObject", "PlaceObjectAtPoint",
]

OBJECT_STATE_ACTIONS = [
    "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
    "DirtyObject", "CleanObject", "FillObjectWithLiquid", "SliceObject",
    "EmptyLiquidFromObject", "UseUpObject",
]

ALL_ACTIONS = NAVIGATION_ACTIONS + OBJECT_INTERACTION_ACTIONS + OBJECT_STATE_ACTIONS

SUCCESS_DISTANCE = 1.5  # metres — slightly generous since AI2-THOR measures to object center

# AI2-THOR objectType names differ from natural language; map the tricky ones.
OBJECT_SYNONYMS: dict[str, str] = {
    # navigation synonyms
    "wash basin": "Sink",
    "washbasin": "Sink",
    "washing machine": "Clothes_Dryer",  # scene 280 has Clothes_Dryer, not WashingMachine
    "toilet seat": "Toilet",
    "notebook": "Book",
    "book on the bed": "Book",
    "couch": "Sofa",
    "tv": "Television",
    "television": "Television",
    "refrigerator": "Fridge",
    "fridge": "Fridge",
    "trash bag": "GarbageBag",
    "garbage bag": "GarbageBag",
    "painting on the wall": "Painting",
    "painting": "Painting",
    "study desk": "Desk",
    "desk": "Desk",
}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    task_id: str
    scene_index: int
    task: str
    success: bool
    shortest_path_length: float | None  # ℓᵢ — None when pathfinder failed
    actual_path_length: float           # pᵢ
    steps_taken: int


def parse_target_type(task: str) -> str:
    """Extract target object type from 'Navigate to the X' task strings."""
    match = re.search(r"navigate to (?:the|a|an)?\s*(.+)", task, re.IGNORECASE)
    raw = match.group(1).strip().lower() if match else task.lower()
    for phrase, obj_type in OBJECT_SYNONYMS.items():
        if phrase in raw:
            return obj_type
    # PascalCase conversion (e.g. "flower vase" → "FlowerVase")
    return "".join(w.capitalize() for w in raw.split())


def find_target_objects(metadata_objects: list[dict], target_type: str) -> list[dict]:
    """Return all scene objects whose objectType contains the target string."""
    tl = target_type.lower()
    return [o for o in metadata_objects if tl in o["objectType"].lower()]


def check_success(event, target_type: str, threshold: float = SUCCESS_DISTANCE) -> bool:
    """True if any matching object is within threshold metres (distance-only, no visibility raycast)."""
    for obj in find_target_objects(event.metadata["objects"], target_type):
        if obj["distance"] <= threshold:
            return True
    return False


def get_shortest_path_length(controller: Controller, target_obj_id: str) -> float | None:
    """Query AI2-THOR's navmesh pathfinder for the shortest obstacle-aware path length."""
    try:
        event = controller.step(action="GetShortestPath", objectId=target_obj_id)
    except ValueError:
        return None
    if not event.metadata.get("lastActionSuccess"):
        return None
    corners = (event.metadata.get("actionReturn") or {}).get("corners", [])
    if len(corners) < 2:
        return None
    total = 0.0
    for i in range(1, len(corners)):
        dx = corners[i]["x"] - corners[i - 1]["x"]
        dz = corners[i]["z"] - corners[i - 1]["z"]
        total += math.sqrt(dx * dx + dz * dz)
    return total


def agent_position(event) -> tuple[float, float]:
    pos = event.metadata["agent"]["position"]
    return pos["x"], pos["z"]


def compute_sr(results: list[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.success for r in results) / len(results)


def compute_spl(results: list[EpisodeResult]) -> float:
    """SPL = (1/N) Σ Sᵢ · ℓᵢ / max(pᵢ, ℓᵢ).  Episodes with no shortest path contribute 0."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.success and r.shortest_path_length is not None:
            denom = max(r.actual_path_length, r.shortest_path_length)
            total += r.shortest_path_length / denom if denom > 0 else 1.0
    return total / len(results)


def print_metrics(results: list[EpisodeResult]):
    sr = compute_sr(results)
    spl = compute_spl(results)
    print(f"\n{'=' * 60}")
    print(f"  Episodes : {len(results)}")
    print(f"  SR       : {sr:.3f}  ({sum(r.success for r in results)}/{len(results)} success)")
    print(f"  SPL      : {spl:.3f}")
    print(f"{'=' * 60}")
    for r in results:
        status = "+" if r.success else "-"
        if r.shortest_path_length is not None:
            denom = max(r.actual_path_length, r.shortest_path_length)
            spl_ep = f"{r.shortest_path_length / denom:.3f}" if denom > 0 else "1.000"
            path_info = f"l={r.shortest_path_length:.2f}m  p={r.actual_path_length:.2f}m  SPL={spl_ep}"
        else:
            path_info = f"l=n/a  p={r.actual_path_length:.2f}m  SPL=n/a"
        print(f"  [{status}] Scene {r.scene_index} {r.task_id}: {r.task[:38]:<38}  {path_info}")


SYSTEM_PROMPT = """You are an embodied AI agent navigating a 3D indoor environment.

You receive:
- The task to complete
- A first-person RGB image of the current scene

Your job: decide the single best next action to make progress toward the task based purely on what you see in the image.

Rules:
- Return ONLY the controller.step(...) call, nothing else — no explanation, no markdown.
- For navigation: controller.step(action='MoveAhead')
- If the target object is not visible, explore by rotating or moving forward.
- SUCCESS is defined as being within 1.5 metres of the target object.
"""

ACTION_PROMPT_TEMPLATE = """Task: {task}

Available actions:
{actions}

Current step: {step} / {max_steps}
Action history (last 5): {history}

Based only on the image above, output the single best next controller.step(...) call."""


def encode_pil_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_client(tritonai: bool, api_key: str | None = None) -> tuple[OpenAI, str]:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    if tritonai:
        key = api_key or os.environ.get("TRITONAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("TRITONAI_API_KEY not found in .env")
        return OpenAI(base_url=TRITONAI_BASE_URL, api_key=key), DEFAULT_TRITONAI_MODEL
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            # fall back to key.txt used by the existing planner
            key_path = os.path.join(os.path.dirname(__file__), "key.txt")
            if os.path.exists(key_path):
                with open(key_path) as f:
                    key = f.read().strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not found in .env or key.txt")
        return OpenAI(api_key=key), DEFAULT_OPENAI_MODEL


def build_controller(scene_index: int, width: int = 1280, height: int = 720) -> Controller:
    print(f"[Sim] Loading ProcTHOR house {scene_index}...")
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][scene_index]
    print(f"[Sim] Starting AI2-THOR controller ({width}x{height})...")
    try:
        return Controller(scene=house, platform="CloudRendering", width=width, height=height)
    except Exception:
        return Controller(scene=house, width=width, height=height)


ALL_ACTION_NAMES = set(NAVIGATION_ACTIONS + OBJECT_INTERACTION_ACTIONS + OBJECT_STATE_ACTIONS)

def extract_step_call(text: str) -> str | None:
    """Extract a controller.step(...) call from VLM output, handling many response styles."""
    # Strip markdown code fences
    clean = re.sub(r"```(?:python)?", "", text).replace("```", "").strip()

    # 1. Full controller.step(...) anywhere in the text
    m = re.search(r"controller\.step\([^)]*\)", clean)
    if m:
        return m.group(0)

    # 2. step(...) without "controller." prefix
    m = re.search(r"\bstep\(([^)]*)\)", clean)
    if m:
        return f"controller.step({m.group(1)})"

    # 3. action='X' or action="X" anywhere
    m = re.search(r"""action\s*=\s*['"](\w+)['"]""", clean)
    if m:
        action = m.group(1)
        obj_m = re.search(r"""objectId\s*=\s*['"]([^'"]+)['"]""", clean)
        if obj_m:
            return f"controller.step(action='{action}', objectId='{obj_m.group(1)}')"
        return f"controller.step(action='{action}')"

    # 4. Bare action name on its own line (e.g. model just says "MoveAhead")
    for line in clean.splitlines():
        word = line.strip().strip("'\".,")
        if word in ALL_ACTION_NAMES:
            return f"controller.step(action='{word}')"

    # 5. Action name mentioned anywhere in the text
    for action in ALL_ACTION_NAMES:
        if re.search(rf"\b{action}\b", clean):
            obj_m = re.search(r"""objectId\s*=\s*['"]([^'"]+)['"]""", clean)
            if obj_m:
                return f"controller.step(action='{action}', objectId='{obj_m.group(1)}')"
            return f"controller.step(action='{action}')"

    return None


class BaselineVLMAgent:
    """Single VLM that sees the image and directly outputs a ProcTHOR action."""

    def __init__(self, controller: Controller, client: OpenAI, model: str):
        self.controller = controller
        self.client = client
        self.model = model
        self.action_history: list[str] = []

    def _get_objects(self) -> list[dict]:
        return [
            {"objectType": obj["objectType"], "objectId": obj["objectId"]}
            for obj in self.controller.last_event.metadata["objects"]
        ]

    def _call_vlm(self, image: Image.Image, prompt_text: str) -> tuple[str, str]:
        """Returns (content, reasoning) — reasoning may be empty if model doesn't expose it."""
        image_b64 = encode_pil_image(image)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.2,
                )
                msg = response.choices[0].message
                content = msg.content or ""
                # Try all known fields where reasoning models expose their trace
                reasoning = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or (msg.model_extra or {}).get("reasoning_content")
                    or (msg.model_extra or {}).get("reasoning")
                    or ""
                )
                if not content:
                    print(f"  [VLM] Empty response — finish_reason={response.choices[0].finish_reason}, usage={response.usage}")
                    print(f"  [VLM] Message fields: {list(msg.model_fields_set)} extra={msg.model_extra}")
                return content.strip(), str(reasoning).strip() if reasoning else ""
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = min(2 ** (attempt + 2), 120)  # 4s, 8s, 16s, 32s, 64s, cap 120s
                    print(f"  [RateLimit] Attempt {attempt+1}/8 — waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("VLM call failed after 8 retries due to rate limiting.")

    def step(self, task: str, step_num: int, max_steps: int, last_action_success: bool = True) -> tuple[str, str]:
        """Observe → decide → act. Returns (command, raw_vlm_response)."""
        image = Image.fromarray(self.controller.last_event.frame)

        actions_str = (
            f"  Navigation: {', '.join(NAVIGATION_ACTIONS)}\n"
            f"  Object interaction: {', '.join(OBJECT_INTERACTION_ACTIONS)}\n"
            f"  Object state change: {', '.join(OBJECT_STATE_ACTIONS)}"
        )

        recent_history = self.action_history[-5:] if self.action_history else ["(none)"]
        failure_note = "" if last_action_success else "\nWARNING: Your last action FAILED (blocked by wall or obstacle). Choose a different action.\n"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            task=task,
            actions=actions_str,
            step=step_num,
            max_steps=max_steps,
            history=", ".join(recent_history),
        ) + failure_note

        raw_response, reasoning = self._call_vlm(image, prompt)
        print(f"  [VLM raw] {raw_response}")
        if reasoning:
            print(f"  [VLM reasoning] {reasoning}")

        command = extract_step_call(raw_response)
        if command is None:
            print(f"  [Baseline] WARNING: could not parse action from response: {repr(raw_response[:200])}, defaulting to MoveAhead.")
            command = "controller.step(action='MoveAhead')"

        self.action_history.append(command)
        return command, raw_response, reasoning

    def execute(self, command: str) -> bool:
        try:
            exec(command, {"controller": self.controller})
            success = self.controller.last_event.metadata.get("lastActionSuccess", False)
            return success
        except Exception as e:
            print(f"  [Baseline] Action execution error: {e}")
            return False


def parse_eval_tasks(path: str) -> list[dict]:
    """Parse eval_tasks.txt into a list of {scene_index, task_id, task} dicts."""
    tasks = []
    current_scene = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Scene Index:"):
                current_scene = int(line.split(":")[1].strip())
            elif line[:2] in ("T1", "T2", "T3") and current_scene is not None:
                task_id = line.split()[0]
                task_text = line[len(task_id):].strip()
                tasks.append({"scene_index": current_scene, "task_id": task_id, "task": task_text})
    return tasks


def run_single_task(task: str, scene_index: int, task_id: str, steps: int,
                    output_dir: str, client: OpenAI, model: str,
                    success_distance: float = SUCCESS_DISTANCE,
                    step_delay: float = 1.0,
                    width: int = 640, height: int = 480) -> tuple[EpisodeResult, list[str]]:
    """Run the baseline loop for one task. Returns (EpisodeResult, log_lines)."""
    os.makedirs(output_dir, exist_ok=True)
    controller = build_controller(scene_index, width=width, height=height)
    agent = BaselineVLMAgent(controller, client, model)
    log_lines: list[str] = [f"Task: {task}\nScene: {scene_index}\nModel: {model}\nSteps: {steps}\n"]

    controller.step("Pass")
    event = controller.last_event

    # --- Episode setup: target type + shortest path length ---
    target_type = parse_target_type(task)
    targets = find_target_objects(event.metadata["objects"], target_type)
    print(f"  [Metrics] Target type: '{target_type}' — {len(targets)} object(s) found in scene")

    shortest_path_length: float | None = None
    if targets:
        # Use the closest matching object for the reference shortest path
        best = min(targets, key=lambda o: o["distance"])
        shortest_path_length = get_shortest_path_length(controller, best["objectId"])
        obj_pos = best["position"]
        print(f"  [Target] '{best['objectType']}' at x={obj_pos['x']:.3f}  y={obj_pos['y']:.3f}  z={obj_pos['z']:.3f}")
        print(f"  [Metrics] Shortest path to '{best['objectType']}': "
              f"{shortest_path_length:.2f}m" if shortest_path_length else "  [Metrics] Pathfinder failed")

    prev_pos = agent_position(event)
    actual_path_length = 0.0
    episode_success = False
    last_action_success = True

    try:
        for step in range(1, steps + 1):
            print(f"\n{'=' * 60}\nStep {step}/{steps}\n{'=' * 60}")

            frame = Image.fromarray(controller.last_event.frame)
            frame.save(os.path.join(output_dir, f"step_{step:03d}_before.jpg"))

            command, vlm_trace, vlm_reasoning = agent.step(task, step, steps, last_action_success)
            print(f"  [Action] {command}")

            agent.execute(command)
            event = controller.last_event

            # Accumulate path length (x,z Euclidean)
            cur_pos = agent_position(event)
            actual_path_length += math.sqrt(
                (cur_pos[0] - prev_pos[0]) ** 2 + (cur_pos[1] - prev_pos[1]) ** 2
            )
            prev_pos = cur_pos

            last_action_success = event.metadata.get("lastActionSuccess", False)
            episode_success = check_success(event, target_type, success_distance)

            pos = event.metadata["agent"]["position"]
            # Compute distance to closest matching target for logging
            target_objs = find_target_objects(event.metadata["objects"], target_type)
            dist_to_target = min((o["distance"] for o in target_objs), default=None)
            dist_str = f"{dist_to_target:.3f}m" if dist_to_target is not None else "n/a"

            print(f"  [Position] x={pos['x']:.3f}  y={pos['y']:.3f}  z={pos['z']:.3f}")
            print(f"  [Distance to target] {dist_str}  (threshold={success_distance}m)")
            print(f"  [Result] action_success={last_action_success}  episode_success={episode_success}")
            reasoning_block = f"  Reasoning:\n{vlm_reasoning}\n" if vlm_reasoning else ""
            log_lines.append(
                f"Step {step}:\n"
                f"{reasoning_block}"
                f"  VLM: {vlm_trace}\n"
                f"  Action: {command}\n"
                f"  Position: x={pos['x']:.3f} y={pos['y']:.3f} z={pos['z']:.3f}\n"
                f"  Distance to target: {dist_str}\n"
                f"  action_ok={last_action_success}  goal_reached={episode_success}"
            )

            if episode_success:
                print("  [Metrics] Goal reached — stopping early.")
                break

            if step_delay > 0:
                time.sleep(step_delay)
    finally:
        controller.stop()

    log_lines.append(
        f"\nFinal: success={episode_success}  "
        f"shortest_path={shortest_path_length}  actual_path={actual_path_length:.3f}"
    )

    result = EpisodeResult(
        task_id=task_id,
        scene_index=scene_index,
        task=task,
        success=episode_success,
        shortest_path_length=shortest_path_length,
        actual_path_length=actual_path_length,
        steps_taken=step,
    )
    return result, log_lines


def run_baseline(args):
    client, model = build_client(args.tritonai, args.api_key)
    if args.model:
        model = args.model

    all_results: list[EpisodeResult] = []

    if args.eval_tasks:
        tasks = parse_eval_tasks(args.eval_tasks)
        # Apply --start-from filter: "SCENE_IDX:TASK_ID" e.g. "120:T3"
        if args.start_from:
            parts = args.start_from.split(":")
            start_scene = int(parts[0])
            start_task = parts[1].strip().upper() if len(parts) > 1 else "T1"
            reached = False
            filtered = []
            for t in tasks:
                if not reached:
                    if t["scene_index"] == start_scene and t["task_id"] == start_task:
                        reached = True
                if reached:
                    filtered.append(t)
            tasks = filtered
            print(f"[Eval] Starting from scene {start_scene} {start_task} — {len(tasks)} tasks remaining.")
        print(f"[Eval] Running {len(tasks)} tasks from {args.eval_tasks}")
        for entry in tasks:
            scene_idx = entry["scene_index"]
            task_id = entry["task_id"]
            task = entry["task"]
            task_dir = os.path.join(args.output_dir, f"scene_{scene_idx}", task_id)
            print(f"\n{'#' * 60}\n[Eval] Scene {scene_idx} | {task_id}: {task}\n{'#' * 60}")
            result, log_lines = run_single_task(
                task, scene_idx, task_id, args.steps, task_dir, client, model,
                step_delay=args.step_delay, width=args.width, height=args.height,
            )
            all_results.append(result)
            with open(os.path.join(task_dir, "baseline_run.txt"), "w") as f:
                f.write("\n".join(log_lines))
            print(f"[Eval] Done — success={result.success}  SPL_ep contribution logged.")

        print(f"\n[Eval] All tasks complete. Outputs in {args.output_dir}/")
        print_metrics(all_results)

        # Save aggregate metrics
        sr = compute_sr(all_results)
        spl = compute_spl(all_results)
        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"SR={sr:.4f}\nSPL={spl:.4f}\n\nPer-episode:\n")
            for r in all_results:
                f.write(
                    f"Scene {r.scene_index} {r.task_id}: success={r.success}  "
                    f"l={r.shortest_path_length}  p={r.actual_path_length:.3f}\n"
                )
        print(f"[Eval] Metrics saved to {metrics_path}")
    else:
        result, log_lines = run_single_task(
            args.task, args.scene_index, args.task_id or "T0",
            args.steps, args.output_dir, client, model,
            step_delay=args.step_delay, width=args.width, height=args.height,
        )
        log_path = os.path.join(args.output_dir, "baseline_run.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"\n[Baseline] success={result.success}  "
              f"actual_path={result.actual_path_length:.2f}m  "
              f"shortest_path={result.shortest_path_length}m")
        print(f"[Baseline] Log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline single-VLM agent loop for ProcTHOR.")
    parser.add_argument("--task", default="Navigate to the Television", help="Task description (single-task mode).")
    parser.add_argument("--task-id", default="T0", help="Task ID label for single-task mode.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps per task.")
    parser.add_argument("--scene-index", type=int, default=5, help="ProcTHOR train scene index (single-task mode).")
    parser.add_argument("--output-dir", default="baseline_outputs", help="Directory to save frames and logs.")
    parser.add_argument("--eval-tasks", default=None, metavar="PATH",
                        help="Path to eval_tasks.txt — runs all scenes/tasks in the file.")
    parser.add_argument("--start-from", default=None, metavar="SCENE:TASKID",
                        help="Skip tasks before this point, e.g. '120:T3' starts at scene 120 T3.")
    parser.add_argument("--success-distance", type=float, default=SUCCESS_DISTANCE,
                        help=f"Distance threshold (m) for navigation success (default: {SUCCESS_DISTANCE}).")
    parser.add_argument("--step-delay", type=float, default=1.0,
                        help="Seconds to sleep between steps to avoid rate limits (default: 1.0).")
    parser.add_argument("--width", type=int, default=640, help="Simulator frame width in pixels (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Simulator frame height in pixels (default: 480).")
    parser.add_argument("--tritonai", action="store_true", help="Use TritonAI (UCSD) instead of OpenAI.")
    parser.add_argument("--model", default=None, help="Override the default model name.")
    parser.add_argument("--api-key", default=None, help="API key (falls back to .env / key.txt).")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())

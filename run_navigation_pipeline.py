"""Integration pipeline for Florence-2 perception, mapping, planning, and critic.

This script wires together the agents that exist today:

1. Florence-2 perception reads the current ProcTHOR RGB frame.
2. MappingAgent stores the current pose and Florence object labels.
3. PlanningAgent proposes the next sub-goal/action.
4. CriticAgent reviews the planner output before any execution.
5. ActionAgent translates the approved subgoal into a concrete AI2-THOR action and executes it.

Run from the repo root, after installing sim/perception dependencies:
    uv run python run_navigation_pipeline.py --task "find a flower vase"

Evaluate all tasks from eval_tasks.txt:
    uv run python run_navigation_pipeline.py --eval-tasks eval_tasks.txt
"""

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime

import prior
from ai2thor.controller import Controller
from PIL import Image

from agents.act.agent import ActionAgent
from agents.critic_agent import CriticAgent
from agents.mapping_agent import MappingAgent
try:
    from agents.perception.agent import FlorencePerceptionAgent
except ImportError as e:
    print(f"Error importing FlorencePerceptionAgent: {e}")
    FlorencePerceptionAgent = None
from agents.plan.agent import PlanningAgent, encode_image


FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
SAM_WEIGHTS = "sam2_b.pt"

# ---------------------------------------------------------------------------
# Evaluation metrics helpers (mirrors run_baseline_vlm.py)
# ---------------------------------------------------------------------------

SUCCESS_DISTANCE = 1.5  # metres

OBJECT_SYNONYMS: dict[str, str] = {
    "wash basin": "Sink",
    "washbasin": "Sink",
    "washing machine": "Clothes_Dryer",
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


@dataclass
class EpisodeResult:
    task_id: str
    scene_index: int
    task: str
    success: bool
    shortest_path_length: float | None
    actual_path_length: float
    steps_taken: int


def parse_target_type(task: str) -> str:
    match = re.search(r"navigate to (?:the|a|an)?\s*(.+)", task, re.IGNORECASE)
    raw = match.group(1).strip().lower() if match else task.lower()
    for phrase, obj_type in OBJECT_SYNONYMS.items():
        if phrase in raw:
            return obj_type
    return "".join(w.capitalize() for w in raw.split())


def find_target_objects(metadata_objects: list[dict], target_type: str) -> list[dict]:
    tl = target_type.lower()
    return [o for o in metadata_objects if tl in o["objectType"].lower()]


def check_success(event, target_type: str, threshold: float = SUCCESS_DISTANCE) -> bool:
    for obj in find_target_objects(event.metadata["objects"], target_type):
        if obj["distance"] <= threshold:
            return True
    return False


def get_shortest_path_length(controller: Controller, target_obj_id: str) -> float | None:
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


WEDGE_THRESHOLD = 3


def consecutive_trailing_failures(history: list[str]) -> int:
    """Count [FAILED]-tagged entries at the end of action_history."""
    n = 0
    for entry in reversed(history):
        if "[FAILED:" in entry:
            n += 1
        else:
            break
    return n


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


@dataclass
class PerceptionResult:
    description: str
    objects: list[dict]
    frame_path: str


def _slugify(text, max_len=40):
    """Turn a free-form task string into a filesystem-safe label."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "run"


def make_run_dir(base_dir, task, scene_index, now=None):
    """Return a unique, timestamped + labeled run directory under base_dir.

    Example: pipeline_outputs/20260529_193045_find-a-flower-vase_scene5
    Each run gets its own subdir so prior runs are retained instead of wiped.
    """
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{_slugify(task)}_scene{scene_index}"
    run_dir = os.path.join(base_dir, name)
    suffix = 2
    while os.path.exists(run_dir):  # guard against same-second reruns
        run_dir = os.path.join(base_dir, f"{name}_{suffix}")
        suffix += 1
    return run_dir


def write_run_meta(run_dir, args, task=None, scene_index=None, extra=None):
    """Write run_meta.json describing how this run was configured."""
    meta = {
        "run_dir": os.path.basename(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "task": task if task is not None else args.task,
        "scene_index": scene_index if scene_index is not None else args.scene_index,
        "steps": args.steps,
        "florence_model": args.florence_model,
        "sam_weights": args.sam_weights,
        "device": args.device,
        "tritonai": args.tritonai,
        "headless_perception": args.headless_perception,
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_controller(scene_index):
    print("[Sim] Loading ProcTHOR house...")
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][scene_index]
    print("[Sim] Starting AI2-THOR controller with CloudRendering...")
    try:
        return Controller(scene=house, platform="CloudRendering")
    except Exception as e:
        print(f"[Sim] CloudRendering failed ({e}); falling back to default platform.")
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


def run_single_task(task: str, scene_index: int, task_id: str, args) -> tuple[EpisodeResult, list[str]]:
    """Run the navigation pipeline for one task. Returns (EpisodeResult, log_lines)."""
    controller = build_controller(scene_index)
    mapping_agent = MappingAgent()
    run_dir = make_run_dir(args.output_dir, task, scene_index)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    incontext_dir = os.path.join(project_dir, "incontext_examples") if args.in_context else None
    print(f"[Pipeline] Run outputs -> {run_dir}")

    perception_agent = FlorencePerceptionAgent(
        florence_model=args.florence_model,
        sam_weights=args.sam_weights,
        save_dir=run_dir,
        headless=args.headless_perception,
        device=args.device,
    )
    # Written after perception agent, which clears its own save_dir on init.
    write_run_meta(run_dir, args, task=task, scene_index=scene_index)

    temp_dir = os.path.join(project_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temp folder at: {temp_dir}")

    if args.tritonai:
        planner = PlanningAgent.from_tritonai(
            name="Planner",
            role="navigation planner",
            controller=controller,
            mapping_agent=mapping_agent,
            save_dir=temp_dir,
            incontext_dir=incontext_dir,
            use_incontext_example=args.in_context
        )
        critic = CriticAgent.from_tritonai()
        action_agent = ActionAgent.from_tritonai(
            name="ActionAgent",
            role="action executor",
            controller=controller,
            planning_agent=planner,
            save_dir=temp_dir,
            in_context_example=args.in_context,
        )
    else:
        planner = PlanningAgent(
            name="Planner",
            role="Planning",
            controller=controller,
            mapping_agent=None,
            perception_agent=perception_agent,
            save_dir=temp_dir,
            incontext_dir=incontext_dir,
            use_incontext_example=args.in_context
        )
        critic = CriticAgent.from_openai()
        action_agent = ActionAgent(
            name="ActionAgent",
            role="User",
            controller=controller,
            planning_agent=planner,
            in_context_example=args.in_context,
        )

    # --- Episode setup: initial state + metrics ---
    event = controller.step("Pass")

    target_type = parse_target_type(task)
    targets = find_target_objects(event.metadata["objects"], target_type)
    print(f"  [Metrics] Target type: '{target_type}' — {len(targets)} object(s) found in scene")

    shortest_path_length: float | None = None
    if targets:
        best = min(targets, key=lambda o: o["distance"])
        shortest_path_length = get_shortest_path_length(controller, best["objectId"])
        obj_pos = best["position"]
        print(f"  [Target] '{best['objectType']}' at x={obj_pos['x']:.3f}  y={obj_pos['y']:.3f}  z={obj_pos['z']:.3f}")
        if shortest_path_length is not None:
            print(f"  [Metrics] Shortest path to '{best['objectType']}': {shortest_path_length:.2f}m")
        else:
            print(f"  [Metrics] Pathfinder failed for '{best['objectType']}'")

    prev_pos = agent_position(event)
    actual_path_length = 0.0
    episode_success = False
    critic_aborted = False
    action_history = []
    # Seed distance_history with the pre-action distance so the critic sees the
    # full trajectory (pre-action distance, post-action distance, ...).
    _initial_targets = find_target_objects(event.metadata["objects"], target_type)
    _initial_dist = min((o["distance"] for o in _initial_targets), default=None)
    distance_history: list[float | None] = [_initial_dist]
    # Perception-recency tracking: which step did the perception model last
    # detect the target, and how many actions had been executed by then?
    # Used by the critic's (B) carve-out: a target seen recently with no
    # rotation since is still in the forward camera frustum.
    target_last_seen_step: int | None = None
    actions_at_last_sight: int = 0
    completed_steps = 0
    log_lines: list[str] = [f"Task: {task}\nScene: {scene_index}\nSteps: {args.steps}\n"]

    target_objs = find_target_objects(event.metadata["objects"], target_type)
    dist_to_target = min((o["distance"] for o in target_objs), default=None)
    prev_dist_to_target = None
    try:
        for step in range(args.steps):
            print(f"\n{'=' * 72}\nCycle {step + 1}/{args.steps}: observe -> map -> plan -> critique\n{'=' * 72}")

            frame_name = f"step_{step:02d}"
            step_dir = os.path.join(run_dir, frame_name)
            frame = Image.fromarray(event.frame)
            perception = run_perception(perception_agent, frame, frame_name)

            # Update perception-recency tracking before invoking the critic.
            if target_type and target_type.lower() in perception.description.lower():
                target_last_seen_step = step
                actions_at_last_sight = len(action_history)
            if target_last_seen_step is not None:
                cycles_since_target_seen = step - target_last_seen_step
                actions_since_target_seen = list(action_history[actions_at_last_sight:])
            else:
                cycles_since_target_seen = None
                actions_since_target_seen = None

            mapping_agent.update(
                event,
                perception_output=perception.objects,
                action=event.metadata.get("lastAction"),
            )
            map_summary = mapping_agent.get_context_string()

            plan = planner.generate_plan(
                task=task,
                visual_input=encode_image(perception.frame_path),
                perception_description=perception.description,
                map_summary=map_summary,
                prev_dist_to_target=prev_dist_to_target,
                curr_dist_to_target=dist_to_target,
            )

            # Wedge guard: if the agent has accumulated >=3 consecutive
            # failed movement actions, the LLM critic has historically not
            # caught this signal even when it appears in both action_history
            # ([FAILED] tags) and map_summary (blocked_actions_at_current_node).
            # Skip the critic LLM and inject a verdict-shaped rejection that
            # routes through the existing planner-feedback channel. The next
            # planner call receives this rejection's reason as critic_feedback
            # and is expected to emit a rotation-shaped plan, which the real
            # critic then reviews normally.
            n_fails = consecutive_trailing_failures(action_history)
            if n_fails >= WEDGE_THRESHOLD:
                print(
                    f"\n  [Pipeline guard] Wedge detected "
                    f"({n_fails} consecutive failed moves); "
                    f"injecting critic rejection."
                )
                verdict = {
                    "approved": False,
                    "reason": (
                        f"Wedge: {n_fails} consecutive movement actions blocked "
                        f"at this node. Propose a sub-goal that rotates the "
                        f"agent by 90 degrees or more to face a different "
                        f"direction, then re-plan from the new view."
                    ),
                    "revised_subgoal": None,
                }
            else:
                verdict = critic.review(
                    task=task,
                    proposed_subgoal=plan,
                    map_summary=map_summary,
                    action_history=action_history,
                    perception_description=perception.description,
                    distance_history=distance_history,
                    target_type=target_type,
                    cycles_since_target_seen=cycles_since_target_seen,
                    actions_since_target_seen=actions_since_target_seen,
                )

            critic_attempts = [verdict]
            if verdict["approved"]:
                selected_subgoal = plan

            loop_count = 0
            while not verdict["approved"] and loop_count < 5:
                print(f"\nCritic rejected (attempt {loop_count + 1}): {verdict['reason']}\n")
                loop_count += 1
                plan = planner.generate_plan(
                    task=task,
                    visual_input=encode_image(perception.frame_path),
                    perception_description=perception.description,
                    map_summary=map_summary,
                    critic_feedback=verdict["reason"],
                )

                verdict = critic.review(
                    task=task,
                    proposed_subgoal=plan,
                    map_summary=map_summary,
                    action_history=action_history,
                    perception_description=perception.description,
                    distance_history=distance_history,
                    target_type=target_type,
                    cycles_since_target_seen=cycles_since_target_seen,
                    actions_since_target_seen=actions_since_target_seen,
                )
                critic_attempts.append(verdict)

            # If the rejection loop exited because the ceiling was hit (not
            # because the critic approved), terminate the episode cleanly
            # instead of executing the rejected plan. The pipeline's
            # safety-valve fall-through (execute the last rejected plan
            # anyway) is a drift accelerator at long step budgets — the
            # critic correctly identified that it cannot approve any plan,
            # so the episode is unrecoverable from this state.
            if not verdict["approved"]:
                print(
                    f"\n  [Critic] Ceiling hit after {loop_count} rejections "
                    f"at cycle {step + 1}; terminating episode."
                )
                critic_aborted = True
                with open(os.path.join(step_dir, "pipeline_result.txt"), "w") as f:
                    f.write(f"Task:\n{task}\n\n")
                    f.write(f"Perception:\n{perception.description}\n\n")
                    f.write(f"Map:\n{map_summary}\n\n")
                    f.write(f"Planner proposed:\n{plan}\n\n")
                    f.write(f"Critic verdict:\n{verdict}\n\n")
                    f.write(f"Critic attempts ({len(critic_attempts)}):\n")
                    for i, v in enumerate(critic_attempts):
                        f.write(f"  [{i}] {v}\n")
                    f.write(
                        f"\nAborted: critic rejected all {loop_count} plans "
                        f"(ceiling hit).\n"
                    )
                completed_steps += 1
                break

            selected_subgoal = plan

            print("\n[Perception]\n" + perception.description)
            print("\n[Map]\n" + map_summary)
            print("\n[Planner proposed]\n" + plan)
            print("\n[Critic verdict]\n" + str(verdict))
            print("\n[Selected subgoal]\n" + selected_subgoal)

            done = False
            action_command = None
            error_message = None

            # Cap inner action loop. perform_action returns False on collision
            # (lastActionSuccess=False) so an LLM that keeps picking the same
            # blocked MoveAhead would loop forever. Three attempts gives the
            # LLM room to try alternatives within the cycle; past that, break
            # and let the next cycle's planner+critic see the failure tags in
            # action_history and re-strategize.
            MAX_ACTION_ATTEMPTS = 3
            attempt = 0
            while not done and attempt < MAX_ACTION_ATTEMPTS:
                action_command = action_agent.choose_action(
                    selected_subgoal,
                    failure_msg= error_message,
                    perception_description=perception.description,
                )
                print(f"\n[ActionAgent] Executing: {action_command}")

                agent_meta = controller.last_event.metadata["agent"]
                start = (round(agent_meta["position"]["x"], 2),
                         round(agent_meta["position"]["z"], 2))
                print(f"Agent starting position (x, z): {start}")

                done, error_message = action_agent.perform_action(action_command)
                attempt += 1

                # Accumulate path length after each physical action
                cur_pos = agent_position(controller.last_event)
                actual_path_length += math.sqrt(
                    (cur_pos[0] - prev_pos[0]) ** 2 + (cur_pos[1] - prev_pos[1]) ** 2
                )
                prev_pos = cur_pos

                agent_meta = controller.last_event.metadata["agent"]
                final = (round(agent_meta["position"]["x"], 2),
                         round(agent_meta["position"]["z"], 2))
                print(f"Agent final position (x, z): {final}")

                # Tag failures so the next cycle's critic can see them via the
                # action_history field and apply criteria (C) non-repetition
                # and (D) progress-trend correctly. A bare action string would
                # be indistinguishable from a successful execution.
                if done:
                    action_history.append(action_command)
                else:
                    err = error_message or "unknown error"
                    action_history.append(f"{action_command} [FAILED: {err}]")

            if not done:
                print(
                    f"  [ActionAgent] {MAX_ACTION_ATTEMPTS} attempts failed; "
                    f"breaking to next cycle for re-plan."
                )

            event = controller.last_event
            action_agent.get_current_visual_output(frame_id=step)

            #update distance to target
            prev_dist_to_target = dist_to_target
            
            # Check navigation success
            episode_success = check_success(event, target_type, SUCCESS_DISTANCE)
            target_objs = find_target_objects(event.metadata["objects"], target_type)
            dist_to_target = min((o["distance"] for o in target_objs), default=None)
            distance_history.append(dist_to_target)
            dist_str = f"{dist_to_target:.3f}m" if dist_to_target is not None else "n/a"

            pos = event.metadata["agent"]["position"]
            print(f"  [Position] x={pos['x']:.3f}  y={pos['y']:.3f}  z={pos['z']:.3f}")
            print(f"  [Distance to target] {dist_str}  (threshold={SUCCESS_DISTANCE}m)")
            print(f"  [Result] episode_success={episode_success}")

            with open(os.path.join(step_dir, "pipeline_result.txt"), "w") as f:
                f.write(f"Task:\n{task}\n\n")
                f.write(f"Perception:\n{perception.description}\n\n")
                f.write(f"Map:\n{map_summary}\n\n")
                f.write(f"Planner proposed:\n{plan}\n\n")
                f.write(f"Critic verdict:\n{verdict}\n\n")
                if len(critic_attempts) > 1:
                    f.write(f"Critic attempts ({len(critic_attempts)}):\n")
                    for i, v in enumerate(critic_attempts):
                        f.write(f"  [{i}] {v}\n")
                    f.write("\n")
                f.write(f"Selected subgoal:\n{selected_subgoal}\n\n")
                f.write(f"Action executed:\n{action_command}\n")
                f.write(f"Position: x={pos['x']:.3f} y={pos['y']:.3f} z={pos['z']:.3f}\n")
                f.write(f"Distance to target: {dist_str}\n")
                f.write(f"episode_success={episode_success}\n")

            log_lines.append(
                f"Cycle {step + 1}:\n"
                f"  Perception: {perception.description[:120]}\n"
                f"  Plan: {plan[:120]}\n"
                f"  Critic: approved={verdict['approved']}\n"
                f"  Action: {action_command}\n"
                f"  Position: x={pos['x']:.3f} y={pos['y']:.3f} z={pos['z']:.3f}\n"
                f"  Distance to target: {dist_str}\n"
                f"  goal_reached={episode_success}"
            )

            completed_steps += 1

            if episode_success:
                print("  [Metrics] Goal reached — stopping early.")
                break

    finally:
        write_run_meta(
            run_dir,
            args,
            task=task,
            scene_index=scene_index,
            extra={
                "completed_steps": completed_steps,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "action_history": action_history,
                "success": episode_success,
                "actual_path_length": actual_path_length,
                "shortest_path_length": shortest_path_length,
                "critic_aborted": critic_aborted,
            },
        )
        print(f"[Pipeline] Completed {completed_steps}/{args.steps} cycles -> {run_dir}")
        controller.stop()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[Pipeline] Removed temp dir: {temp_dir}")

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
        steps_taken=completed_steps,
    )
    return result, log_lines


def run_pipeline(args):
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
            print(f"\n{'#' * 60}\n[Eval] Scene {scene_idx} | {task_id}: {task}\n{'#' * 60}")
            result, log_lines = run_single_task(task, scene_idx, task_id, args)
            all_results.append(result)

            # Save per-task log alongside the run_dir (which is already inside output_dir)
            task_log_dir = os.path.join(args.output_dir, f"scene_{scene_idx}", task_id)
            os.makedirs(task_log_dir, exist_ok=True)
            with open(os.path.join(task_log_dir, "pipeline_run.txt"), "w") as f:
                f.write("\n".join(log_lines))
            print(f"[Eval] Done — success={result.success}  SPL_ep contribution logged.")

        print(f"\n[Eval] All tasks complete. Outputs in {args.output_dir}/")
        print_metrics(all_results)

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
            args.task, args.scene_index, args.task_id or "T0", args
        )
        log_path = os.path.join(args.output_dir, "pipeline_run.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"\n[Pipeline] success={result.success}  "
              f"actual_path={result.actual_path_length:.2f}m  "
              f"shortest_path={result.shortest_path_length}m")
        print(f"[Pipeline] Log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current navigation agent integration pipeline.")
    parser.add_argument("--task", default="find a flower vase")
    parser.add_argument("--task-id", default="T0", help="Task ID label for single-task mode.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--scene-index", type=int, default=5)
    parser.add_argument("--in-context", action="store_true", help="Use in-context example prompting for the planner (disabled by default for cleaner ablations).")
    parser.add_argument(
        "--output-dir",
        default="pipeline_outputs",
        help="Parent dir for run outputs. Each run is saved to a retained "
        "timestamped+labeled subdir, e.g. pipeline_outputs/<ts>_<task>_scene<N>.",
    )
    parser.add_argument("--eval-tasks", default=None, metavar="PATH",
                        help="Path to eval_tasks.txt — runs all scenes/tasks in the file.")
    parser.add_argument("--start-from", default=None, metavar="SCENE:TASKID",
                        help="Skip tasks before this point, e.g. '120:T3' starts at scene 120 T3.")
    parser.add_argument("--florence-model", default=FLORENCE_MODEL_ID)
    parser.add_argument("--sam-weights", default=SAM_WEIGHTS)
    parser.add_argument("--tritonai", action="store_true", help="Use TritonAI-hosted models for planner and critic.")
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

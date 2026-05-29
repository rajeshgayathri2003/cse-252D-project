"""Compute SR and SPL from existing baseline_run.txt logs.

Shortest-path length is approximated as the straight-line Euclidean distance
from the agent's spawn position to the nearest matching target object, measured
at episode start (before any action is taken).

Usage:
    uv run python compute_spl.py --output-dir baseline_outputs_3 \
                                  --eval-tasks eval_scenes/eval_tasks.txt
"""

import argparse
import math
import os
import re

import prior
from ai2thor.controller import Controller

OBJECT_SYNONYMS = {
    "wash basin": "Sink", "washbasin": "Sink",
    "washing machine": "Clothes_Dryer",
    "toilet seat": "Toilet",
    "notebook": "Book", "book on the bed": "Book",
    "couch": "Sofa",
    "tv": "Television", "television": "Television",
    "refrigerator": "Fridge", "fridge": "Fridge",
    "trash bag": "GarbageBag", "garbage bag": "GarbageBag",
    "painting on the wall": "Painting", "painting": "Painting",
    "study desk": "Desk", "desk": "Desk",
}


def parse_target_type(task: str) -> str:
    match = re.search(r"navigate to (?:the|a|an)?\s*(.+)", task, re.IGNORECASE)
    raw = match.group(1).strip().lower() if match else task.lower()
    for phrase, obj_type in OBJECT_SYNONYMS.items():
        if phrase in raw:
            return obj_type
    return "".join(w.capitalize() for w in raw.split())


def parse_log(log_path: str) -> dict | None:
    """Extract success and actual_path from a baseline_run.txt."""
    with open(log_path) as f:
        text = f.read()
    m = re.search(r"Final: success=(\w+)\s+shortest_path=\S+\s+actual_path=([\d.]+)", text)
    if not m:
        return None
    return {
        "success": m.group(1) == "True",
        "actual_path": float(m.group(2)),
    }


def parse_eval_tasks(path: str) -> list[dict]:
    tasks, current_scene = [], None
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


def get_shortest_path_length(controller, target_obj_id: str) -> float | None:
    """Navmesh shortest path length via AI2-THOR GetShortestPath."""
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


def nearest_target_id(controller, target_type: str) -> str | None:
    tl = target_type.lower()
    candidates = [o for o in controller.last_event.metadata["objects"] if tl in o["objectType"].lower()]
    if not candidates:
        return None
    return min(candidates, key=lambda o: o["distance"])["objectId"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-tasks", required=True)
    args = parser.parse_args()

    tasks = parse_eval_tasks(args.eval_tasks)
    results = []

    dataset = prior.load_dataset("procthor-10k")

    for entry in tasks:
        scene_idx = entry["scene_index"]
        task_id = entry["task_id"]
        task = entry["task"]
        log_path = os.path.join(args.output_dir, f"scene_{scene_idx}", task_id, "baseline_run.txt")

        if not os.path.exists(log_path):
            print(f"[SKIP] No log found: {log_path}")
            continue

        parsed = parse_log(log_path)
        if parsed is None:
            print(f"[SKIP] Could not parse Final line in {log_path}")
            continue

        # Load scene briefly to get spawn position and target distance
        print(f"[Scene {scene_idx} {task_id}] Loading scene for shortest-path measurement...")
        house = dataset["train"][scene_idx]
        try:
            ctrl = Controller(scene=house, platform="CloudRendering", width=300, height=300)
        except Exception:
            ctrl = Controller(scene=house, width=300, height=300)

        ctrl.step("Pass")
        target_type = parse_target_type(task)
        target_id = nearest_target_id(ctrl, target_type)
        shortest = get_shortest_path_length(ctrl, target_id) if target_id else None
        if shortest is None:
            print(f"  [Metrics] Pathfinder failed for '{target_type}' (id={target_id})")
        ctrl.stop()

        results.append({
            "scene_index": scene_idx,
            "task_id": task_id,
            "task": task,
            "success": parsed["success"],
            "actual_path": parsed["actual_path"],
            "shortest_path": shortest,
        })

        status = "+" if parsed["success"] else "-"
        spl_ep = "n/a"
        l_str = f"{shortest:.2f}m" if shortest is not None else "n/a"
        if parsed["success"] and shortest is not None:
            denom = max(parsed["actual_path"], shortest)
            spl_ep = f"{shortest / denom:.3f}" if denom > 0 else "1.000"
        print(f"  [{status}] l={l_str}  p={parsed['actual_path']:.2f}m  SPL={spl_ep}")

    # Aggregate metrics
    n = len(results)
    sr = sum(r["success"] for r in results) / n if n else 0.0
    spl_total = 0.0
    for r in results:
        if r["success"] and r["shortest_path"] is not None:
            denom = max(r["actual_path"], r["shortest_path"])
            spl_total += r["shortest_path"] / denom if denom > 0 else 1.0
    spl = spl_total / n if n else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Episodes : {n}")
    print(f"  SR       : {sr:.3f}  ({sum(r['success'] for r in results)}/{n})")
    print(f"  SPL      : {spl:.3f}  (straight-line shortest-path approximation)")
    print(f"{'=' * 60}")
    for r in results:
        status = "+" if r["success"] else "-"
        l = r["shortest_path"]
        p = r["actual_path"]
        if l is not None:
            denom = max(p, l)
            spl_ep = f"{l / denom:.3f}" if denom > 0 else "1.000"
            path_info = f"l={l:.2f}m  p={p:.2f}m  SPL_ep={spl_ep}"
        else:
            path_info = f"l=n/a  p={p:.2f}m  SPL_ep=n/a"
        print(f"  [{status}] Scene {r['scene_index']} {r['task_id']}: {r['task'][:38]:<38}  {path_info}")

    # Save
    metrics_path = os.path.join(args.output_dir, "metrics_spl.txt")
    with open(metrics_path, "w") as f:
        f.write(f"SR={sr:.4f}\nSPL={spl:.4f}\n\nPer-episode:\n")
        for r in results:
            l = r["shortest_path"]
            p = r["actual_path"]
            spl_ep = (l / max(p, l)) if (r["success"] and l and max(p, l) > 0) else 0.0
            f.write(f"Scene {r['scene_index']} {r['task_id']}: success={r['success']}  l={l}  p={p:.3f}  SPL_ep={spl_ep:.3f}\n")
    print(f"\nSaved to {metrics_path}")


if __name__ == "__main__":
    main()

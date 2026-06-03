"""View pipeline run outputs: a fast terminal digest and an optional HTML report.

Reads the directories written by run_navigation_pipeline.py under
pipeline_outputs/<timestamp>_<task>_scene<N>/ and presents them in a readable
form. Does not modify any run output.

Usage:
    uv run python view_run.py                 # newest run, detailed digest
    uv run python view_run.py <substring>     # run dir matching <substring>
    uv run python view_run.py --list          # one line per run
    uv run python view_run.py [<sub>] --html [--overlays]   # write report.html
    uv run python view_run.py --output-dir DIR ...          # parent dir (default pipeline_outputs)
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = {"find", "a", "an", "the", "for", "go", "to", "please", "locate", "get", "of"}
_ACTION_RE = re.compile(r"action=['\"]([^'\"]+)['\"]")
_STEP_RE = re.compile(r"controller\.step\([^\n]*\)")
_POSITION_RE = re.compile(
    r"^Position:\s*x=(-?\d+\.?\d*)\s+y=(-?\d+\.?\d*)\s+z=(-?\d+\.?\d*)",
    re.MULTILINE,
)
_DISTANCE_RE = re.compile(r"^Distance to target:\s*(-?\d+\.?\d*)\s*m", re.MULTILINE)


@dataclass
class StepData:
    index: int
    step_dir: Path
    labels: list = field(default_factory=list)
    perception: str = ""
    plan: str = ""
    critic: dict | None = None
    selected_subgoal: str = ""
    action: str = ""
    has_overlay: bool = False
    position: tuple | None = None  # (x, y, z) after the action executed
    distance_m: float | None = None  # straight-line distance to target after the action

    @property
    def action_verb(self):
        m = _ACTION_RE.search(self.action or "")
        return m.group(1) if m else None

    @property
    def approved(self):
        return bool(self.critic and self.critic.get("approved") is True)


@dataclass
class RunData:
    run_dir: Path
    name: str
    meta: dict | None
    steps: list

    # ---- derived stats (factual, from saved files) ----

    def task(self):
        if self.meta and self.meta.get("task"):
            return self.meta["task"]
        # Derive from dir name: <ts>_<slug-task>_scene<N>  ->  "slug task"
        m = re.match(r"\d{8}_\d{6}_(.*?)(?:_scene\d+)?$", self.name)
        slug = m.group(1) if m else self.name
        return slug.replace("-", " ")

    def scene(self):
        if self.meta and self.meta.get("scene_index") is not None:
            return self.meta["scene_index"]
        m = re.search(r"_scene(\d+)$", self.name)
        return int(m.group(1)) if m else None

    def requested_steps(self):
        if self.meta and self.meta.get("steps") is not None:
            return self.meta["steps"]
        return len(self.steps)

    def completed_steps(self):
        if self.meta and self.meta.get("completed_steps") is not None:
            return self.meta["completed_steps"]
        return len(self.steps)

    def approval_count(self):
        return sum(1 for s in self.steps if s.approved)

    def action_breakdown(self):
        counts = {}
        for s in self.steps:
            verb = s.action_verb
            if verb:
                counts[verb] = counts.get(verb, 0) + 1
        return counts

    def unique_labels(self):
        seen = set()
        for s in self.steps:
            seen.update(s.labels)
        return sorted(seen)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _parse_labels(perception_path):
    """Labels from perception.txt lines like '- chair located in the ...'."""
    if not perception_path.exists():
        return []
    labels = []
    for line in perception_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("- ") and " located" in line:
            labels.append(line[2:].split(" located")[0].strip())
    return labels


def _section(text, header, next_headers):
    """Return the block under `header` up to the next of `next_headers`."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == header)
    except StopIteration:
        return ""
    out = []
    for l in lines[start + 1:]:
        if l.strip() in next_headers:
            break
        out.append(l)
    return "\n".join(out).strip()


def _parse_result(result_path):
    """Parse pipeline_result.txt into (plan, critic, selected_subgoal, action)."""
    if not result_path.exists():
        return "", None, "", ""
    text = result_path.read_text()
    headers = {"Task:", "Perception:", "Map:", "Planner proposed:",
               "Critic verdict:", "Selected subgoal:", "Action executed:"}

    plan = _section(text, "Planner proposed:", headers)
    selected = _section(text, "Selected subgoal:", headers)

    critic_block = _section(text, "Critic verdict:", headers)
    critic = None
    if critic_block:
        try:
            critic = ast.literal_eval(critic_block.splitlines()[0])
        except (ValueError, SyntaxError):
            critic = None

    # Action: last controller.step(...) line — robust to the duplicate
    # "Action executed:" header bug in some runs.
    matches = _STEP_RE.findall(text)
    action = matches[-1] if matches else ""

    return plan, critic, selected, action


def _parse_position_distance(result_path):
    """Extract (position, distance_m) from a step's pipeline_result.txt."""
    if not result_path.exists():
        return None, None
    text = result_path.read_text()
    position = None
    m = _POSITION_RE.search(text)
    if m:
        position = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    distance = None
    md = _DISTANCE_RE.search(text)
    if md:
        distance = float(md.group(1))
    return position, distance


def parse_run(run_dir):
    run_dir = Path(run_dir)
    meta = None
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = None

    steps = []
    for step_dir in sorted(run_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        idx = int(step_dir.name.split("_")[-1])
        result_path = step_dir / "pipeline_result.txt"
        plan, critic, selected, action = _parse_result(result_path)
        position, distance_m = _parse_position_distance(result_path)
        steps.append(StepData(
            index=idx,
            step_dir=step_dir,
            labels=_parse_labels(step_dir / "perception.txt"),
            perception=(step_dir / "perception.txt").read_text() if (step_dir / "perception.txt").exists() else "",
            plan=plan,
            critic=critic,
            selected_subgoal=selected,
            action=action,
            has_overlay=(step_dir / "frame_overlay.jpg").exists(),
            position=position,
            distance_m=distance_m,
        ))

    return RunData(run_dir=run_dir, name=run_dir.name, meta=meta, steps=steps)


def find_runs(base_dir):
    """Run directories under base_dir, newest first (by dir-name timestamp sort)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = [d for d in base.iterdir() if d.is_dir() and re.match(r"\d{8}_\d{6}_", d.name)]
    return sorted(runs, key=lambda d: d.name, reverse=True)


def target_perceived(run_data):
    """Was the task's target object ever detected? Returns (bool, keyword_used)."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", run_data.task().lower()) if t and t not in STOPWORDS]
    keyword = " ".join(tokens)
    target_token = tokens[-1] if tokens else ""  # most specific noun, e.g. "vase"
    if not target_token:
        return False, keyword
    perceived = any(target_token in lab.lower() for lab in run_data.unique_labels())
    return perceived, keyword


# --------------------------------------------------------------------------
# CLI rendering
# --------------------------------------------------------------------------

def _short_action(action):
    verb = _ACTION_RE.search(action or "")
    verb = verb.group(1) if verb else (action or "?")
    pos = re.search(r"x=([\d.\-]+),\s*y=[\d.\-]+,\s*z=([\d.\-]+)", action or "")
    if pos:
        return f"{verb}({pos.group(1)},{pos.group(2)})"
    return verb


def render_cli(run_data):
    lines = []
    scene = run_data.scene()
    scene_str = f"scene {scene}" if scene is not None else "scene ?"
    backend = "tritonai" if (run_data.meta and run_data.meta.get("tritonai")) else "openai"
    perceived, keyword = target_perceived(run_data)

    lines.append(f"RUN  {run_data.name}")
    lines.append(f"task  {run_data.task()}   {scene_str}   {backend}   "
                 f"{run_data.completed_steps()}/{run_data.requested_steps()} steps")
    lines.append("-" * 72)
    lines.append(f"{'#':<3}{'perception (labels)':<30}{'critic':<10}{'action':<14}{'pos(x,z)':<14}dist")
    for s in run_data.steps:
        labs = ",".join(s.labels[:3]) or "-"
        if len(labs) > 28:
            labs = labs[:27] + "…"
        verdict = "APPROVED" if s.approved else ("revised" if s.critic else "?")
        pos_str = f"{s.position[0]:.2f},{s.position[2]:.2f}" if s.position else "-"
        dist_str = f"{s.distance_m:.2f}m" if s.distance_m is not None else "-"
        lines.append(
            f"{s.index:<3}{labs:<30}{verdict:<10}{_short_action(s.action):<14}"
            f"{pos_str:<14}{dist_str}"
        )
    lines.append("-" * 72)

    bd = run_data.action_breakdown()
    bd_str = "  ".join(f"{v}:{n}" for v, n in sorted(bd.items()))
    lines.append(f"critic approvals {run_data.approval_count()}/{len(run_data.steps)}   "
                 f"unique labels {len(run_data.unique_labels())}")
    lines.append(f"actions  {bd_str}")
    flag = "" if perceived else "  ⚠"
    lines.append(f"target '{keyword}' perceived?  {'YES' if perceived else 'NO'}{flag}")
    return "\n".join(lines)


def render_list(base_dir):
    runs = find_runs(base_dir)
    if not runs:
        return f"No runs found under {base_dir}/"
    lines = [f"{'RUN DIRECTORY':<44}{'SC':<4}{'STEPS':<8}{'TARGET?'}"]
    for rd in runs:
        run = parse_run(rd)
        scene = run.scene()
        perceived, _ = target_perceived(run)
        steps = f"{run.completed_steps()}/{run.requested_steps()}"
        lines.append(f"{run.name:<44}{str(scene if scene is not None else '?'):<4}"
                     f"{steps:<8}{'yes' if perceived else 'no'}")
    lines.append(f"\n{len(runs)} run(s) · use `view_run.py <dir>` for detail")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

def _ensure_overlays(run_data):
    """Generate any missing frame_overlay.jpg via the existing renderer."""
    try:
        from render_perception_overlay import render as render_overlay
    except Exception as e:  # noqa: BLE001
        print(f"[view_run] overlay renderer unavailable ({e}); skipping overlays", file=sys.stderr)
        return
    for s in run_data.steps:
        if (s.step_dir / "frame.jpg").exists() and not (s.step_dir / "frame_overlay.jpg").exists():
            try:
                render_overlay(str(s.step_dir))
                s.has_overlay = True
            except Exception as e:  # noqa: BLE001
                print(f"[view_run] overlay failed for {s.step_dir.name}: {e}", file=sys.stderr)


def _esc(text):
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(run_data, overlays=False):
    if overlays:
        _ensure_overlays(run_data)

    perceived, keyword = target_perceived(run_data)
    scene = run_data.scene()
    traj = [s.distance_m for s in run_data.steps if s.distance_m is not None]
    traj_str = " → ".join(f"{d:.2f}m" for d in traj) if traj else "(no distance recorded)"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_esc(run_data.name)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#f7f7f8;color:#222}"
        ".step{display:flex;gap:1rem;background:#fff;border:1px solid #ddd;border-radius:8px;"
        "padding:1rem;margin:1rem 0}.imgs img{max-width:300px;border-radius:4px;display:block;margin-bottom:.5rem}"
        ".meta{flex:1}.k{color:#666;font-size:.85rem}.approved{color:#197;font-weight:600}"
        ".revised{color:#c60;font-weight:600}pre{white-space:pre-wrap;font-size:.85rem}"
        ".pose{font-family:ui-monospace,monospace;font-size:.85rem;color:#333;"
        "background:#f3f3f5;padding:.3rem .5rem;border-radius:4px;display:inline-block}"
        ".traj{font-family:ui-monospace,monospace;font-size:.85rem;background:#fff;"
        "border:1px solid #ddd;border-radius:6px;padding:.5rem .75rem;margin-top:.5rem;"
        "word-break:break-all}"
        "header{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem 1.5rem}</style></head><body>",
        "<header>",
        f"<h2>{_esc(run_data.task())}</h2>",
        f"<p>{_esc(run_data.name)} · scene {scene if scene is not None else '?'} · "
        f"{run_data.completed_steps()}/{run_data.requested_steps()} steps · "
        f"critic {run_data.approval_count()}/{len(run_data.steps)} approved · "
        f"target '<b>{_esc(keyword)}</b>' perceived: <b>{'YES' if perceived else 'NO'}</b></p>",
        f"<p class='k'>distance trajectory</p><div class='traj'>{traj_str}</div>",
        "</header>",
    ]
    for s in run_data.steps:
        imgs = ""
        if (s.step_dir / "frame.jpg").exists():
            imgs += f"<img src='step_{s.index:02d}/frame.jpg' alt='frame {s.index}'>"
        if (s.step_dir / "frame_overlay.jpg").exists():
            imgs += f"<img src='step_{s.index:02d}/frame_overlay.jpg' alt='overlay {s.index}'>"
        verdict_cls = "approved" if s.approved else "revised"
        verdict_txt = "APPROVED" if s.approved else ("REVISED" if s.critic else "?")
        reason = _esc(s.critic.get("reason", "")) if s.critic else ""
        pose_bits = []
        if s.position:
            pose_bits.append(
                f"pos x={s.position[0]:.2f} y={s.position[1]:.2f} z={s.position[2]:.2f}"
            )
        if s.distance_m is not None:
            pose_bits.append(f"dist {s.distance_m:.2f}m")
        pose_html = (
            f"<p><span class='pose'>{_esc(' · '.join(pose_bits))}</span></p>"
            if pose_bits else ""
        )
        parts.append(
            f"<div class='step'><div class='imgs'>{imgs or '<span class=k>(no frame)</span>'}</div>"
            f"<div class='meta'><h3>Step {s.index:02d}</h3>"
            f"{pose_html}"
            f"<p class='k'>perception</p><pre>{_esc(s.perception)}</pre>"
            f"<p class='k'>planner</p><pre>{_esc(s.plan)}</pre>"
            f"<p class='k'>critic</p><p class='{verdict_cls}'>{verdict_txt}</p><pre>{reason}</pre>"
            f"<p class='k'>action</p><pre>{_esc(s.action)}</pre></div></div>"
        )
    parts.append("</body></html>")

    out_path = run_data.run_dir / "report.html"
    out_path.write_text("\n".join(parts))
    return out_path


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------

def _resolve_run(base_dir, substring):
    runs = find_runs(base_dir)
    if not runs:
        print(f"No runs found under {base_dir}/", file=sys.stderr)
        raise SystemExit(1)
    if not substring:
        return runs[0]  # newest
    matches = [r for r in runs if substring in r.name]
    if not matches:
        print(f"No run matching '{substring}' under {base_dir}/", file=sys.stderr)
        raise SystemExit(1)
    if len(matches) > 1:
        print(f"Ambiguous '{substring}' matches:", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        raise SystemExit(1)
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="View pipeline run outputs.")
    parser.add_argument("run", nargs="?", help="Substring of a run dir (default: newest).")
    parser.add_argument("--list", action="store_true", help="List all runs, one line each.")
    parser.add_argument("--html", action="store_true", help="Write report.html into the run dir.")
    parser.add_argument("--overlays", action="store_true",
                        help="With --html: generate missing segmentation overlays first.")
    parser.add_argument("--output-dir", default="pipeline_outputs",
                        help="Parent dir of runs (default: pipeline_outputs).")
    args = parser.parse_args(argv)

    if args.list:
        print(render_list(args.output_dir))
        return

    run_dir = _resolve_run(args.output_dir, args.run)
    run = parse_run(run_dir)

    if args.html:
        out = render_html(run, overlays=args.overlays)
        print(f"Wrote {out}")
    else:
        print(render_cli(run))


if __name__ == "__main__":
    main()

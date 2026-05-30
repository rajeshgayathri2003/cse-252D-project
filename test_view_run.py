"""Tests for view_run.py — the pipeline output viewer.

Run with:  uv run python test_view_run.py

Uses hermetic synthetic run directories built in a tempdir (deterministic, no
dependency on machine state), plus a light smoke check against any real runs in
pipeline_outputs/.  No pytest dependency — plain asserts, like verify_mapping.py.
"""

import json
import tempfile
from pathlib import Path

import view_run


# --------------------------------------------------------------------------
# Fixture builder: writes files in the exact format run_navigation_pipeline.py
# produces, so the parser is tested against reality.
# --------------------------------------------------------------------------

def _write_step(step_dir, labels, plan, approved, action, dup_action_header=False):
    step_dir.mkdir(parents=True, exist_ok=True)

    perception_lines = ["Visible Objects (Segmented):"]
    for lab in labels:
        perception_lines.append(
            f"- {lab} located in the center, covering roughly 1.0% of the view."
        )
    perception = "\n".join(perception_lines)
    (step_dir / "perception.txt").write_text(perception)

    verdict = {
        "approved": approved,
        "reason": "test reason",
        "revised_subgoal": None,
    }
    action_block = "Action executed:\n"
    if dup_action_header:  # reproduce the teammate dup-header bug
        action_block += "Action executed:\n"
    action_block += action + "\n"

    result = (
        f"Task:\nfind a flower vase\n\n"
        f"Perception:\n{perception}\n\n"
        f"Map:\nSpatial map summary:\n- nodes: 1\n\n"
        f"Planner proposed:\n{plan}\n\n"
        f"Critic verdict:\n{verdict}\n\n"
        f"Selected subgoal:\n{plan}\n\n"
        f"{action_block}"
    )
    (step_dir / "pipeline_result.txt").write_text(result)
    # frame.jpg / frame.txt intentionally omitted for most steps (parser must cope)


def _build_run(base, name, steps, meta=None):
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    for i, st in enumerate(steps):
        _write_step(run_dir / f"step_{i:02d}", **st)
    return run_dir


# Two representative steps: a Teleport (approved) and a MoveAhead (approved),
# one of which carries the duplicate "Action executed:" header.
SAMPLE_STEPS = [
    dict(labels=["chair", "table"], plan="look at the table",
         approved=True, action="controller.step(action='Teleport', position=dict(x=4.9, y=1, z=3.6))"),
    dict(labels=["table", "window"], plan="move_to(table)",
         approved=True, action="controller.step(action='MoveAhead')",
         dup_action_header=True),
    dict(labels=["coffee table", "sofa bed"], plan="keep looking",
         approved=False, action="controller.step(action='RotateLeft')"),
]

SAMPLE_META = {
    "run_dir": "20260101_000000_find-a-flower-vase_scene5",
    "task": "find a flower vase",
    "scene_index": 5,
    "steps": 3,
    "completed_steps": 3,
    "tritonai": True,
}


def test_parse_run_step_count():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert len(run.steps) == 3, len(run.steps)


def test_parse_run_labels():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert run.steps[0].labels == ["chair", "table"], run.steps[0].labels
        assert run.steps[2].labels == ["coffee table", "sofa bed"], run.steps[2].labels


def test_parse_run_critic_verdict():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert run.steps[0].critic["approved"] is True
        assert run.steps[2].critic["approved"] is False


def test_parse_run_action_extraction():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert "Teleport" in run.steps[0].action, run.steps[0].action
        # dup "Action executed:" header must NOT break extraction
        assert run.steps[1].action == "controller.step(action='MoveAhead')", run.steps[1].action


def test_action_breakdown():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        bd = run.action_breakdown()
        assert bd == {"Teleport": 1, "MoveAhead": 1, "RotateLeft": 1}, bd


def test_approval_count():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert run.approval_count() == 2, run.approval_count()


def test_unique_labels():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        assert run.unique_labels() == ["chair", "coffee table", "sofa bed", "table", "window"], run.unique_labels()


def test_target_perceived_false_for_vase():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        perceived, keyword = view_run.target_perceived(run)
        assert perceived is False, "no vase label present"
        assert "vase" in keyword, keyword  # keyword derived from task, transparent


def test_target_perceived_true_when_vase_seen():
    steps = SAMPLE_STEPS + [dict(labels=["vase"], plan="found it",
                                 approved=True, action="controller.step(action='MoveAhead')")]
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], steps, SAMPLE_META)
        run = view_run.parse_run(rd)
        perceived, keyword = view_run.target_perceived(run)
        assert perceived is True, "vase label present"


def test_parse_run_without_meta():
    """Legacy run with no run_meta.json: task/scene derived from dir name, steps from fs."""
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, "20251212_120000_find-a-flower-vase_scene7", SAMPLE_STEPS, meta=None)
        run = view_run.parse_run(rd)
        assert run.meta is None
        assert len(run.steps) == 3
        assert "find a flower vase" in run.task().lower(), run.task()
        assert run.completed_steps() == 3


def test_render_cli_contains_key_facts():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        out = view_run.render_cli(run)
        assert "find a flower vase" in out
        assert "Teleport" in out
        assert "scene" in out.lower()


def test_render_cli_truncates_long_labels():
    """A step with many long labels must not blow out the column width."""
    steps = [dict(labels=["coffee table", "studio couch", "studio couch"],
                  plan="x", approved=True,
                  action="controller.step(action='MoveAhead')")]
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], steps, SAMPLE_META)
        run = view_run.parse_run(rd)
        out = view_run.render_cli(run)
        data_rows = [l for l in out.splitlines() if l[:2].strip().isdigit()]
        assert data_rows, "no data rows rendered"
        # raw label join is 38 chars; rendered must be truncated with an ellipsis
        assert any("…" in l for l in data_rows), data_rows
        assert max(len(l) for l in data_rows) <= 80, max(len(l) for l in data_rows)


def test_render_html_writes_file_and_references_steps():
    with tempfile.TemporaryDirectory() as base:
        rd = _build_run(base, SAMPLE_META["run_dir"], SAMPLE_STEPS, SAMPLE_META)
        run = view_run.parse_run(rd)
        out = view_run.render_html(run, overlays=False)  # no frames in fixture -> no overlay gen
        assert out.exists() and out.name == "report.html"
        html = out.read_text()
        assert "find a flower vase" in html
        assert "Step 00" in html and "Step 02" in html
        assert "APPROVED" in html  # step 0/1 approved
        assert "REVISED" in html   # step 2 not approved


def test_find_runs_newest_first():
    with tempfile.TemporaryDirectory() as base:
        _build_run(base, "20260101_000000_find-a-flower-vase_scene5", SAMPLE_STEPS, SAMPLE_META)
        _build_run(base, "20260202_000000_find-a-flower-vase_scene5", SAMPLE_STEPS, SAMPLE_META)
        runs = view_run.find_runs(base)
        assert len(runs) == 2
        assert runs[0].name.startswith("20260202"), [r.name for r in runs]  # newest first


def _smoke_real_runs():
    """Parse every real run in pipeline_outputs/ without error (no assertions on values)."""
    base = Path("pipeline_outputs")
    if not base.exists():
        print("  (skip real-run smoke: no pipeline_outputs/)")
        return
    runs = view_run.find_runs(base)
    for rd in runs:
        run = view_run.parse_run(rd)
        print(f"  smoke ok: {run.name} ({len(run.steps)} steps, {run.approval_count()} approved)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("\n--- real-run smoke ---")
    try:
        _smoke_real_runs()
    except Exception as e:
        failed += 1
        print(f"ERROR smoke: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} unit tests passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

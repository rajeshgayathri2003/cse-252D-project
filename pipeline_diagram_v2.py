"""
pipeline_diagram.py
--------------------
Generates pipeline_diagram.png in the Image-2 style:
  - Large annotated frame on the LEFT
  - Stacked horizontal agent cards on the RIGHT (Mapping → Planning → Critic → Action)

Usage:
    python pipeline_diagram.py              # uses step_02 by default
    python pipeline_diagram.py --step 04   # any step
"""

import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "pipeline_outputs_slides")
VIZ_DIR = os.path.dirname(__file__)

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
CARD_BG = "#0f1923"
TEXT    = "#e8f4f8"
MONO    = "monospace"

# (title, header_fill, border_color)
CARD_META = [
    ("Mapping",          "#4a148c", "#9c27b0"),
    ("Planning",         "#e65100", "#ff9800"),
    ("Critic",           "#1b5e20", "#4caf50"),
    ("Action",           "#311b92", "#7c4dff"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def parse_outputs(step: str, run_dir: str | None = None) -> dict:
    """Parse perception.txt and pipeline_result.txt for a given step."""
    base     = os.path.join(run_dir or OUT_DIR, f"step_{step}")
    perc_raw = read_file(os.path.join(base, "perception.txt"))
    result   = read_file(os.path.join(base, "pipeline_result.txt"))

    # ── Task ──────────────────────────────────────────────────────────────────
    task_m = re.search(r"Task:\n(.+)", result)
    task_name = task_m.group(1).strip() if task_m else "Navigation"

    # ── Perception: top objects by coverage ──────────────────────────────────
    rows = []
    for m in re.finditer(
        r"- ([\w ]+?) located in ([^,]+), covering roughly ([\d.]+)%", perc_raw
    ):
        obj = m.group(1).strip()
        loc = m.group(2).strip()
        cov = float(m.group(3))
        rows.append((cov, obj, loc))
    rows.sort(reverse=True)
    perc_lines = [f"{obj}  {cov:.1f}%  ({loc})" for cov, obj, loc in rows[:4]]
    perc_text  = "\n".join(perc_lines)
    # caption below image
    top = rows[0] if rows else (0, "?", "?")
    extras = len(rows) - 1
    perc_caption = (
        f"Florence-2 + SAM2: {top[1]} ({top[2][0].upper()+top[2][1:]}, "
        f"{top[0]:.1f}% of view)"
        + (f" and {extras} more object{'s' if extras > 1 else ''} detected" if extras else "")
    )

    # ── Mapping ───────────────────────────────────────────────────────────────
    map_raw = re.search(
        r"Map:\n(.*?)\n\nPlanner proposed:", result, re.DOTALL
    ).group(1).strip()
    nodes = re.search(r"nodes: (\d+)", map_raw).group(1)
    edges = re.search(r"edges: (\d+)", map_raw).group(1)
    pos   = re.search(r"current_node: (\([^)]+\))", map_raw).group(1)
    objs  = re.findall(r"  - ([\w ]+?):", map_raw)
    map_text = (
        f"nodes: {nodes}  |  edges: {edges}\n"
        f"current_node: {pos}\n"
        f"known: {', '.join(objs[:6])}"
        + ("…" if len(objs) > 6 else "")
    )

    # ── Planning ──────────────────────────────────────────────────────────────
    plan_raw = re.search(
        r"Planner proposed:\n(.*?)\n\nCritic verdict:", result, re.DOTALL
    ).group(1).strip()
    items = re.findall(r"\d+\.\s+\*\*([^*]+)\*\*[:\s]*([^\n]*)", plan_raw)
    if items:
        parts = []
        for h, d in items[:3]:
            line = f"{h.strip()}: {d.strip()}" if d.strip() else h.strip()
            parts.append(textwrap.fill(line, width=60))
        plan_text = "\n".join(parts)
    else:
        plain = [l.strip("- •").strip() for l in plan_raw.split("\n") if l.strip()][:3]
        plan_text = "\n".join(textwrap.fill(l, width=60) for l in plain)

    # ── Critic ────────────────────────────────────────────────────────────────
    critic_raw = re.search(
        r"Critic verdict:\n(.*?)\n\nSelected subgoal:", result, re.DOTALL
    ).group(1).strip()
    approved = re.search(r"'approved': (True|False)", critic_raw)
    reason   = re.search(r"'reason': '([^']+)'", critic_raw)
    verdict  = "APPROVED" if (approved and approved.group(1) == "True") else "REJECTED"
    reason_raw  = reason.group(1) if reason else ""
    # Wrap at 46 so that the prepended 'reason: "' (10 chars) keeps each
    # line within ~56 chars total — preventing horizontal clip.
    reason_text = textwrap.fill(reason_raw, width=46)
    critic_title = f"Critic  ({verdict})"
    critic_text  = f"reason: \"{reason_text}\""

    # ── Action ────────────────────────────────────────────────────────────────
    action_text = re.search(
        r"Action executed:\n(.+)$", result, re.DOTALL
    ).group(1).strip()

    return {
        "perc_text":    perc_text,
        "perc_caption": perc_caption,
        "task_name": task_name,
        "cards": [
            ("Mapping",       "#4a148c", "#9c27b0", map_text),
            ("Planning",      "#e65100", "#ff9800", plan_text),
            (critic_title,    "#1b5e20", "#4caf50", critic_text),
            ("Action",        "#311b92", "#7c4dff", action_text),
        ],
    }


# ── Main drawing function ─────────────────────────────────────────────────────

def make_pipeline_diagram(step: str = "02", run_dir: str | None = None, out_path: str | None = None) -> None:
    src = run_dir or OUT_DIR
    data = parse_outputs(step, run_dir=src)

    overlay_path = os.path.join(src, f"step_{step}", "frame_overlay.jpg")
    raw_path     = os.path.join(src, f"step_{step}", "frame.jpg")
    frame_img = np.array(
        Image.open(overlay_path if os.path.exists(overlay_path) else raw_path).convert("RGB")
    )

    fig = plt.figure(figsize=(16, 9), facecolor=BG)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.02, right=0.98, top=0.93, bottom=0.05,
        wspace=0.04,
        width_ratios=[1.05, 1.4],
    )

    fig.suptitle(
        f'Pipeline Outputs — Step {step}  |  Task: "{data["task_name"]}"',
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Left: RGB frame ───────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(frame_img)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#5c6bc0"); sp.set_linewidth(2.5)
    ax_img.set_facecolor(BG)
    ax_img.set_title(
        "Perception  —  Florence-2 + SAM2",
        color="white", fontsize=13, fontweight="bold", pad=0,
        backgroundcolor="#1a237e",
    )
    ax_img.text(
        0.5, -0.03, data["perc_caption"],
        transform=ax_img.transAxes, ha="center", va="top",
        fontsize=10, color="#b0bec5", family=MONO,
    )

    # ── Right: stacked cards ──────────────────────────────────────────────────
    n = len(data["cards"])
    right_gs = gridspec.GridSpecFromSubplotSpec(
        n, 1, subplot_spec=gs[0, 1], hspace=0.07,
    )

    HDR = 0.36   # header band = 36% of each card's height

    for i, (title, hdr_col, bdr_col, content) in enumerate(data["cards"]):
        ax = fig.add_subplot(right_gs[i, 0])
        ax.set_facecolor(CARD_BG)
        ax.axis("off")

        # card outline
        ax.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            transform=ax.transAxes, clip_on=False,
            boxstyle="round,pad=0.01",
            facecolor=CARD_BG, edgecolor=bdr_col, linewidth=2.0, zorder=1,
        ))

        # header band
        ax.add_patch(FancyBboxPatch(
            (0.0, 1.0 - HDR), 1.0, HDR,
            transform=ax.transAxes, clip_on=False,
            boxstyle="square,pad=0.0",
            facecolor=hdr_col, edgecolor="none", zorder=2,
        ))

        # title
        ax.text(
            0.015, 1.0 - HDR / 2, title,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=4,
        )

        # content
        ax.text(
            0.015, 1.0 - HDR - 0.07, content,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=TEXT, family=MONO,
            linespacing=1.6, zorder=4, clip_on=True,
        )

    out = out_path or os.path.join(VIZ_DIR, "pipeline_diagram.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved  →  {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="02",
                        help="Step number to visualise, e.g. '02' or '04'")
    parser.add_argument("--run-dir", default=None,
                        help="Path to run directory containing step_XX/ subdirs")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: pipeline_diagram.png next to this script)")
    args = parser.parse_args()
    make_pipeline_diagram(step=args.step, run_dir=args.run_dir, out_path=args.out)

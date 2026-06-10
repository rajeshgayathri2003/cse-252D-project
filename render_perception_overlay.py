"""Render segmentation overlays onto frame.jpg → frame_overlay.jpg.

Called by view_run.py as:
    from render_perception_overlay import render
    render(step_dir)   # step_dir is a path string to e.g. step_01/
"""

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Distinct colours for up to 20 classes (RGBA with alpha for fill)
_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 54, 162, 235), ( 75, 192, 140),
    (153, 102, 255), (255, 206,  86), ( 23, 190, 207), (214,  39,  40),
    (148, 103, 189), ( 44, 160,  44), (188, 189,  34), ( 31, 119, 180),
    (255, 127,  14), (174, 199, 232), (255, 187, 120), (152, 223, 138),
    (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141),
]
_FILL_ALPHA = 90   # polygon fill transparency
_FONT_SIZE  = 10
_HUD_FONT_SIZE = 10


def _load_classes(run_dir: Path) -> dict[int, str]:
    """Read classes.txt → {class_id: label}."""
    classes_path = run_dir / "classes.txt"
    if not classes_path.exists():
        return {}
    mapping = {}
    for line in classes_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+):\s*(.+)$", line)
        if m:
            mapping[int(m.group(1))] = m.group(2).strip()
    return mapping


def _parse_masks(frame_txt: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Parse YOLO-polygon frame.txt → list of (class_id, [(x,y)...]) in 0-1 coords."""
    if not frame_txt.exists():
        return []
    masks = []
    for line in frame_txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:   # need class_id + at least 3 points
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        # coords are interleaved x y x y ...
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
        masks.append((class_id, points))
    return masks


def _hud_text(step_dir: Path) -> list[str]:
    """Build HUD lines from pipeline_result.txt."""
    result_path = step_dir / "pipeline_result.txt"
    if not result_path.exists():
        return []
    text = result_path.read_text()

    lines = []

    # Step index from dir name (step_01 → Step 1)
    m = re.search(r"step_(\d+)$", step_dir.name)
    if m:
        lines.append(f"Step {int(m.group(1))}")

    # Action
    action_m = re.search(r"action=['\"]([^'\"]+)['\"]", text)
    if action_m:
        lines.append(f"Action: {action_m.group(1)}")

    # Distance
    dist_m = re.search(r"Distance to target:\s*(-?\d+\.?\d*)\s*m", text, re.MULTILINE)
    if dist_m:
        lines.append(f"Dist to target: {float(dist_m.group(1)):.2f} m")

    # Critic verdict
    approved_m = re.search(r"'approved':\s*(True|False)", text)
    if approved_m:
        verdict = "APPROVED" if approved_m.group(1) == "True" else "REVISED"
        lines.append(f"Critic: {verdict}")

    return lines


def render(step_dir: str) -> None:
    """Draw polygon masks + HUD onto frame.jpg and save frame_overlay.jpg."""
    step_path = Path(step_dir)
    frame_path = step_path / "frame.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"frame.jpg not found in {step_dir}")

    run_dir = step_path.parent
    classes = _load_classes(run_dir)
    masks = _parse_masks(step_path / "frame.txt")

    img = Image.open(frame_path).convert("RGBA")
    W, H = img.size

    # Draw each polygon mask on a transparent overlay, then composite
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for class_id, norm_points in masks:
        color = _PALETTE[class_id % len(_PALETTE)]
        pixel_points = [(x * W, y * H) for x, y in norm_points]
        if len(pixel_points) < 3:
            continue
        fill = (*color, _FILL_ALPHA)
        outline = (*color, 220)
        draw.polygon(pixel_points, fill=fill, outline=outline)

        # Label at centroid
        cx = sum(p[0] for p in pixel_points) / len(pixel_points)
        cy = sum(p[1] for p in pixel_points) / len(pixel_points)
        label = classes.get(class_id, str(class_id))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", _FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((cx, cy), label, font=font, anchor="mm")
        pad = 3
        draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                       fill=(0, 0, 0, 160))
        draw.text((cx, cy), label, fill=(255, 255, 255, 255), font=font, anchor="mm")

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Draw HUD in top-left corner
    hud_lines = _hud_text(step_path)
    if hud_lines:
        draw2 = ImageDraw.Draw(img)
        try:
            hud_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _HUD_FONT_SIZE)
        except OSError:
            hud_font = ImageFont.load_default()

        x, y = 10, 10
        for line in hud_lines:
            tb = draw2.textbbox((x, y), line, font=hud_font)
            draw2.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=(0, 0, 0, 180))
            draw2.text((x, y), line, fill=(255, 255, 255), font=hud_font)
            y += tb[3] - tb[1] + 6

    out_path = step_path / "frame_overlay.jpg"
    img.save(out_path, quality=92)

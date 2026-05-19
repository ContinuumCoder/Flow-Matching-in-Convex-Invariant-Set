

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

CANVAS_W = 1902
CANVAS_H = 592
LEFT_W = 258
MARGIN_X = 24
GAP_X = 13
HEADER_Y = 36
ROW1_Y = 118
ROW2_Y = 348
FRAME_W = (CANVAS_W - LEFT_W - 2 * MARGIN_X - 3 * GAP_X) // 4
FRAME_H = 215

RED = (197, 28, 28)
GREEN = (0, 125, 61)
TEXT = (30, 30, 30)
MUTED = (45, 45, 45)
WHITE = (255, 255, 255)
GRID = (230, 230, 230)


@dataclass(frozen=True)
class FigureSpec:
    output: Path
    headers: tuple[str, str, str, str]
    top_label: tuple[str, str]
    bottom_label: tuple[str, str]
    top_frames: tuple[Path, Path, Path, Path]
    bottom_frames: tuple[Path, Path, Path, Path]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_HEADER = font(24, bold=True)
FONT_METHOD = font(24, bold=True)
FONT_OUTCOME = font(16, bold=False)


def cover_resize(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    new_size = (round(src_w * scale), round(src_h * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    left = (img.width - dst_w) // 2
    top = (img.height - dst_h) // 2
    return img.crop((left, top, left + dst_w, top + dst_h))


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, fill=TEXT) -> None:
    draw.text(xy, value, font=fnt, fill=fill)


def label_block(draw: ImageDraw.ImageDraw, y: int, method: str, outcome: str, color: tuple[int, int, int]) -> None:
    x = MARGIN_X
    text(draw, (x, y + 48), method, FONT_METHOD, color)
    text(draw, (x, y + 80), outcome, FONT_OUTCOME, MUTED)


def draw_panel(draw: ImageDraw.ImageDraw, x: int, y: int, img: Image.Image, final_color: tuple[int, int, int] | None) -> None:
    draw.rectangle((x - 1, y - 1, x + FRAME_W + 1, y + FRAME_H + 1), fill=WHITE, outline=GRID, width=1)
    canvas.paste(img, (x, y))
    if final_color is not None:
        draw.rectangle((x + 3, y + 3, x + FRAME_W - 4, y + FRAME_H - 4), outline=final_color, width=4)


def render(spec: FigureSpec) -> None:
    global canvas
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), WHITE)
    draw = ImageDraw.Draw(canvas)
    frame_x = [LEFT_W + MARGIN_X + i * (FRAME_W + GAP_X) for i in range(4)]

    for x, header in zip(frame_x, spec.headers):
        bbox = draw.textbbox((0, 0), header, font=FONT_HEADER)
        draw.text((x + (FRAME_W - (bbox[2] - bbox[0])) / 2, HEADER_Y), header, font=FONT_HEADER, fill=TEXT)

    label_block(draw, ROW1_Y, *spec.top_label, RED)
    label_block(draw, ROW2_Y, *spec.bottom_label, GREEN)

    for idx, path in enumerate(spec.top_frames):
        draw_panel(draw, frame_x[idx], ROW1_Y, cover_resize(path, (FRAME_W, FRAME_H)), RED if idx == 3 else None)

    for idx, path in enumerate(spec.bottom_frames):
        draw_panel(draw, frame_x[idx], ROW2_Y, cover_resize(path, (FRAME_W, FRAME_H)), GREEN if idx == 3 else None)

    canvas.save(spec.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figure_sheets")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lift_dir = ROOT / "results" / "isaaclab_lift_gen_compare_ikabs_feedback_wide_v1" / "frame_checks"
    drawer_direct_dir = ROOT / "results" / "isaaclab_drawer_visual_compare_front_v2_direct" / "frame_checks"
    drawer_oc_dir = ROOT / "results" / "isaaclab_drawer_visual_compare_camera_scout_front" / "frame_checks"

    render(
        FigureSpec(
            output=args.out_dir / "isaac_lift_direct_vs_phase_feedback.png",
            headers=("start", "approach", "grasp", "final"),
            top_label=("Direct generator", "failure: no cube lift"),
            bottom_label=("OC-FMCS", "success: cube lifted"),
            top_frames=(
                lift_dir / "direct_t0p6.png",
                lift_dir / "direct_t1p6.png",
                lift_dir / "direct_t2p6.png",
                lift_dir / "direct_t4p3.png",
            ),
            bottom_frames=(
                lift_dir / "feedback_t0p6.png",
                lift_dir / "feedback_t1p4.png",
                lift_dir / "feedback_t2p2.png",
                lift_dir / "feedback_t4p3.png",
            ),
        )
    )

    render(
        FigureSpec(
            output=args.out_dir / "drawer_direct_vs_ocfmcs_camera_comparison.png",
            headers=("approach", "contact", "pull", "final"),
            top_label=("Direct generator", "failure: drawer closed"),
            bottom_label=("OC-FMCS", "success: drawer opened"),
            top_frames=(
                drawer_direct_dir / "direct_gen_0.8.jpg",
                drawer_direct_dir / "direct_gen_2.0.jpg",
                drawer_direct_dir / "direct_gen_4.5.jpg",
                drawer_direct_dir / "direct_gen_7.5.jpg",
            ),
            bottom_frames=(
                drawer_oc_dir / "oc_fmcs_0.8.jpg",
                drawer_oc_dir / "oc_fmcs_2.0.jpg",
                drawer_oc_dir / "oc_fmcs_4.5.jpg",
                drawer_oc_dir / "oc_fmcs_7.5.jpg",
            ),
        )
    )


if __name__ == "__main__":
    main()

"""Clip renderer: one ffmpeg filter_complex per clip.

The sendcmd architecture (vendored from mutonby/openshorts punch_in.py +
reframe_v2.py, MIT): the director's per-frame trajectory array becomes a
deduped sendcmd command file driving a labeled crop filter — hard cuts are
just discontinuities in the same array, pans are smooth regions, punch-ins
already live in the w/h values. One decode, one encode:

    sendcmd → crop@c → scale 1080x1920 → subtitles burn → loudnorm

Deduping to change-points matters: a 45 s clip at 25 fps is 1125 frames and
writing every parameter every frame slows the filter measurably (openshorts'
own comment). Even dimensions everywhere — x264/NVENC reject odd ones.

Encoder tiers follow openshorts ffmpeg_utils.py: try hardware
(h264_videotoolbox on macOS), fall back to libx264, mapping quality between
CRF and the hardware encoder's bitrate model.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import ffmpeg_bin

OUT_W = 1080
OUT_H = 1920
X264_CRF = 19
VT_BITRATE = "10M"

_vt_checked: bool | None = None


def videotoolbox_available() -> bool:
    """Probe once: encode 0.2 s of black through h264_videotoolbox."""
    global _vt_checked
    if _vt_checked is None:
        proc = subprocess.run(
            [
                ffmpeg_bin.ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
                "-c:v", "h264_videotoolbox", "-f", "null", "-",
            ],
            capture_output=True, timeout=60,
        )
        _vt_checked = proc.returncode == 0
    return _vt_checked


def crop_boxes(frames: list[list[float]], src_w: int, src_h: int) -> list[tuple[int, int, int, int]]:
    """Director frames [x, y, w, h] → even-int (w, h, x, y) crop boxes,
    clamped in-bounds (openshorts crop_boxes rounding rules)."""
    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in frames:
        wi = max(2, min(int(w) - int(w) % 2, src_w))
        hi = max(2, min(int(h) - int(h) % 2, src_h))
        xi = max(0, min(int(round(x)), src_w - wi))
        yi = max(0, min(int(round(y)), src_h - hi))
        boxes.append((wi, hi, xi - xi % 2, yi - yi % 2))
    return boxes


def sendcmd_lines(boxes: list[tuple[int, int, int, int]], fps: float, target: str = "crop@c") -> list[str]:
    """Per-frame w/h/x/y commands, deduped to change-points (openshorts)."""
    lines: list[str] = []
    prev: tuple[int, int, int, int] | None = None
    for i, box in enumerate(boxes):
        if box == prev:
            continue
        t = i / fps
        w, h, x, y = box
        pw, ph, px, py = prev if prev else (None, None, None, None)
        if w != pw:
            lines.append(f"{t:.4f} {target} w {w};")
        if h != ph:
            lines.append(f"{t:.4f} {target} h {h};")
        if x != px:
            lines.append(f"{t:.4f} {target} x {x};")
        if y != py:
            lines.append(f"{t:.4f} {target} y {y};")
        prev = box
    return lines


def _q(path: str) -> str:
    """ffmpeg filter-option quoting: single quotes make the value literal;
    an embedded quote closes, escapes, reopens ('\\'').

    Windows adds two wrinkles the mac path never sees: backslash is
    ffmpeg's escape character even inside quotes (av_get_token), and the
    drive-letter colon reads as an option separator on some parse levels.
    Forward slashes (fine for libass and every filter) plus an escaped
    colon is the canonical portable form: 'C\\:/Users/…/clip.ass'."""
    text = str(path)
    if os.name == "nt":
        text = text.replace("\\", "/").replace(":", "\\:")
    return "'" + text.replace("'", "'\\''") + "'"


def _piecewise_expr(pairs: list[tuple[float, int]], default: int) -> str:
    """Nested if(lt(t,T),v,...) expression for a crop parameter.

    ``pairs`` = [(change_time, value), ...] in ascending time order.
    Built back-to-front so the final value is the fallthrough:
        if(lt(t,T1),V1,if(lt(t,T2),V2,...,Vn))
    """
    expr = str(default)
    for t, v in reversed(pairs):
        expr = f"if(lt(t,{t:.4f}),{v},{expr})"
    return expr


def crop_expr(boxes: list[tuple[int, int, int, int]], fps: float) -> str:
    """Per-parameter piecewise time expressions replacing the sendcmd file.

    sendcmd (a command file driving a labeled crop) deadlocks the filter
    graph on some ffmpeg builds — the render hangs mid-encode with the
    output file frozen. The crop filter evaluates expressions per frame
    with ``t`` (frame PTS in seconds), so the same change-point trajectory
    can be expressed directly: no command file, no external state, no hang.
    """
    # Change-points per parameter, starting from boxes[0] at t=0.
    w_pairs: list[tuple[float, int]] = []
    h_pairs: list[tuple[float, int]] = []
    x_pairs: list[tuple[float, int]] = []
    y_pairs: list[tuple[float, int]] = []
    prev: tuple[int, int, int, int] | None = None
    for i, (w, h, x, y) in enumerate(boxes):
        t = i / fps
        if prev is None or w != prev[0]:
            w_pairs.append((t, w))
        if prev is None or h != prev[1]:
            h_pairs.append((t, h))
        if prev is None or x != prev[2]:
            x_pairs.append((t, x))
        if prev is None or y != prev[3]:
            y_pairs.append((t, y))
        prev = (w, h, x, y)

    w0, h0, x0, y0 = boxes[0]
    we = _piecewise_expr(w_pairs[1:], w0)
    he = _piecewise_expr(h_pairs[1:], h0)
    xe = _piecewise_expr(x_pairs[1:], x0)
    ye = _piecewise_expr(y_pairs[1:], y0)
    # crop evaluates its expressions per frame with `t` = frame PTS (s) —
    # no eval= option (that's an overlay/geq thing). Trajectory change-points
    # become nested if(lt(t,T),v,...) per parameter. No sendcmd file → no
    # filter-graph deadlock.
    return f"crop=w='{we}':h='{he}':x='{xe}':y='{ye}'"


def render_clip(
    media_path: str,
    out_path: Path,
    clip_start: float,
    clip_end: float,
    trajectory: dict,
    ass_path: Path | None,
    fonts_dir: Path | None,
    lufs: float = -14.0,
    true_peak: float = -1.0,
    src_w: int = 1920,
    src_h: int = 1080,
    timeout: float = 1800.0,
) -> None:
    duration = clip_end - clip_start
    boxes = crop_boxes(trajectory["frames"], src_w, src_h)
    if not boxes:
        boxes = [(src_h * 9 // 16 // 2 * 2, src_h - src_h % 2, 0, 0)]
    fps = float(trajectory.get("fps", 25))

    vf_parts = [
        crop_expr(boxes, fps),
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "setsar=1",
    ]
    if ass_path is not None:
        sub = f"subtitles=filename={_q(ass_path)}"
        if fonts_dir is not None:
            sub += f":fontsdir={_q(fonts_dir)}"
        vf_parts.append(sub)

    if videotoolbox_available():
        vcodec = ["-c:v", "h264_videotoolbox", "-b:v", VT_BITRATE, "-allow_sw", "1"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(X264_CRF)]

    args = [
        ffmpeg_bin.ffmpeg(), "-y", "-v", "error",
        "-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}",
        "-i", media_path,
        "-vf", ",".join(vf_parts),
        "-af", f"loudnorm=I={lufs}:TP={true_peak}:LRA=11",
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        "-map_metadata", "-1",  # metadata scrub (openshorts ffmpeg_utils)
        str(out_path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Render failed: {(proc.stderr or '')[-800:]}")


def verify_output(out_path: Path, expected_duration: float) -> dict:
    """Post-render sanity: exists, has both streams, duration within 1.5 s."""
    proc = subprocess.run(
        [
            ffmpeg_bin.ffprobe(), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(out_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    import json

    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    has_v = any(s.get("codec_type") == "video" for s in streams)
    has_a = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(info.get("format", {}).get("duration", 0.0))
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "ok": has_v and has_a and abs(duration - expected_duration) < 1.5,
        "duration": duration,
        "width": video.get("width"),
        "height": video.get("height"),
    }

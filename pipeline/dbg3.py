import json, subprocess, sys, shutil
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin
from pathlib import Path

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
chunks = renderer._chunk_boxes(boxes, fps)
ch = chunks[0]
expr = renderer.crop_expr(ch["boxes"], fps, 1280, 720)
fonts = renderer._clean_fonts_dir(Path("/opt/publikclip/pipeline/publikclip_pipeline/captions/fonts"), Path("/tmp"))
ass = "/root/.publikclip/jobs/20260819-064456-8d02da/clips/clip_00.ass"

def run(label, vf_extra):
    vf = [expr, "scale=1080:1920:flags=lanczos", "setsar=1"] + vf_extra
    args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-ss", "5.546", "-t", "3",
            "-i", "/tmp/vid1.MOV", "-vf", ",".join(vf),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "/tmp/dbg3_%s.mp4" % label]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    print("== %s rc=%d" % (label, p.returncode))
    if p.returncode:
        print("  ", p.stderr.strip().replace("\n", "\n   ")[-400:])

run("base", [])
run("setpts", ["setpts=PTS+5.5460"])
run("subs", ["subtitles=filename=%s:fontsdir=%s" % (renderer._q(ass), renderer._q(str(fonts)))])
run("subs+setpts", ["setpts=PTS+5.5460", "subtitles=filename=%s:fontsdir=%s" % (renderer._q(ass), renderer._q(str(fonts)))])
shutil.rmtree(fonts, ignore_errors=True)

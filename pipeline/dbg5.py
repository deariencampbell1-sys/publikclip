import json, subprocess, sys
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
ch0 = renderer._chunk_boxes(boxes, fps)[0]["boxes"]

def count_pairs(bs):
    xs = [b[2] for b in bs]; ys = [b[3] for b in bs]
    xc = sum(1 for a,b in zip(xs, xs[1:]) if a != b)
    yc = sum(1 for a,b in zip(ys, ys[1:]) if a != b)
    return xc, yc

for n in (100, 150, 200, 240, 286):
    bs = ch0[:n]
    xc, yc = count_pairs(bs)
    expr = renderer.crop_expr(bs, fps, 1280, 720)
    args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-t", "2", "-i", "/tmp/vid1.MOV",
            "-vf", ",".join([expr, "scale=1080:1920:flags=lanczos"]),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "/tmp/dbg5.mp4"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    print("n=%d xchanges=%d ychanges=%d rc=%d" % (n, xc, yc, p.returncode))

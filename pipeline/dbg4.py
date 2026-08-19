import json, subprocess, sys
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
chunks = renderer._chunk_boxes(boxes, fps)
ch = chunks[0]
expr = renderer.crop_expr(ch["boxes"], fps, 1280, 720)
print("chunk0 boxes:", len(ch["boxes"]), "expr:", expr[:120], "...")

def run(label, seek):
    args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error"]
    if seek: args += ["-ss", seek]
    args += ["-t", "3", "-i", "/tmp/vid1.MOV", "-vf", ",".join([expr, "scale=1080:1920:flags=lanczos"]),
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "/tmp/dbg4_%s.mp4" % label]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    print(label, "rc=", p.returncode)

run("noseek", None)
run("seek5546", "5.546")
run("seek1", "1.0")

import subprocess, sys
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin

def try_depth(depth):
    pairs = [(0.05 + i * 0.05, 200 + (i % 3)) for i in range(depth)]
    expr = "crop=w=404:h=720:x='%s':y=0" % renderer._piecewise_expr(pairs, 200)
    args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-t", "2", "-i", "/tmp/vid1.MOV",
            "-vf", ",".join([expr, "scale=1080:1920:flags=lanczos"]),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "/tmp/dbg8.mp4"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return p.returncode

for d in (60, 70, 80, 90, 100):
    print("depth", d, "rc", try_depth(d))

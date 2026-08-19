import json, subprocess, sys, shutil, time
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin
from pathlib import Path

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
cmd = "/tmp/dbg7.cmd"
open(cmd, "w").write("\n".join(renderer.sendcmd_lines(boxes, fps)) + "\n")
print("cmd lines:", sum(1 for _ in open(cmd)))
fonts = renderer._clean_fonts_dir(Path("/opt/publikclip/pipeline/publikclip_pipeline/captions/fonts"), Path("/tmp"))
ass = "/root/.publikclip/jobs/20260819-064456-8d02da/clips/clip_00.ass"
w0,h0,x0,y0 = boxes[0]
vf = ["sendcmd=f=%s" % cmd, "crop@c=w=%d:h=%d:x=%d:y=%d" % (w0,h0,x0,y0),
      "scale=1080:1920:flags=lanczos", "setsar=1",
      "subtitles=filename=%s:fontsdir=%s" % (renderer._q(ass), renderer._q(str(fonts)))]
args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-ss", "5.546", "-t", "20",
        "-i", "/tmp/vid1.MOV", "-vf", ",".join(vf),
        "-af", "loudnorm=I=-14.0:TP=-1.0:LRA=11",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-map_metadata", "-1", "/tmp/dbg7.mp4"]
t0=time.time()
p = subprocess.run(args, capture_output=True, text=True, timeout=120)
print("rc:", p.returncode, "elapsed:", round(time.time()-t0,1), "size:", __import__("os").path.getsize("/tmp/dbg7.mp4"))
if p.returncode: print(p.stderr[-300:])
shutil.rmtree(fonts, ignore_errors=True)

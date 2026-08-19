import json, subprocess, sys, time, shutil
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
chunks = renderer._chunk_boxes(boxes, fps)
ch = chunks[0]
expr = renderer.crop_expr(ch["boxes"], fps, 1280, 720)
fonts = renderer._clean_fonts_dir(__import__("pathlib").Path("/opt/publikclip/pipeline/publikclip_pipeline/captions/fonts"), __import__("pathlib").Path("/tmp"))
ass = "/root/.publikclip/jobs/20260819-064456-8d02da/clips/clip_00.ass"
vf = [expr, "setpts=PTS+5.5460", "scale=1080:1920:flags=lanczos", "setsar=1",
      "subtitles=filename=%s:fontsdir=%s" % (renderer._q(ass), renderer._q(str(fonts)))]
args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "info", "-ss", "5.546", "-t", "11.4",
        "-i", "/tmp/vid1.MOV", "-vf", ",".join(vf),
        "-af", "loudnorm=I=-14.0:TP=-1.0:LRA=11",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-map_metadata", "-1", "/tmp/dbg2.mp4"]
t0 = time.time()
p = subprocess.run(args, capture_output=True, text=True, timeout=90)
print("rc:", p.returncode, "elapsed:", round(time.time()-t0,1), "outsize:", __import__("os").path.getsize("/tmp/dbg2.mp4"))
err = [l for l in p.stderr.splitlines() if "error" in l.lower() or "frame=" in l.lower()][-6:]
print("\n".join(err))
shutil.rmtree(fonts, ignore_errors=True)

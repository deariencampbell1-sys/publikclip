import json, sys, time, shutil
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer
from pathlib import Path

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
ass = "/root/.publikclip/jobs/20260819-064456-8d02da/clips/clip_00.ass"
t0 = time.time()
renderer.render_clip(
    "/tmp/vid1.MOV", Path("/tmp/full_test.mp4"), 5.546, 43.232, traj,
    Path(ass), Path("/opt/publikclip/pipeline/publikclip_pipeline/captions/fonts"),
    src_w=1280, src_h=720, timeout=600,
)
print("RENDERED in %.1fs" % (time.time()-t0))
check = renderer.verify_output(Path("/tmp/full_test.mp4"), 43.232)
print("verify:", check)

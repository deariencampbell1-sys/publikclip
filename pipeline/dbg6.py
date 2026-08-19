import json, subprocess, sys
sys.path.insert(0, "/opt/publikclip/pipeline")
from publikclip_pipeline.render import renderer, ffmpeg_bin

traj = json.load(open("/root/.publikclip/jobs/20260819-064456-8d02da/trajectory_00.json"))
frames = traj["frames"]; fps = float(traj.get("fps", 30))
boxes = renderer.crop_boxes(frames, 1280, 720)
ch0 = renderer._chunk_boxes(boxes, fps)[0]["boxes"]
bs = ch0[:240]  # y changes in here

# Build the pieces manually
w0 = max(b[0] for b in bs); h0 = max(b[1] for b in bs)
x_pairs=[]; y_pairs=[]; prev=None
for i,(w,h,x,y) in enumerate(bs):
    t=i/fps
    cx=x+w/2; cy=y+h/2
    nx=min(max(int(round(cx-w0/2)),0),1280-w0); ny=min(max(int(round(cy-h0/2)),0),720-h0)
    nx-=nx%2; ny-=ny%2
    if prev is None or nx!=prev[0] or ny!=prev[1]:
        x_pairs.append((t,nx)); y_pairs.append((t,ny)); prev=(nx,ny)
print("w0", w0, "h0", h0, "xpairs", len(x_pairs), "ypairs", len(y_pairs), "yvals", sorted(set(v for _,v in y_pairs)))

xe = renderer._piecewise_expr(x_pairs[1:], x_pairs[0][1])
ye = renderer._piecewise_expr(y_pairs[1:], y_pairs[0][1])

def run(label, expr):
    args = [ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-t", "2", "-i", "/tmp/vid1.MOV",
            "-vf", ",".join([expr, "scale=1080:1920:flags=lanczos"]),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "/tmp/dbg6_%s.mp4" % label]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    print(label, "rc=", p.returncode)
    if p.returncode: print("   ", p.stderr.strip()[-200:].replace("\n", " | "))

run("xonly", "crop=w=%d:h=%d:x='%s':y=0" % (w0, h0, xe))
run("yonly", "crop=w=%d:h=%d:x=0:y='%s'" % (w0, h0, ye))
run("both", "crop=w=%d:h=%d:x='%s':y='%s'" % (w0, h0, xe, ye))

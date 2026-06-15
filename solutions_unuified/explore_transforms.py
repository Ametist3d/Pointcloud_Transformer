"""
Brute-force coordinate-system explorer.

Sweeps every constant-free formula of the form  L @ M @ R  or  L @ inv(M) @ R
where L, R are signed axis permutations (48 each).  That family covers every
90°-multiple rotation and every mirror between two coordinate frames — if a
universal axis-relabel solves the layout, it lives here.

Keys
----
  ] / [   next / previous          G / V   jump ±10        J   jump +100
  D       only det = +1 outputs    S       only forms with inv(M)
  X       export current formula + traj to out_explore/
  R       screenshot every currently-filtered candidate to out_explore/screens/
"""
import os
import csv
from itertools import permutations, product
import numpy as np
import open3d as o3d

DATA      = "input/Points"
TRAJ      = "input/traj.txt"
OUT       = "out_explore"
PLY_NAMES = ["image1.ply", "image2.ply", "image3.ply"]
OPTIMIZE  = 20

# ── load ──────────────────────────────────────────────────────────────────────

poses = []
with open(TRAJ) as f:
    for line in f:
        v = list(map(float, line.split()))
        if len(v) == 16:
            poses.append(np.array(v).reshape(4, 4))

raw_clouds = []
for name in PLY_NAMES:
    pc  = o3d.io.read_point_cloud(os.path.join(DATA, name))
    pts = np.asarray(pc.points)[::OPTIMIZE]
    col = np.asarray(pc.colors)[::OPTIMIZE] if pc.has_colors() else None
    raw_clouds.append((pts, col))


# ── candidate family ─────────────────────────────────────────────────────────

def signed_perms():
    axes = "xyz"
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            M = np.zeros((4, 4)); M[3, 3] = 1
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            name = "[" + ",".join(("+" if s > 0 else "-") + axes[c] for c, s in zip(perm, signs)) + "]"
            yield M, name


PERMS = list(signed_perms())
I_NAME = "[+x,+y,+z]"

CANDIDATES = [("M", lambda M: M, False),
              ("inv(M)", lambda M: np.linalg.inv(M), True)]
for L, ln in PERMS:
    for R, rn in PERMS:
        if ln == I_NAME and rn == I_NAME: continue
        CANDIDATES.append((f"{ln} @ M @ {rn}",
                           (lambda L=L, R=R: lambda M: L @ M @ R)(),
                           False))
for L, ln in PERMS:
    for R, rn in PERMS:
        if ln == I_NAME and rn == I_NAME: continue
        CANDIDATES.append((f"{ln} @ inv(M) @ {rn}",
                           (lambda L=L, R=R: lambda M: L @ np.linalg.inv(M) @ R)(),
                           True))

print(f"built {len(CANDIDATES)} candidates")


# ── viewer ───────────────────────────────────────────────────────────────────

vis_pcds = []
for pts, col in raw_clouds:
    vp = o3d.geometry.PointCloud()
    vp.points = o3d.utility.Vector3dVector(pts.copy())
    if col is not None: vp.colors = o3d.utility.Vector3dVector(col)
    vis_pcds.append(vp)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Coordinate-system explorer", 1400, 900)
for vp in vis_pcds: vis.add_geometry(vp)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))


# ── state ────────────────────────────────────────────────────────────────────

idx        = [0]
det_filter = [False]
inv_filter = [False]


def candidate_dets(fn):
    try:
        return [np.linalg.det(fn(M)[:3, :3]) for M in poses]
    except Exception:
        return [float("nan")] * len(poses)


def current_indices():
    out = []
    for i, (_, fn, is_inv) in enumerate(CANDIDATES):
        if inv_filter[0] and not is_inv: continue
        if det_filter[0] and any(d < 0.99 for d in candidate_dets(fn)): continue
        out.append(i)
    return out


def apply_current(verbose=True):
    name, fn, _ = CANDIDATES[idx[0]]
    dets = candidate_dets(fn)
    for k, (pts, _) in enumerate(raw_clouds):
        T = fn(poses[k])
        vis_pcds[k].points = o3d.utility.Vector3dVector((T[:3, :3] @ pts.T).T + T[:3, 3])
        vis.update_geometry(vis_pcds[k])
    if verbose:
        print(f"[{idx[0]+1:5d}/{len(CANDIDATES)}]  {name:42s}  dets = {[f'{d:+.1f}' for d in dets]}")
    return dets


def step(delta):
    inds = current_indices()
    if not inds:
        print("(no candidates pass filters)"); return False
    pos = inds.index(idx[0]) if idx[0] in inds else 0
    idx[0] = inds[(pos + delta) % len(inds)]
    apply_current()
    return False


def toggle_det():
    det_filter[0] = not det_filter[0]
    print(f"det=+1 filter: {'ON' if det_filter[0] else 'OFF'}  ({len(current_indices())} pass)")
    if idx[0] not in current_indices() and current_indices():
        idx[0] = current_indices()[0]; apply_current()
    return False


def toggle_inv():
    inv_filter[0] = not inv_filter[0]
    print(f"inv(M) filter: {'ON' if inv_filter[0] else 'OFF'}  ({len(current_indices())} pass)")
    if idx[0] not in current_indices() and current_indices():
        idx[0] = current_indices()[0]; apply_current()
    return False


def export():
    name, fn, _ = CANDIDATES[idx[0]]
    os.makedirs(OUT, exist_ok=True)
    stem = f"candidate_{idx[0]+1:05d}"
    with open(os.path.join(OUT, f"formula_{stem}.txt"), "w") as f:
        f.write(f"index: {idx[0]+1}/{len(CANDIDATES)}\nformula: {name}\n")
    with open(os.path.join(OUT, f"traj_{stem}.txt"), "w") as f:
        for M in poses:
            f.write(" ".join(f"{v:.18e}" for v in fn(M).flatten()) + "\n")
    print(f"exported '{name}' → {OUT}/formula_{stem}.txt + traj_{stem}.txt")
    return False


def screenshot_sweep():
    inds = current_indices()
    if not inds:
        print("sweep: no candidates pass filters"); return False
    screens = os.path.join(OUT, "screens")
    os.makedirs(screens, exist_ok=True)
    rows = []
    old = idx[0]
    print(f"sweep: writing {len(inds)} screenshots to {screens}")
    for n, ci in enumerate(inds, start=1):
        idx[0] = ci
        name, _, _ = CANDIDATES[ci]
        dets = apply_current(verbose=False)
        vis.reset_view_point(True)         # auto-frame the scene
        vis.poll_events(); vis.update_renderer()
        fname = f"screen_{ci+1:05d}.png"
        vis.capture_screen_image(os.path.join(screens, fname), do_render=True)
        rows.append({"file": fname, "index": ci+1, "formula": name,
                     "dets": " ".join(f"{d:+.6f}" for d in dets)})
        if n == 1 or n == len(inds) or n % 25 == 0:
            print(f"  {n:5d}/{len(inds)}  {fname}  {name}")
    with open(os.path.join(screens, "screen_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "index", "formula", "dets"])
        w.writeheader(); w.writerows(rows)
    idx[0] = old
    apply_current()
    print(f"sweep done → {screens}/screen_index.csv")
    return False


# ── bindings ─────────────────────────────────────────────────────────────────

vis.register_key_callback(ord(']'), lambda v: step(+1))
vis.register_key_callback(ord('['), lambda v: step(-1))
vis.register_key_callback(ord('G'), lambda v: step(+10))
vis.register_key_callback(ord('V'), lambda v: step(-10))
vis.register_key_callback(ord('J'), lambda v: step(+100))
vis.register_key_callback(ord('D'), lambda v: toggle_det())
vis.register_key_callback(ord('S'), lambda v: toggle_inv())
vis.register_key_callback(ord('X'), lambda v: export())
vis.register_key_callback(ord('R'), lambda v: screenshot_sweep())

print(f"""
{len(CANDIDATES)} candidates total. Suggested:  D  then  S  then  ] / [ .

  ] / [   next / previous     G / V   jump ±10     J   jump +100
  D       only det = +1       S       only with inv(M)
  X       export current      R       screenshot all filtered
""")

apply_current()
vis.run()
vis.destroy_window()

import os
import numpy as np
import open3d as o3d

DATA      = "input/Points"
TRAJ      = "input/traj.txt"
OUT       = "out_custom"
PLY_NAMES = ["image1.ply", "image2.ply", "image3.ply"]

poses = []
with open(TRAJ) as f:
    for line in f:
        v = list(map(float, line.split()))
        if len(v) == 16:
            poses.append(np.array(v).reshape(4, 4))

def load(name, M):
    pc  = o3d.io.read_point_cloud(os.path.join(DATA, name))
    pts = (M[:3, :3] @ np.asarray(pc.points).T).T + M[:3, 3]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if pc.has_colors(): out.colors = pc.colors
    return out

base_pcds  = [load(n, poses[i]) for i, n in enumerate(PLY_NAMES)]
local_pts  = [np.asarray(p.points).copy() for p in base_pcds]
centroids  = [pts.mean(axis=0) for pts in local_pts]
adj_T      = [np.eye(4) for _ in PLY_NAMES]
selected   = [0]
step_t     = [0.2]
step_r     = [5.0]

vis_pcds = []
for p in base_pcds:
    vp = o3d.geometry.PointCloud()
    vp.points = o3d.utility.Vector3dVector(np.asarray(p.points).copy())
    if p.has_colors(): vp.colors = p.colors
    vis_pcds.append(vp)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Step 2 — Editor", 1400, 900)
for vp in vis_pcds: vis.add_geometry(vp)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

def wcent(i):
    T = adj_T[i]; return T[:3,:3] @ centroids[i] + T[:3,3]

def refresh(i):
    T = adj_T[i]
    vis_pcds[i].points = o3d.utility.Vector3dVector((T[:3,:3] @ local_pts[i].T).T + T[:3,3])
    vis.update_geometry(vis_pcds[i])

def select(i):
    selected[0] = i; print(f">>> PLY {i+1}"); return False

def translate(axis, sign):
    d = np.zeros(3); d[axis] = sign * step_t[0]
    adj_T[selected[0]][:3,3] += d; refresh(selected[0]); return False

def rotate(axis, sign):
    a = np.deg2rad(sign * step_r[0]); c, s = np.cos(a), np.sin(a)
    if   axis == 0: R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis == 1: R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    else:           R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    i = selected[0]; cw = wcent(i)
    Tp = np.eye(4); Tp[:3,:3] = R; Tp[:3,3] = cw - R @ cw
    adj_T[i] = Tp @ adj_T[i]; refresh(i); return False

def flip(axis):
    F = np.eye(3); F[axis,axis] = -1.0
    i = selected[0]; cw = wcent(i)
    Tp = np.eye(4); Tp[:3,:3] = F; Tp[:3,3] = cw - F @ cw
    adj_T[i] = Tp @ adj_T[i]; refresh(i); return False

def reset_one():
    adj_T[selected[0]] = np.eye(4); refresh(selected[0]); return False

def reset_all():
    for i in range(3): adj_T[i] = np.eye(4); refresh(i); return False

def export_traj():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "traj_raw.txt"), "w") as f:
        for i in range(3):
            T = adj_T[i] @ poses[i]
            f.write(" ".join(f"{v:.18e}" for v in T.flatten()) + "\n")
    print(f"  saved {OUT}/traj_raw.txt"); return False

vis.register_key_callback(ord('1'), lambda v: select(0))
vis.register_key_callback(ord('2'), lambda v: select(1))
vis.register_key_callback(ord('3'), lambda v: select(2))
vis.register_key_callback(ord('A'), lambda v: translate(0, -1))
vis.register_key_callback(ord('D'), lambda v: translate(0, +1))
vis.register_key_callback(ord('Q'), lambda v: translate(1, -1))
vis.register_key_callback(ord('E'), lambda v: translate(1, +1))
vis.register_key_callback(ord('S'), lambda v: translate(2, -1))
vis.register_key_callback(ord('W'), lambda v: translate(2, +1))
vis.register_key_callback(ord('K'), lambda v: rotate(0, -1))
vis.register_key_callback(ord('I'), lambda v: rotate(0, +1))
vis.register_key_callback(ord('J'), lambda v: rotate(1, -1))
vis.register_key_callback(ord('L'), lambda v: rotate(1, +1))
vis.register_key_callback(ord('O'), lambda v: rotate(2, -1))
vis.register_key_callback(ord('U'), lambda v: rotate(2, +1))
vis.register_key_callback(ord('F'), lambda v: flip(0))
vis.register_key_callback(ord('G'), lambda v: flip(1))
vis.register_key_callback(ord('H'), lambda v: flip(2))
vis.register_key_callback(ord('R'), lambda v: reset_one())
vis.register_key_callback(ord('T'), lambda v: reset_all())
vis.register_key_callback(ord('X'), lambda v: export_traj())

print("1/2/3 select   WASDQE translate   IKJLUO rotate   FGH flip   R/T reset   X export")
vis.run()
vis.destroy_window()

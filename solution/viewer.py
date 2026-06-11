import os
import numpy as np
import open3d as o3d

DATA      = "input/Points"
TRAJ      = "input/traj.txt"
OUT       = "output"
PLY_NAMES = ["image1.ply", "image2.ply", "image3.ply"]
OPTIMIZE  = 20  # use every Nth point for speed

poses = []
with open(TRAJ) as f:
    for line in f:
        v = list(map(float, line.split()))
        if len(v) == 16:
            poses.append(np.array(v).reshape(4, 4))

def load(name, M):
    pc  = o3d.io.read_point_cloud(os.path.join(DATA, name))
    pts = np.asarray(pc.points)[::OPTIMIZE]
    col = np.asarray(pc.colors)[::OPTIMIZE] if pc.has_colors() else None
    pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if col is not None:
        out.colors = o3d.utility.Vector3dVector(col)
    return out

base_pcds = [load(n, poses[i]) for i, n in enumerate(PLY_NAMES)]

local_pts  = [np.asarray(p.points).copy() for p in base_pcds]
centroids  = [pts.mean(axis=0) for pts in local_pts]
adj_T      = [np.eye(4) for _ in PLY_NAMES]
selected   = [0]
step_t     = [0.2]
step_r     = [5.0]

vis_pcds = []
for i, p in enumerate(base_pcds):
    vp = o3d.geometry.PointCloud()
    vp.points = o3d.utility.Vector3dVector(local_pts[i].copy())
    if p.has_colors(): vp.colors = p.colors
    vis_pcds.append(vp)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Step 2 — Editor", 1400, 900)
for vp in vis_pcds: vis.add_geometry(vp)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

marker      = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
marker_base = np.asarray(marker.vertices).copy()
vis.add_geometry(marker)

def world_centroid(i):
    T = adj_T[i]
    return T[:3, :3] @ centroids[i] + T[:3, 3]

def refresh(i):
    T   = adj_T[i]
    pts = (T[:3, :3] @ local_pts[i].T).T + T[:3, 3]
    vis_pcds[i].points = o3d.utility.Vector3dVector(pts)
    vis.update_geometry(vis_pcds[i])

def refresh_marker():
    cw = world_centroid(selected[0])
    marker.vertices = o3d.utility.Vector3dVector(marker_base + cw)
    vis.update_geometry(marker)

def select(i):
    selected[0] = i; refresh_marker()
    print(f">>> PLY {i+1}  centroid {world_centroid(i)}")
    return False

def translate(axis, sign):
    d = np.zeros(3); d[axis] = sign * step_t[0]
    adj_T[selected[0]][:3, 3] += d
    refresh(selected[0]); refresh_marker()
    print(f"  PLY {selected[0]+1}: T{'XYZ'[axis]} {sign*step_t[0]:+.2f}")
    return False

def rotate(axis, sign):
    a = np.deg2rad(sign * step_r[0]); c, s = np.cos(a), np.sin(a)
    if   axis == 0: R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis == 1: R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    else:           R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    i  = selected[0]; cw = world_centroid(i)
    Tp = np.eye(4); Tp[:3,:3] = R; Tp[:3,3] = cw - R @ cw
    adj_T[i] = Tp @ adj_T[i]
    refresh(i); refresh_marker()
    print(f"  PLY {i+1}: R{'XYZ'[axis]} {sign*step_r[0]:+.1f}°")
    return False

def flip(axis):
    F = np.eye(3); F[axis, axis] = -1.0
    i  = selected[0]; cw = world_centroid(i)
    Tp = np.eye(4); Tp[:3,:3] = F; Tp[:3,3] = cw - F @ cw
    adj_T[i] = Tp @ adj_T[i]
    refresh(i); refresh_marker()
    print(f"  PLY {i+1}: flip {'XYZ'[axis]}")
    return False

def reset_one():
    adj_T[selected[0]] = np.eye(4); refresh(selected[0]); refresh_marker()
    print(f"  reset PLY {selected[0]+1}"); return False

def reset_all():
    for i in range(3): adj_T[i] = np.eye(4); refresh(i)
    refresh_marker(); print("  reset all"); return False

def step_up():
    step_t[0] *= 2; step_r[0] *= 2
    print(f"  step t={step_t[0]:.3f}  r={step_r[0]:.1f}°"); return False

def step_down():
    step_t[0] /= 2; step_r[0] /= 2
    print(f"  step t={step_t[0]:.3f}  r={step_r[0]:.1f}°"); return False

def print_state():
    np.set_printoptions(precision=4, suppress=True)
    for i in range(3):
        print(f"\n  PLY {i+1} adj_T:\n{adj_T[i]}")
    return False

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
vis.register_key_callback(ord('Z'), lambda v: step_down())
vis.register_key_callback(ord('C'), lambda v: step_up())
vis.register_key_callback(ord('P'), lambda v: print_state())
vis.register_key_callback(ord('X'), lambda v: export_traj())

print("""
  1/2/3  select    W A S D  translate Z/X    Q/E  translate Y
  I/K    pitch     J/L      yaw               U/O  roll
  F/G/H  mirror    R  reset one   T  reset all
  Z/C    step      P  print       X  export
""")

vis.run()
vis.destroy_window()

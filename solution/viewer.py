import os
import numpy as np
import open3d as o3d

DATA      = "input/Points"
TRAJ      = "input/traj.txt"
PLY_NAMES = ["image1.ply", "image2.ply", "image3.ply"]

poses = []
with open(TRAJ) as f:
    for line in f:
        v = list(map(float, line.split()))
        if len(v) == 16:
            poses.append(np.array(v).reshape(4, 4))

geoms = []
for i, name in enumerate(PLY_NAMES):
    pc  = o3d.io.read_point_cloud(os.path.join(DATA, name))
    pts = np.asarray(pc.points)
    pts = (poses[i][:3, :3] @ pts.T).T + poses[i][:3, 3]
    pc.points = o3d.utility.Vector3dVector(pts)
    geoms.append(pc)

geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
o3d.visualization.draw_geometries(geoms, window_name="Step 1 — Preview")

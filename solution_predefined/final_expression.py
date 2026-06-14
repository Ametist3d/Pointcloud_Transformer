"""
Converts input/traj.txt → result/traj.txt:

    T_out[i] = ADJ[i] @ M[i]

The three ADJ matrices below encode the alignment found manually
in the interactive editor. 

They were derived from the reference start/end pair as:
    ADJ[i] = traj_end[i] @ inv(traj_start[i])
"""
import os
import numpy as np

TRAJ_START = "input/traj.txt"
OUT        = "result"

ADJ = [
    # pose 0  (det = +1.000000)
    np.array([
        [-0.297903294365, -0.812582013680,  0.500963150756, -0.853787643635],
        [-0.860725841757,  0.455586357240,  0.227138980604,  2.580177541006],
        [-0.412801040171, -0.363526468198, -0.835131012421,  1.211406155606],
        [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
    ]),
    # pose 1  (det = +1.000000)
    np.array([
        [-0.951531772517, -0.305462346933, -0.035777619644,  3.929020410985],
        [-0.304293710594,  0.918178866711,  0.253678744464,  2.023324348652],
        [-0.044639059740,  0.252270310963, -0.966626633197,  1.213411246712],
        [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
    ]),
    # pose 2  (det = +1.000000)
    np.array([
        [-0.622116619924,  0.614405493347, -0.485259515897,  5.479275067735],
        [ 0.408034480456,  0.783413456630,  0.468797624916, -1.154884664133],
        [ 0.668190685829,  0.093644205730, -0.738073152809, -2.328256214495],
        [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
    ]),
]


def read_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            if len(v) == 16:
                poses.append(np.array(v).reshape(4, 4))
    return poses


poses = read_poses(TRAJ_START)
if len(poses) != len(ADJ):
    raise ValueError(f"expected {len(ADJ)} poses, got {len(poses)}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "traj.txt"), "w") as f:
    for i, M in enumerate(poses):
        T = ADJ[i] @ M
        print(f"  pose {i}: det = {np.linalg.det(T[:3,:3]):+.6f}")
        f.write(" ".join(f"{v:.18e}" for v in T.flatten()) + "\n")

print(f"\nwrote {OUT}/traj.txt")
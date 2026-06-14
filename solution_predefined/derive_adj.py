"""
Derives the per-pose adjustment matrices that convert traj_start into traj_end:

    ADJ[i] = traj_end[i] @ inv(traj_start[i])
    so that:  traj_end[i] = ADJ[i] @ traj_start[i]

Run this after finding the correct cloud positions in the editor (step 2).
It prints the ADJ matrices ready to paste into build_submission.py, and
verifies they reproduce traj_end exactly.

Usage:
    python derive_adj.py traj_start.txt traj_end.txt
"""
import sys
import numpy as np

START = sys.argv[1] if len(sys.argv) > 1 else "traj_start.txt"
END   = sys.argv[2] if len(sys.argv) > 2 else "traj_end.txt"


def read_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            if len(v) == 16:
                poses.append(np.array(v).reshape(4, 4))
    return poses


S = read_poses(START)
E = read_poses(END)
assert len(S) == len(E), f"pose count mismatch: {len(S)} vs {len(E)}"

print("# Paste into build_submission.py:")
print("ADJ = [")
for i in range(len(S)):
    adj = E[i] @ np.linalg.inv(S[i])
    det = np.linalg.det(adj[:3, :3])
    print(f"    # pose {i}  (det = {det:+.6f})")
    print("    np.array([")
    for row in adj:
        print("        [" + ", ".join(f"{v: .10f}" for v in row) + "],")
    print("    ]),")
print("]")

# verify
print("\n# Verification (ADJ[i] @ start[i] vs end[i]):")
for i in range(len(S)):
    adj  = E[i] @ np.linalg.inv(S[i])
    diff = np.abs(adj @ S[i] - E[i]).max()
    print(f"#   pose {i}: max diff = {diff:.2e}")
    
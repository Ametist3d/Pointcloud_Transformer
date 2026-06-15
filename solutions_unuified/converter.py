"""
Applies candidate from the explorer (formula: L @ M @ R) and bakes the
full per-pose transform into the PLY files, leaving traj.txt as identity
matrices.

Why bake into PLY:
  In Open3D the points and matrix are in the same convention, so applying
  T_out = L @ M @ R to the raw PLY points renders correctly.
  Unity's PLY importer may apply its own axis flip on load (PLY is right-
  handed, Unity is left-handed). If we left the matrix correction and PLYs
  raw, that hidden flip can fight our matrix and collapse the clouds.

  By writing each PLY in its final world-space coordinates and using identity
  matrices in traj.txt, the relative alignment between the three clouds is
  fixed inside the PLY data itself — Unity's matrix-and-loader conventions
  can rotate/flip the whole scene but cannot disturb the alignment.

Formula:
    L = [[ 1, 0, 0],   world axis remap (x, y, z) → (+x, -z, -y)
         [ 0, 0,-1],
         [ 0,-1, 0]]
    R = [[-1, 0, 0],   local axis remap (x, y, z) → (-x, -z, +y)
         [ 0, 0,-1],
         [ 0, 1, 0]]
    T_i = L @ M_i @ R     applied to every point of PLY i

"""
import os
import numpy as np

DATA      = "input/Points"
TRAJ      = "input/traj.txt"
OUT       = "output"
PLY_NAMES = ["image1.ply", "image2.ply", "image3.ply"]

L = np.array([
    [ 1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
], dtype=float)   # candidate L  ×  Rz(180°)  ×  Ry(180°) → upright AND facing camera

R = np.array([
    [-1,  0,  0,  0],
    [ 0,  0, -1,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
], dtype=float)


# ── helpers for streaming PLY conversion ─────────────────────────────────────

def transform_ply(src_path, dst_path, T):
    """Read PLY (ascii or binary), apply T to xyz of every vertex, write to dst."""
    R3 = T[:3, :3]
    t  = T[:3,  3]

    with open(src_path, "rb") as f:
        # header is always ascii
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        # parse header
        fmt = None
        vcount = 0
        props = []   # list of (type_str, name)
        elem  = None
        for hl in header_lines:
            s = hl.decode("ascii", errors="ignore").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                parts = s.split()
                elem  = parts[1]
                if elem == "vertex":
                    vcount = int(parts[2])
            elif s.startswith("property") and elem == "vertex":
                parts = s.split()
                # property <type> <name>  (no lists in vertex usually)
                props.append((parts[1], parts[-1]))

        names = [n for _, n in props]
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        has_n = all(n in names for n in ("nx", "ny", "nz"))
        if has_n:
            inx, iny, inz = names.index("nx"), names.index("ny"), names.index("nz")

        # build numpy dtype for binary, or read ascii line-by-line
        if fmt == "ascii":
            with open(dst_path, "wb") as fout:
                for hl in header_lines:
                    fout.write(hl)
                for _ in range(vcount):
                    line = f.readline().decode("ascii").split()
                    x, y, z = float(line[ix]), float(line[iy]), float(line[iz])
                    p = R3 @ np.array([x, y, z]) + t
                    line[ix], line[iy], line[iz] = f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"
                    if has_n:
                        nx, ny, nz = float(line[inx]), float(line[iny]), float(line[inz])
                        n = R3 @ np.array([nx, ny, nz])
                        line[inx], line[iny], line[inz] = f"{n[0]:.6f}", f"{n[1]:.6f}", f"{n[2]:.6f}"
                    fout.write((" ".join(line) + "\n").encode("ascii"))
                # any non-vertex bytes left
                fout.write(f.read())
        else:
            # binary
            type_map = {
                "char":   ("i1", 1), "uchar":  ("u1", 1),
                "short":  ("i2", 2), "ushort": ("u2", 2),
                "int":    ("i4", 4), "uint":   ("u4", 4),
                "float":  ("f4", 4), "double": ("f8", 8),
            }
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype  = np.dtype([(n, endian + type_map[t][0]) for t, n in props])

            verts = np.frombuffer(f.read(dtype.itemsize * vcount), dtype=dtype).copy()
            xyz   = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
            xyz   = xyz @ R3.T + t
            verts["x"], verts["y"], verts["z"] = xyz[:, 0].astype(verts["x"].dtype), \
                                                  xyz[:, 1].astype(verts["y"].dtype), \
                                                  xyz[:, 2].astype(verts["z"].dtype)
            if has_n:
                nrm = np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=1).astype(np.float64)
                nrm = nrm @ R3.T
                verts["nx"], verts["ny"], verts["nz"] = nrm[:, 0].astype(verts["nx"].dtype), \
                                                        nrm[:, 1].astype(verts["ny"].dtype), \
                                                        nrm[:, 2].astype(verts["nz"].dtype)
            tail = f.read()
            with open(dst_path, "wb") as fout:
                for hl in header_lines:
                    fout.write(hl)
                fout.write(verts.tobytes())
                fout.write(tail)


# ── main ─────────────────────────────────────────────────────────────────────

os.makedirs(OUT, exist_ok=True)

# read poses
poses = []
with open(TRAJ) as f:
    for line in f:
        v = list(map(float, line.split()))
        if len(v) == 16:
            poses.append(np.array(v).reshape(4, 4))

# write transformed PLYs
for i, name in enumerate(PLY_NAMES):
    T = L @ poses[i] @ R
    print(f"  pose {i}: det = {np.linalg.det(T[:3,:3]):+.4f}  writing {name}")
    transform_ply(os.path.join(DATA, name), os.path.join(OUT, name), T)

# write identity traj
with open(os.path.join(OUT, "traj.txt"), "w") as f:
    I = np.eye(4)
    for _ in PLY_NAMES:
        f.write(" ".join(f"{v:.18e}" for v in I.flatten()) + "\n")
print(f"\nwrote {OUT}/traj.txt  (identity matrices — all transform baked into PLYs)")

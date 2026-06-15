# Pointcloud Transformer

Solution for the computer-vision coordinate-system assignment.

The prvided viewer loads three monocular point clouds (`image1.ply`,
`image2.ply`, `image3.ply`) and a per-frame trajectory (`traj.txt`). With the
raw files the three clouds collapse on top of each other with wrong
orientations — the capture system and the viewer use different coordinate
conventions. The task is to produce replcement versions of all four files
that render correctly together in the provided viewer.

---

## Prepare
put ply files in input/Points/
put traj.txt to input/

---

## Quick start

```bash
pip install -r requirements.txt
python solutions_unuified/converter.py
```

The resulting folder contains four files that load correctly in the
provided Unity viewer.

Tested with Python 3.10.11, NumPy 2.x, Open3D 0.19.

---

## Approach

I attemptd the problem twice, and the difference between the two attempts
is worth describing because the second one is much closer to what the
assignment actually rewards.

### First attempt — manual alignment with per-pose constants

[`solution_predefined/`](https://github.com/Ametist3d/Pointcloud_Transformer/tree/main/solution_predefined)

The first approach was bottom-up: build an interactive Open3D editor that
loads the clouds the way the Unity viewer does, then drag and rotate each
cloud individually until the room geometry looks right. The editor's keys
translate/rotate/mirror the selected cloud and export the resulting
trajectory.

This worked in the sense that the exprted scene rendered correctly. The
"formula" it produced, however, was three independent per-pose adjustment
matrices (`ADJ[i] = traj_end[i] @ inv(traj_start[i])`) applied as
`T_out[i] = ADJ[i] @ M[i]`. That is **36 hand-tuned numbers** baked into the
script — not a description of the coordinate-system mismatch, just a record
of where I happened to drag the clouds. It would not work on any other
capture, and it gave no insight into *why* the original data was wrong.

I kept this folder in the repo as an honest record of the first attempt and
because the interactive editor remained useful in step 2 to validate the
real solution visually.

### Second attempt — exhaustive search over coordinate conventions

[`solutions_unuified/`](https://github.com/Ametist3d/Pointcloud_Transformer/tree/main/solutions_unuified)

The mismatch is a coordinate-system convention, so the right family of
candidate fixes is finite and small. Up to handedness, axis ordering, and
direction, there are **48 signed axis permutations** of ℝ³. Combined with
the obvious world-to-camera vs camera-to-world choice, every "constant-free"
formula of the form `L @ M @ R` or `L @ inv(M) @ R` with L, R signed axis
permutations gives a search space of **9,218 candidates** — small enough to
enumerate exhaustively.

`explore_transforms.py` does exactly that. It builds the full candidate
list, lets you step through them in an Open3D window that mirrors the Unity
viewer's matrix-application behaviour, and offers two filters that
dramatically narrow the search:

- **det = +1 only.** The Unity script extracts a quaternion from each pose
  via `Matrix4x4.rotation`, which silently fails on improper rotations
  (det = -1). Any candidate that doesn't preserve det = +1 on every pose is
  unloadable in Unity regardless of how nice it looks in Open3D.
- **`inv(M)` only.** The trajectory turned out to be in world-to-camera
  form, so the right answer almost certainly contains `inv(M)`.

With both filters on, ~150 candidates remain. A single keypress dumps a
screenshot of every one to a folder with a CSV mapping back to the formula,
so the rest of the search is just visually flipping through thumbnails and
picking the one that matches the assignment's target image. Once chosen,
its formula is read off the CSV and dropped into the converter — no fitted
constants, just `L`, `R`, and the original `traj.txt`.

The chosen formula has det(L · R) = +1 on every pose (so Unity's quaternion
extraction works) and applies the same operation to all three clouds (so it
generalises beyond this specific dataset — it's a description of the
convention mismatch, not of these particular three frames).

### Note on baking into PLYs

The submission converter writes the full per-pose transform into the
**vertex data** of each PLY file and ships `traj.txt` as identity matrices.
This is defensive: if the Unity viewer's PLY importer applies its own axis
flip on load (PLY is right-handed, Unity is left-handed), that flip would
fight a corrected `traj.txt` and collapse the layout. Baking into PLYs puts
the three clouds into their correct relative world positions *before* Unity
touches anything, so any global flip the importer applies can only rotate
the whole scene as one piece — it cannot disturb the alignment.

---

## Assumptions

- The trajectory is in **world-to-camera** form. Confirmed empirically by
  the search: every working candidate contained `inv(M)`.
- Unity uses a **left-handed, Y-up** coordinate system and decomposes the
  rotation block via `Matrix4x4.rotation`, which requires det = +1.
- The mismatch is a pure coordinate-system convention, identical for all
  three frames — not a per-frame miscalibration.
- The PLY files are correct for the source convention as-is; the entire
  correction lives in the trajectory matrices (and is then folded into the
  PLY vertex data for the bake-into-PLY safety step).

---

## Tools and references

The conversion math, the Unity convention checks, and the brute-force
strategy were developed in conversation with an LLM (Claude / Anthropic).
The LLM helped:

- decode the decompiled `PhotoPosesPlacer.cs` to confirm Unity's
  matrix-reading and the det = +1 quaternion-extraction requirement
- prove that the first-attempt per-pose adjustment matrices could not be
  reduced to any unified rigid formula `G @ M @ H` (so the second attempt
  really did need a different framing, not just better fitting)
- design and iterate the screenshot-sweep filtering workflow
- track down the PLY-importer behaviour that motivated baking the
  transform into the vertex data

Background reading consulted along the way:

- Open3D point cloud I/O — <https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html>
- Unity `Matrix4x4.rotation` quaternion extraction — <https://docs.unity3d.com/ScriptReference/Matrix4x4-rotation.html>
- PLY file format spec — <https://www.cs.cmu.edu/~/afs/cs/project/anim/ply/format.html>
- OpenCV ↔ OpenGL ↔ Unity coordinate conventions; signed permutation group
  of ℝ³ (general 3D-graphics background)

---

## Reproducing the search (optional)

If you want to redo the search rather than take the formula:

```bash
python solutions_unuified/explore_transforms.py
```

Keys:

| Key | Action |
|-----|--------|
| `]` / `[` | next / previous candidate |
| `G` / `V` | jump ±10 |
| `J` | jump +100 |
| `D` | toggle "only det = +1 outputs" filter |
| `S` | toggle "only forms with inv(M)" filter |
| `X` | export current candidate's formula + trajectory |
| `R` | screenshot every currently-filtered candidate to `out_explore/screens/` |

Recommended order: `D`, then `S`, then `R` to dump all viable candidates to
disk. Open the folder, find the screenshot that matches the assignment's
target image, look up its row in `out_explore/screens/screen_index.csv` to
read off the formula.

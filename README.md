# Training the registry on real YCB-Video data

## What's proven vs. what's new

**Reused, unchanged, already validated tonight:** `superquadric.py`
(fitting), `registry.py` (Mode, spawn/merge, Welford variance,
save/load). These scripts call them exactly as `test_registry_session.py`
and `train_val_harness.py` did on synthetic data.

**New, built against YCB-Video's documented format, NOT yet run against
real files** (no dataset download access in the sandbox this was built
in): `ycb_dataset_loader.py`, `ycb_classes.py`, the two training/eval
scripts. **Run `inspect_one_frame()` first** (step 3 below) before
trusting the rest — it prints every field name/shape it finds, so a
format mismatch surfaces immediately instead of silently producing
garbage point clouds.

---

## 1. Download YCB-Video

Official source (PoseCNN / Yu Xiang, UW Robotics):
https://rse-lab.cs.washington.edu/projects/posecnn/

Toolbox + download links: https://github.com/yuxng/YCB_Video_toolbox

**Warning: the full dataset is large (~265GB with synthetic images
included).** You only need the *real* `data/` folder for this — skip
`data_syn/` (synthetic renders) entirely, that alone cuts it drastically.

If 265GB is still too much, use **YCB-M** instead (smaller, multi-camera,
includes explicit occlusion annotations — arguably a better fit given
tonight's multi-object-occlusion finding):
https://zenodo.org/record/2579173

Either way, after extracting you should have:
```
YCB_Video_Dataset/
  data/
    0000/
      000001-color.png
      000001-depth.png
      000001-label.png
      000001-meta.mat
      ...
    0001/
    ...
    0091/
  image_sets/
    train.txt
    val.txt          <- name may differ slightly by mirror; check what's there
```

## 2. Copy the code

Copy the `ycbv_training/` folder (from this handoff) next to your existing
`superquadric.py` / `registry.py` — same layout as tonight's `pnsg/src/`.

## 3. Sanity-check the format FIRST (do not skip this)

```bash
cd pnsg/src
python3 -c "
from ycbv_training.ycb_dataset_loader import inspect_one_frame
inspect_one_frame('/path/to/YCB_Video_Dataset', '0000/000001')
"
```

This prints the actual field names in `meta.mat` and the actual
depth/label image shapes and value ranges. Compare against what
`ycb_dataset_loader.py` expects (`factor_depth`, `intrinsic_matrix`,
`cls_indexes`) — these are the standard documented field names, but if
your specific download differs, **fix the loader before running
anything else**, not after.

Also check `image_sets/` for the actual split filenames on your copy —
some distributions call the held-out set `val.txt`, others
`keyframe.txt`. Pass whatever's actually there via `--split`.

## 4. Train

```bash
cd pnsg/src
python3 -m ycbv_training.train_registry_on_ycbv \
    --dataset_root /path/to/YCB_Video_Dataset \
    --split train \
    --max_frames 2000 \
    --out trained_ycbv_model.json
```

Start with `--max_frames 200` first to confirm it runs end-to-end and
the per-class counts look sane, before committing to a multi-hour full
run. Each object instance is one `fit_superquadric()` call
(~0.1-1s depending on point count) — a few thousand frames with
multiple objects each will take a while; there's no GPU-training-style
epoch loop to speed this up, since each example is a single closed-form
update, not a gradient step.

The `--checkpoint_every` flag saves the registry periodically, so a
long run can be interrupted and you keep partial progress.

## 5. Evaluate on held-out data

```bash
python3 -m ycbv_training.evaluate_on_ycbv \
    --dataset_root /path/to/YCB_Video_Dataset \
    --split val \
    --model trained_ycbv_model.json \
    --max_frames 500
```

Reports overall + per-class top-1 accuracy, confusion pairs, and mean
confidence for correct vs. incorrect predictions — the real-data version
of tonight's `train_val_harness.py` metrics.

## 6. What this does NOT cover yet

- **Multi-part objects** (e.g. a mug's handle) are trained here as a
  single primitive per instance, using YCB-Video's ground-truth mask —
  the blind segmentation (`iterative_segment.py`) that was needed for
  RGB-bridge tonight isn't needed here, since masks are given. If you
  want the multi-part graph registry trained on real data too, that's
  a further extension (fit multiple primitives within one mask via
  connected components on depth discontinuities, same idea as
  `segmentation.py` but seeded by the real mask instead of blind
  fit-and-remove).
- **Grasp outcomes** aren't in YCB-Video at all (it's a pose-estimation
  dataset, not a grasping dataset) — that part of the registry stays
  populated from real robot trials only, as originally planned.
- **Occlusion-heavy frames**: `min_points=80` will silently skip
  instances with very little visible mask (heavy occlusion). Worth
  checking how many instances get dropped this way — if it's a lot,
  that itself is useful data about how hard the real deployment
  conditions are.

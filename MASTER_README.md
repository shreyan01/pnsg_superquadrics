# PNSG — Complete Current Codebase

This is everything currently live and working, end to end: shape fitting →
multi-part segmentation → graph structure → vocabulary registry →
introspective gate → grasp computation → RGB-photo bridge → real-dataset
training infrastructure.

**Honest note on history:** partway through tonight's session the sandbox
environment reset, which wiped a batch of earlier test/visualization
scripts (`validate_fitter.py`, `simulate_cup.py`, `deformation.py`,
`train_val_harness.py`, and others). Everything below is what was
recreated and built AFTER that point — it's the complete, current,
working system, not a byte-identical copy of every script written before
the reset. If you need any specific earlier test/demo script recreated,
ask and I'll rebuild that one.

## Core system (fully tested tonight, on synthetic data)

- **`superquadric.py`** — the shape model. Implicit surface equation,
  closed-form fitting (`fit_superquadric`), synthetic surface sampling,
  partial-view cropping, noise injection. Includes size/position bound
  options (`max_size_multiplier`, `min_size_multiplier`,
  `position_margin_multiplier`) added after finding a real optimizer
  degeneracy on small/sparse clusters.

- **`segmentation.py`** — one-shot "fit dominant primitive, cluster the
  leftover residual points" part segmentation. No labels given, no
  training data, closed-form only.

- **`iterative_segment.py`** — refines the one-shot split via iterative
  reassignment (E-step/M-step), with a divergence guard that reverts to
  the best-seen state if fit quality degrades for 2 consecutive rounds
  (found necessary after a real numerical blowup on one test case).

- **`graph.py`** — general N-part object structure: `PartNode`,
  `ObjectGraph`, MST-based relation edges (side/top/bottom attachment,
  coaxial). Generalizes beyond the simple 2-part case.

- **`registry.py`** — the vocabulary memory. `Mode` (single-part,
  Welford mean/variance with a shrinkage prior, spawn/merge, grasp-history
  logging) and `GraphMode` (multi-part: per-role sub-Modes + relation
  stats, with a structural-mismatch penalty so a candidate missing an
  expected part scores lower). `classify()`/`classify_graph()` do
  open-set "which learned word fits best" scoring. `save()`/`load()` —
  this IS the ".pt file" equivalent: a portable, trainable, loadable model.

- **`loop.py`** — the two-source introspective gate. χ = (μ_det − μ_obj)²,
  adaptive α_t (reused from the RO-MAN paper's update rule), PLUS a
  second, independent trigger (`MIN_ABSOLUTE_MU_OBJ`) added after a
  large-scale validation run found a 97% false-accept rate without it —
  restores the "ambiguity/low-evidence" half of RO-MAN's original
  two-trigger design that the simplified two-source reduction had
  silently dropped.

- **`compute_grasp.py`** — antipodal grasp candidates computed directly
  from fitted shape parameters (analytic-style normals via numeric
  gradient of the same implicit function used for fitting), with
  optional history-biased ranking once a mode has enough logged grasp
  outcomes.

- **`pipeline.py`** — wires it all together: raw cloud → segmentation →
  graph → multi-part registry match → gate decision → grasp or ask.

- **`object_zoo.py`** — synthetic object category definitions (mug,
  bottle, bowl, box, small_cup) used across tests.

## RGB-photo bridge (built to answer "can this run on a real photo")

- **`rgb_bridge.py`** — the honest architecture for bridging a flat RGB
  photo into the 3D pipeline: real classical-CV region proposals
  (`propose_regions_2d`), a depth ESTIMATE stub clearly flagged as not a
  real depth model (`estimate_depth_STUB`, later improved to
  `estimate_depth_with_prior` using category size priors), real pinhole
  backprojection math (`lift_region_to_pointcloud`), and real camera
  reprojection for drawing boxes back on the image
  (`reproject_bbox_to_image`).

- **`run_rgb_bridge.py`** — runs the full bridge end-to-end on an
  uploaded photo, draws labeled boxes + confidence with OpenCV.

- **`test_infer_scene.py`** — `teach_registry()`/`make_object_cloud()`
  helpers used to bootstrap a registry from synthetic examples before
  running inference.

- **`real_world_priors.py`** — replaces flat single-guess dimension
  priors with per-category RANGES grounded by inspecting real reference
  photos (image search + vision) — proportions/aspect-ratio confirmed
  visually, absolute scale still a general-knowledge estimate (honestly
  flagged as such throughout).

**Known, diagnosed limitation of this bridge:** confidence stayed near
zero on the one real photo tested, root-caused NOT to depth-calibration
quality (tested and ruled out) but to 2D instance-segmentation quality —
classical contour detection can't cleanly separate multiple touching
objects on a cluttered surface. That needs a learned segmenter (SAM /
Grounded-SAM) or a real depth sensor, not more classical-CV tuning.

## Real-dataset training infrastructure (`ycbv_training/`)

- **`ycb_classes.py`** — maps YCB-Video's 21 standard object classes to
  this project's vocabulary (mug, bowl, bottle, box, can), with
  unmapped classes explicitly excluded rather than force-fit.

- **`ycb_dataset_loader.py`** — parses YCB-Video's documented frame
  format (color/depth/label PNGs + meta.mat), does REAL pinhole
  backprojection using REAL sensor depth (unlike the RGB-bridge's
  stub). Includes `inspect_one_frame()` — run this FIRST against a real
  download to confirm field names before trusting anything else.

- **`train_registry_on_ycbv.py`** — trains the registry on real YCB-Video
  frames: one `fit_superquadric()` + `registry.confirm(F=1)` per labeled
  object instance. Checkpoints periodically.

- **`evaluate_on_ycbv.py`** — evaluates a trained/saved registry on a
  held-out split: top-1 accuracy, per-class accuracy, confusion pairs,
  confidence-when-correct vs confidence-when-wrong.

- **`README.md`** — download links (YCB-Video / YCB-M), exact commands,
  and an explicit list of what this training setup does NOT cover yet
  (multi-part training, grasp-outcome training).

**Honest flag:** the `ycbv_training/` code is written against YCB-Video's
well-documented standard format but has never been run against a real
downloaded file (no dataset download access in this sandbox). Everything
else in this bundle has been run and its behavior is described from
actual execution output tonight.

## Suggested order to explore

1. `superquadric.py` → `registry.py` → `loop.py` — the validated core
2. `segmentation.py` → `iterative_segment.py` → `graph.py` → `pipeline.py`
   — how a raw cloud becomes a grounded, structured decision
3. `rgb_bridge.py` + `run_rgb_bridge.py` — the honest attempt at 2D photos
4. `ycbv_training/` — the path to real data once you're back with hardware

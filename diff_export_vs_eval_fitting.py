"""
Decisive diff: does export_baseline_data.py's color-blind fitting path
produce different features -- and different classifier predictions --
than evaluate_on_ycbv.py's color-aware fitting path, on the SAME real
instances?

WHY THIS SCRIPT EXISTS: a controlled classifier x data-volume 2x2
(run against baseline_data/features_val_sample.npz) gave 93.24-93.51%
for the exact classifier/data that should reproduce the paper's real
89.4% headline (sage_predictions.csv, verified as the real saved
predictions: 992/1109). Every other explanation was ruled out directly:
  - no train/test leakage (zero exact-duplicate feature rows)
  - identical feature column order in both files
  - AXISYMMETRIC_WORDS = {'can'} confirmed identical across all three
    scripts (a stale docstring in evaluate_on_ycbv.py claims
    "can/bottle" -- the ACTUAL constant, checked directly, says
    otherwise; do not trust that comment)
  - re-loading the real, unmodified Registry.load() path reproduces
    93.51%, not 89.4%, ruling out classifier-training non-determinism
  - every trained_ycbv_*.json file with an ML classifier was tested
    directly against the real test set; none reproduce 89.4%

The one CONFIRMED, real code difference: export_baseline_data.py's
fit_one_instance() always calls canonicalize(params, color_features=
None, ...) -- it discards color entirely. evaluate_on_ycbv.py's
_eval_one_frame() always passes the REAL captured color_features. And
critically, the STORED classifier's own training data (ml_raw_X inside
trained_ycbv_ml_v2.json) has ZERO variance in the hue/saturation
columns for every category -- meaning the classifier was ALSO trained
color-blind. If evaluate_on_ycbv.py then feeds it real, varying color
at test time, that's a genuine train/test distribution mismatch on 2 of
13 feature dimensions -- and it would selectively hurt exactly the
categories where real color varies most across instances (bottle/mug/
bowl), which is exactly the pattern observed (my color-blind-consistent
setup scores far HIGHER on those three categories than the real,
reportedly-color-aware evaluation).

This script tests that hypothesis directly and decisively: for a sample
of REAL instances, it runs BOTH fitting paths (importing the real
functions, not reimplementing them) on the identical point cloud, and
reports (1) whether the resulting feature vectors actually differ, (2)
specifically whether they differ on the hue/saturation dimensions, and
(3) whether that difference is enough to flip the loaded classifier's
prediction -- and if so, in which direction.

Usage:
    python3 diff_export_vs_eval_fitting.py \\
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \\
        --split val_sample \\
        --model trained_ycbv_ml_v2.json \\
        --n_frames 40
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ycbv_training'))

from superquadric import fit_superquadric, is_physically_plausible
from iterative_segment import iterative_two_part_segment
from radius_profile import compute_radial_profile
from registry import canonicalize, Registry, HUE_INDEX
from pipeline import build_graph_from_segmentation
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab

AXISYMMETRIC_WORDS = {'can'}   # verified identical across export_baseline_data.py,
                                # train_registry_multiview.py, and evaluate_on_ycbv.py
                                # -- do not change without re-checking all three.
SAT_INDEX = HUE_INDEX + 1       # feature_names confirms saturation immediately follows hue


def fit_export_style(cloud, true_word, max_nfev=1500):
    """Byte-for-byte the same logic as export_baseline_data.py's
    fit_one_instance() -- COLOR-BLIND (color_features=None always)."""
    axisym = true_word in AXISYMMETRIC_WORDS
    if axisym:
        fitted, _ = fit_superquadric(cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
                                      min_size_multiplier=0.05, position_margin_multiplier=2.0,
                                      axisymmetric=True)
        if not is_physically_plausible(fitted):
            return None
        taper = compute_radial_profile(cloud, fitted)
        return canonicalize(fitted, color_features=None, taper_features=taper)
    else:
        params_a, params_b, assignment = iterative_two_part_segment(cloud, verbose=False, max_nfev=max_nfev)
        if not is_physically_plausible(params_a):
            return None
        return canonicalize(params_a, color_features=None, taper_features=None)


def fit_eval_style(cloud, true_word, color_features, max_nfev=1500):
    """Byte-for-byte the same logic as evaluate_on_ycbv.py's
    _eval_one_frame() -- COLOR-AWARE (real color_features passed
    through). Returns the dominant node's canonicalize() output, the
    same value classify_graph_ml() would score against."""
    axisym = true_word in AXISYMMETRIC_WORDS
    if axisym:
        params_a, _ = fit_superquadric(cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
                                        min_size_multiplier=0.05, position_margin_multiplier=2.0,
                                        axisymmetric=True)
        if not is_physically_plausible(params_a):
            return None
        taper_features = compute_radial_profile(cloud, params_a)
        graph = build_graph_from_segmentation(cloud, params_a, None, None,
                                               color_features=color_features, taper_features=taper_features)
    else:
        params_a, params_b, assignment = iterative_two_part_segment(cloud, verbose=False, max_nfev=max_nfev)
        if not is_physically_plausible(params_a):
            return None
        if params_b is not None and not is_physically_plausible(params_b):
            params_b, assignment = None, None
        graph = build_graph_from_segmentation(cloud, params_a, params_b, assignment,
                                               color_features=color_features)

    dominant = next((n for n in graph.nodes if n.role == 'dominant'), None)
    if dominant is None:
        return None
    return canonicalize(dominant.params, dominant.color_features, dominant.taper_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val_sample')
    ap.add_argument('--model', required=True, help='e.g. trained_ycbv_ml_v2.json')
    ap.add_argument('--n_frames', type=int, default=40,
                     help='Number of frames to sample (each frame may contain multiple '
                          'object instances). 40 frames is enough to see the pattern '
                          'clearly without waiting for a full re-fit of all 1,109 instances.')
    ap.add_argument('--max_nfev', type=int, default=1500)
    args = ap.parse_args()

    print(f"Loading classifier from {args.model}...")
    reg = Registry.load(args.model)
    if reg._ml_classifier is None:
        raise RuntimeError(f"{args.model} has no usable ML classifier -- pick a different --model.")
    print(f"  classifier trained on {reg._ml_classifier_n_examples} examples")

    frame_keys = read_split_file(args.dataset_root, args.split)[:args.n_frames]
    print(f"Sampling {len(frame_keys)} frames from split '{args.split}'...")

    n_compared = 0
    n_feature_diff = 0
    n_color_diff = 0
    n_prediction_diff = 0
    n_export_correct_eval_wrong = 0
    n_eval_correct_export_wrong = 0
    max_color_diff = 0.0

    for frame_key in frame_keys:
        try:
            objects = list(iter_frame_objects(args.dataset_root, frame_key, class_id_to_vocab))
        except Exception as e:
            print(f"  [skip {frame_key}]: {e}")
            continue

        for true_word, cloud, color_features in objects:
            f_export = fit_export_style(cloud, true_word, args.max_nfev)
            f_eval = fit_eval_style(cloud, true_word, color_features, args.max_nfev)

            if f_export is None or f_eval is None:
                continue  # one or both fits were implausible -- not the comparison this script targets

            n_compared += 1
            diff = np.abs(f_export - f_eval)
            color_diff = max(diff[HUE_INDEX], diff[SAT_INDEX])
            max_color_diff = max(max_color_diff, color_diff)

            if diff.max() > 1e-9:
                n_feature_diff += 1
            if color_diff > 1e-9:
                n_color_diff += 1

            pred_export = reg._ml_classifier.predict([f_export])[0]
            pred_eval = reg._ml_classifier.predict([f_eval])[0]

            if str(pred_export) != str(pred_eval):
                n_prediction_diff += 1
                export_correct = str(pred_export) == str(true_word)
                eval_correct = str(pred_eval) == str(true_word)
                if export_correct and not eval_correct:
                    n_export_correct_eval_wrong += 1
                elif eval_correct and not export_correct:
                    n_eval_correct_export_wrong += 1
                print(f"  DIVERGENT PREDICTION [{frame_key}, true={true_word}]: "
                      f"export-style(color-blind)->{pred_export}  "
                      f"eval-style(color-aware)->{pred_eval}  "
                      f"(hue_diff={diff[HUE_INDEX]:.2f}, sat_diff={diff[SAT_INDEX]:.4f})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Instances where both fits succeeded and were compared: {n_compared}")
    if n_compared == 0:
        print("No comparable instances -- check --dataset_root/--split are correct.")
        return
    print(f"Instances with ANY feature vector difference:  {n_feature_diff}/{n_compared} "
          f"({n_feature_diff/n_compared*100:.1f}%)")
    print(f"Instances with a hue/saturation difference:    {n_color_diff}/{n_compared} "
          f"({n_color_diff/n_compared*100:.1f}%)  (max diff seen: {max_color_diff:.3f})")
    print(f"Instances where this FLIPPED the classifier's prediction: "
          f"{n_prediction_diff}/{n_compared} ({n_prediction_diff/n_compared*100:.1f}%)")
    print(f"  ...of those, export(color-blind) was right, eval(color-aware) was wrong: "
          f"{n_export_correct_eval_wrong}")
    print(f"  ...of those, eval(color-aware) was right, export(color-blind) was wrong: "
          f"{n_eval_correct_export_wrong}")
    print()
    if n_color_diff > 0 and n_export_correct_eval_wrong > n_eval_correct_export_wrong:
        print("This SUPPORTS the hypothesis: real color features fed into a "
              "color-blind-trained classifier are net HARMFUL on this sample -- "
              "consistent with a real train/test distribution mismatch bug in "
              "export_baseline_data.py's color_features=None, not a classifier or "
              "data-volume effect.")
    elif n_color_diff == 0:
        print("No hue/saturation difference was observed on this sample -- the color "
              "hypothesis is NOT supported by this run. Re-check with a larger "
              "--n_frames, or look elsewhere for the 89.4%-vs-93.5% gap (e.g. a "
              "genuinely different --model file, or a difference this script doesn't "
              "cover).")
    else:
        print("Color differences were observed but did not clearly favor one "
              "direction on this sample -- inconclusive at this --n_frames; try a "
              "larger sample before drawing a conclusion.")


if __name__ == '__main__':
    main()
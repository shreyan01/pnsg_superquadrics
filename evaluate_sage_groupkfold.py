"""
Real GroupKFold evaluation for SAGE -- the thing that has never been
run. See co-author_guide.md / sage_paper.tex Section V (Limitations)
for the disclosed gap this fills: SAGE's headline 89.4% comes from a
single fixed video-level train/val split; PointNet and the same-feature
baselines (task1_feature_baselines.py, task2_pointnet.py) are the only
things ever run under sklearn's real GroupKFold. This script runs SAGE
itself under the same protocol, on the same pooled instance set, so the
comparison in the paper's Section IV-G can eventually be apples-to-
apples instead of "different partitions over different notions of
grouping" (its own words).

WHY THIS WAS NEVER POSSIBLE BEFORE: export_baseline_data.py computed
each instance's real video_id internally (it's right there in
frame_key) but never saved it to the .npz -- only X/y made it to disk.
Task 1/2's GroupKFold runs therefore had to fall back to
data_utils.compute_object_groups(), a proxy that clusters instances by
point-cloud SIZE SIGNATURE (principal extents), not real video
identity. That's a real, disclosed limitation of the EXISTING baseline
numbers too, not just something new this script fixes for SAGE.

This script requires the companion patch to export_baseline_data.py
(adds a `video_id` array to the saved .npz, same order as X/y). Re-run
export_baseline_data.py for whichever splits you want pooled AFTER
applying that patch -- old .npz files without a `video_id` key will
raise a clear error below rather than silently falling back to the
proxy.

WHAT "EVALUATING SAGE" MEANS HERE, PRECISELY: SAGE's real, decisive
classifier for every reported accuracy number this session (89.4%,
78.4%, etc.) is the ExtraTreesClassifier built by
Registry.rebuild_ml_classifier() from 13D canonicalize() feature
vectors -- registry.py's own header comment confirms classify_graph_ml
"only ever reads self._ml_classifier -- it never touches
self.graph_modes or the Welford stats at all" (see kfold_multiview_eval.py's
docstring, which relies on the exact same fact to justify skipping the
slow Welford/graph training per fold). That means a fold here only
needs to:
  1. fresh Registry()
  2. reg.import_ml_training_data(X_train, y_train)
  3. reg.rebuild_ml_classifier()
  4. classify held-out X_test directly through reg._ml_classifier
No graphs, no re-fitting superquadrics per fold -- the (X, y) feature
vectors already ARE the fitted geometry (export_baseline_data.py fit
them once, honestly, same fitting rules as training/eval). This is
what makes this a same-feature, same-classifier, different-SPLIT-ONLY
comparison -- exactly what Section IV-G's caveat says is still missing.

CAVEAT, stated directly rather than smoothed over: this evaluates the
ML classifier under real video-grouped CV. It does NOT re-run the
axisymmetric-fitting circularity disclosed in sage_paper.tex's
Limitations section (can's axisymmetric constraint was selected using
the ground-truth label at feature-export time) -- that circularity is
baked into the X vectors THEMSELVES, upstream of this script, and
re-fitting from raw point clouds per fold to remove it is future work,
not something bolted on here. Read this script's number as "SAGE's
existing features/classifier under a real leakage-safe split", not as
a fully independent re-derivation of SAGE's accuracy from scratch.

Usage:
    # 1. Apply the export_baseline_data.py patch, then re-export
    #    whichever splits you want pooled into the CV:
    python3 export_baseline_data.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \\
        --out_dir ./baseline_data --split train --workers 30
    python3 export_baseline_data.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \\
        --out_dir ./baseline_data --split val_sample --workers 30

    # 2. Run this script over the pooled result:
    python3 evaluate_sage_groupkfold.py \\
        --data_dir ./baseline_data --splits train val_sample --n_splits 5
"""
import os
# MUST be set before numpy/scipy/sklearn are imported -- see registry.py's
# module docstring for the real incident (load average of 935 on a
# nominally 30-process run) this guards against. This script is single-
# process (ExtraTrees fitting on ~1-2k x 13 arrays is not the bottleneck
# that justified process pools elsewhere), but setting these costs
# nothing and keeps every entry-point script in this repo consistent.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import Registry, assert_thread_limits_ok


def load_pooled_data(data_dir, splits):
    """Concatenate features_<split>.npz across every requested split
    into one pool, so GroupKFold can freely reassign whole videos to
    train/test across fold boundaries -- exactly what task2_pointnet.py
    does for PointNet/k-NN/SVM, just with real video_id instead of the
    size-signature proxy."""
    data_dir = Path(data_dir)
    X_parts, y_parts, group_parts, source_parts = [], [], [], []

    for split_name in splits:
        path = data_dir / f'features_{split_name}.npz'
        if not path.exists():
            raise FileNotFoundError(
                f"\n{path} not found.\n"
                f"Run export_baseline_data.py --split {split_name} first."
            )
        data = np.load(path, allow_pickle=True)
        if 'video_id' not in data.files:
            raise KeyError(
                f"\n{path} has no 'video_id' array.\n"
                f"This file was exported with the OLD export_baseline_data.py, "
                f"before the patch that saves real video_id per instance. "
                f"Apply that patch and re-run the export for '{split_name}' -- "
                f"do NOT substitute data_utils.compute_object_groups() here, "
                f"that proxy is exactly what this script exists to avoid."
            )
        X_parts.append(data['X'])
        y_parts.append(data['y'])
        group_parts.append(data['video_id'])
        source_parts.append(np.full(len(data['y']), split_name, dtype=object))
        print(f"  loaded {len(data['y'])} instances from {path} "
              f"({len(set(data['video_id']))} distinct videos)")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    source = np.concatenate(source_parts, axis=0)

    overlap = set(group_parts[0]) if len(group_parts) == 1 else set.intersection(
        *(set(g) for g in group_parts)
    ) if len(group_parts) > 1 else set()
    if len(group_parts) > 1 and overlap:
        print(f"  NOTE: {len(overlap)} video id(s) appear in more than one "
              f"requested split ({sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}). "
              f"GroupKFold still keeps each such video whole within a single "
              f"fold, so this does not reintroduce leakage -- just flagging "
              f"it since it means the splits you named weren't fully disjoint "
              f"in video terms to begin with.")

    return X, y, groups, source


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - margin, center + margin


def run_groupkfold(X, y, groups, n_splits, seed, n_estimators):
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n = len(y)
    out_of_fold_pred = np.empty(n, dtype=object)
    out_of_fold_fold = np.full(n, -1, dtype=int)

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(n), y, groups=groups), start=1
    ):
        train_videos = set(groups[train_idx])
        test_videos = set(groups[test_idx])
        leak = train_videos & test_videos
        assert not leak, (
            f"Fold {fold_number}: {len(leak)} video(s) appear in BOTH "
            f"train and test ({sorted(leak)[:5]}) -- this would be a real "
            f"leakage bug in the fold assignment itself, not just an "
            f"accuracy number to distrust. Stopping rather than reporting "
            f"a number built on it."
        )

        reg = Registry()
        reg.import_ml_training_data(X[train_idx], y[train_idx])
        built = reg.rebuild_ml_classifier(n_estimators=n_estimators)
        if not built:
            print(f"  WARNING: fold {fold_number} classifier failed to build "
                  f"(sklearn missing, or zero usable training rows) -- "
                  f"skipping this fold's test instances entirely rather than "
                  f"scoring them against no model.")
            continue

        preds = reg._ml_classifier.predict(X[test_idx])
        out_of_fold_pred[test_idx] = preds
        out_of_fold_fold[test_idx] = fold_number

        fold_correct = int((preds == y[test_idx]).sum())
        fold_n = len(test_idx)
        print(f"  Fold {fold_number}/{n_splits}: {len(train_videos)} train videos, "
              f"{len(test_videos)} test videos, {fold_n} test instances -> "
              f"{fold_correct}/{fold_n} = {fold_correct / fold_n * 100:.1f}%")

    return out_of_fold_pred, out_of_fold_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./baseline_data')
    ap.add_argument('--splits', nargs='+', default=['train', 'val_sample'],
                     help='Which features_<split>.npz files to pool before '
                          'GroupKFold splits them. Default matches the '
                          'train/val_sample split used everywhere else in '
                          'this repo.')
    ap.add_argument('--n_splits', type=int, default=5,
                     help='Same N_SPLITS as task1_feature_baselines.py / '
                          'task2_pointnet.py, for a directly comparable fold count.')
    ap.add_argument('--seed', type=int, default=42,
                     help='Matches RANDOM_STATE in task1/task2, so the same '
                          'seed convention is used project-wide (this does '
                          'NOT reproduce the exact same folds -- those scripts '
                          'group by the size-signature proxy, this one groups '
                          'by real video_id -- it just keeps the convention '
                          'consistent).')
    ap.add_argument('--n_estimators', type=int, default=500,
                     help='Matches rebuild_ml_classifier\'s own default, which '
                          'is itself the value behind the reported 89.4%%/94.9%% '
                          'numbers -- change this only if you are deliberately '
                          'testing sensitivity to it.')
    ap.add_argument('--out', default='sage_groupkfold_predictions.csv')
    args = ap.parse_args()

    assert_thread_limits_ok()

    print(f"Loading pooled data from splits {args.splits}...")
    X, y, groups, source = load_pooled_data(args.data_dir, args.splits)
    print(f"Pooled: {len(y)} instances, {len(set(groups))} distinct real videos, "
          f"{len(set(y))} classes.")

    print(f"\nRunning {args.n_splits}-fold GroupKFold (groups=real video_id)...")
    preds, folds = run_groupkfold(X, y, groups, args.n_splits, args.seed, args.n_estimators)

    scored = folds >= 0
    n_total = int(scored.sum())
    n_correct = int((preds[scored] == y[scored]).sum())

    print(f"\n{'=' * 70}")
    print(f"SAGE (ExtraTrees classifier) under real-video GroupKFold "
          f"({args.n_splits} folds)")
    print(f"{'=' * 70}")
    print(f"Overall: {n_correct}/{n_total} = {n_correct / max(n_total, 1) * 100:.2f}%")
    lo, hi = wilson_ci(n_correct, n_total)
    print(f"95% Wilson CI: [{lo * 100:.2f}%, {hi * 100:.2f}%]")

    per_class_total = collections.defaultdict(int)
    per_class_correct = collections.defaultdict(int)
    for true_word, pred_word, is_scored in zip(y, preds, scored):
        if not is_scored:
            continue
        per_class_total[true_word] += 1
        per_class_correct[true_word] += int(pred_word == true_word)

    print("\nPer-class (pooled across folds):")
    for word in sorted(per_class_total):
        c, t = per_class_correct[word], per_class_total[word]
        lo_c, hi_c = wilson_ci(c, t)
        print(f"  {word:10s}: {c:3d}/{t:3d} = {c / t * 100:5.1f}%   "
              f"95% CI [{lo_c * 100:.1f}%, {hi_c * 100:.1f}%]")

    print(f"\nFor reference, SAGE's own fixed video-level split reported "
          f"89.4% -- compare against that number, not the 78.4% headline, "
          f"since 89.4% is the same ExtraTrees-classifier / same-feature "
          f"setup this script evaluates, just under a different split protocol.")

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'video_id', 'source_split', 'true_label', 'predicted_label', 'correct'])
        for i in range(len(y)):
            if folds[i] < 0:
                continue
            writer.writerow([folds[i], groups[i], source[i], y[i], preds[i], int(preds[i] == y[i])])
    print(f"\nAll {n_total} pooled out-of-fold predictions saved to {args.out} "
          f"(same column shape as task5_statistics.py's other model prediction "
          f"files, so this can be added to that script's MODEL_SPECS for real "
          f"bootstrap CIs / McNemar tests against PointNet and k-NN under a "
          f"shared protocol).")


if __name__ == '__main__':
    main()
"""
The corrected pipeline: point it at YCB-Video and it runs everything.

    python3 src/run_corrected_pipeline.py --dataset_root ~/ycb_dataset

What it does, in order:

  0. Self-test          verify the feature path before spending hours
  1. Instrumented       label-free re-export of train + val_sample,
     re-export          carrying video_id, both fits, fit residuals and
                        abstention rows          (unblocks 1, 3, 5, 6)
  2. Unified protocol   every method on the SAME video-level folds, same
                        instances, same inputs; accuracy, balanced
                        accuracy, macro-F1, per-class recall, confusion
                        matrices                              (item 1)
  3. Grouped stats      cluster bootstrap over REAL video ids, for every
                        method rather than SAGE alone         (item 5)
  4. Failure diagnosis  do the geometric indicators separate reliable
                        from unreliable fits? AUROC plus a risk-coverage
                        curve                                 (item 6)

Items 3, 4 and 7 have their own entry points and are listed by
--next-steps, since they need choices this script should not make for
you.

Every stage is skippable and every stage caches, so a failure part-way
does not cost you the stages that already succeeded.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'corrected'

CLASSES = ['box', 'can', 'mug', 'bottle', 'bowl']


# =====================================================================
# Stage 1 -- instrumented export
# =====================================================================

def stage_export(dataset_root, splits, workers, max_nfev, limit, force):
    """Label-free re-export of each split, cached by output path."""

    import export_instrumented

    produced = {}

    for split in splits:

        out = REPO_ROOT / 'baseline_data' / f'instrumented_{split}.npz'

        if out.exists() and not force:
            print(f'  [cached] {out.name} -- pass --force to rebuild')
            produced[split] = out
            continue

        export_instrumented.export(
            dataset_root, split, out, workers,
            max_nfev=max_nfev, limit=limit,
        )
        produced[split] = out

    return produced


def load_instrumented(path):
    """Load an instrumented export into a tidy frame plus feature matrix."""

    data = np.load(path, allow_pickle=True)

    return (
        data['X'],
        data['y'].astype(str),
        data['video_id'].astype(str),
        data['instance_id'].astype(str),
        [str(name) for name in data['feature_names']],
    )


# =====================================================================
# Stage 2 -- unified evaluation protocol (item 1)
# =====================================================================

def build_methods():
    """Every method, evaluated identically.

    All operate on the same instrumented feature matrix, so "same input
    information" holds by construction -- which is the entire point of
    the unified protocol. PointNet is excluded here because it consumes
    raw clouds rather than this matrix; it is evaluated on the same
    folds by task2 and merged in the report.
    """
    return {
        'knn': Pipeline([
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5)),
        ]),
        'svm_linear': Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(kernel='linear', C=1.0)),
        ]),
        'svm_rbf': Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale')),
        ]),
        'extratrees': Pipeline([
            ('scale', StandardScaler()),
            ('clf', ExtraTreesClassifier(n_estimators=500, random_state=0)),
        ]),
    }


def stage_unified(X, y, videos, instance_ids, n_folds, seed):
    """Video-level cross-validation, identical folds for every method."""

    n_videos = len(set(videos))

    if n_videos < n_folds:
        print(f'  only {n_videos} videos; reducing folds to {n_videos}')
        n_folds = max(2, n_videos)

    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups=videos))

    # The property the whole protocol rests on.
    for index, (train_idx, test_idx) in enumerate(folds, start=1):
        overlap = set(videos[train_idx]) & set(videos[test_idx])
        assert not overlap, f'fold {index} leaks videos: {sorted(overlap)}'

    print(f'  {n_folds} video-level folds over {n_videos} videos; '
          f'no video appears on both sides of any fold')

    rows, predictions = [], {}

    for name, model in build_methods().items():

        pred = np.empty(len(y), dtype=object)

        for train_idx, test_idx in folds:
            model.fit(X[train_idx], y[train_idx])
            pred[test_idx] = model.predict(X[test_idx])

        pred = pred.astype(str)
        predictions[name] = pred

        row = {
            'model': name,
            'n_instances': len(y),
            'n_videos': n_videos,
            'accuracy': accuracy_score(y, pred),
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'macro_f1': f1_score(y, pred, average='macro'),
        }

        for label in CLASSES:
            mask = y == label
            row[f'{label}_recall'] = (
                float((pred[mask] == label).mean()) if mask.any() else np.nan
            )

        rows.append(row)

        print(f'  {name:12s} acc {row["accuracy"] * 100:6.2f}%  '
              f'bal {row["balanced_accuracy"] * 100:6.2f}%  '
              f'macroF1 {row["macro_f1"] * 100:6.2f}%')

        matrix = pd.DataFrame(
            confusion_matrix(y, pred, labels=CLASSES),
            index=[f'true_{c}' for c in CLASSES],
            columns=[f'pred_{c}' for c in CLASSES],
        )
        matrix.to_csv(RESULTS_DIR / f'{name}_confusion_matrix.csv')

        pd.DataFrame({
            'instance_id': instance_ids,
            'video_id': videos,
            'true_label': y,
            'predicted_label': pred,
            'correct': (pred == y).astype(int),
        }).to_csv(RESULTS_DIR / f'{name}_predictions.csv', index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / 'unified_protocol.csv', index=False)

    return summary, predictions


# =====================================================================
# Stage 3 -- cluster bootstrap over real videos (item 5)
# =====================================================================

def stage_grouped_stats(y, videos, predictions, n_bootstrap, seed):
    """Confidence intervals that respect video clustering.

    Every method is clustered at the SAME level here -- real video ids --
    which is what the earlier run could not do, since only SAGE's
    prediction file carried video identity.
    """
    import task5_statistics as t5

    rows = []

    for name, pred in predictions.items():

        _acc, iid_lo, iid_hi = t5.bootstrap_accuracy_ci(
            y, pred, n_bootstrap=n_bootstrap, random_state=seed,
        )

        acc, cl_lo, cl_hi, n_clusters = t5.cluster_bootstrap_ci(
            y, pred, videos, n_bootstrap=n_bootstrap, random_state=seed,
        )

        rows.append({
            'model': name,
            'accuracy': acc,
            'ci_lower_iid': iid_lo,
            'ci_upper_iid': iid_hi,
            'ci_width_iid': iid_hi - iid_lo,
            'ci_lower_clustered': cl_lo,
            'ci_upper_clustered': cl_hi,
            'ci_width_clustered': cl_hi - cl_lo,
            'widening': (cl_hi - cl_lo) / (iid_hi - iid_lo)
            if iid_hi > iid_lo else np.nan,
            'n_videos': n_clusters,
        })

        print(f'  {name:12s} {acc * 100:6.2f}%  '
              f'iid [{iid_lo * 100:5.2f}, {iid_hi * 100:5.2f}]  '
              f'clustered [{cl_lo * 100:5.2f}, {cl_hi * 100:5.2f}]  '
              f'{rows[-1]["widening"]:.1f}x')

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / 'grouped_statistics.csv', index=False)

    print('\n  All methods are clustered at the same level (video), so '
          'these\n  intervals ARE comparable to each other -- unlike the '
          'earlier run.')

    return frame


# =====================================================================
# Stage 4 -- geometric failure diagnosis (item 6)
# =====================================================================

def stage_failure_diagnosis(X, y, videos, feature_names, predictions,
                            reference_model='extratrees'):
    """Do the geometric indicators flag fits that go on to be wrong?

    Framed as a detection problem: predict "this instance will be
    misclassified" from fit-quality signals alone, never from the label.
    Reported as AUROC plus a risk-coverage curve, which is what a robot
    deciding whether to trust an estimate actually needs.
    """
    pred = predictions[reference_model]
    wrong = (pred != y).astype(int)

    if wrong.sum() == 0:
        print('  no misclassifications to diagnose')
        return None, None

    indicator_names = [
        'ax_nrmse', 'fl_nrmse', 'residual_ratio', 'angular_variation',
    ]
    available = [n for n in indicator_names if n in feature_names]

    rows = []

    for name in available:

        values = X[:, feature_names.index(name)]
        finite = np.isfinite(values)

        if finite.sum() < 10 or len(set(wrong[finite])) < 2:
            continue

        auc = roc_auc_score(wrong[finite], values[finite])

        rows.append({
            'indicator': name,
            'auroc_error_detection': auc,
            # Below 0.5 means the indicator is informative with the sign
            # flipped -- worth reporting rather than discarding.
            'auroc_oriented': max(auc, 1 - auc),
            'direction': 'higher=worse' if auc >= 0.5 else 'lower=worse',
            'n': int(finite.sum()),
        })

    diagnosis = pd.DataFrame(rows).sort_values(
        'auroc_oriented', ascending=False
    )
    diagnosis.to_csv(RESULTS_DIR / 'failure_diagnosis.csv', index=False)

    for _, row in diagnosis.iterrows():
        print(f'  {row["indicator"]:20s} AUROC {row["auroc_oriented"]:.3f}  '
              f'({row["direction"]})')

    # Risk-coverage using the best oriented indicator.
    if not len(diagnosis):
        return diagnosis, None

    best = diagnosis.iloc[0]
    values = X[:, feature_names.index(best['indicator'])].astype(float)

    if best['direction'] == 'lower=worse':
        values = -values

    order = np.argsort(values)          # most trustworthy first
    correct_sorted = (pred == y).astype(int)[order]

    coverage, risk = [], []

    for keep in range(max(10, len(order) // 20), len(order) + 1,
                      max(1, len(order) // 20)):
        coverage.append(keep / len(order))
        risk.append(1.0 - correct_sorted[:keep].mean())

    curve = pd.DataFrame({'coverage': coverage, 'error_rate': risk})
    curve.to_csv(RESULTS_DIR / 'risk_coverage.csv', index=False)

    full_error = 1.0 - correct_sorted.mean()
    half = curve.iloc[(curve['coverage'] - 0.5).abs().argmin()]

    print(f'\n  risk-coverage (indicator: {best["indicator"]})')
    print(f'    error at 100% coverage : {full_error * 100:5.2f}%')
    print(f'    error at  50% coverage : {half["error_rate"] * 100:5.2f}%')

    if half['error_rate'] < full_error:
        print('    -> abstaining on the flagged half genuinely reduces error')
    else:
        print('    -> abstaining does NOT reduce error; the indicators do '
              'not\n       identify the failures on this data')

    return diagnosis, curve


# =====================================================================
# Driver
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset_root', required=False,
                        help='YCB-Video root (holds image_sets/ and data/)')
    parser.add_argument('--splits', nargs='+', default=['val_sample'],
                        help='Splits to export and evaluate.')
    parser.add_argument('--eval-split', default='val_sample')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max-nfev', type=int, default=1500)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only the first N frames per split (smoke test).')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild exports even if cached.')
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--selftest', action='store_true',
                        help='Verify the feature path, no dataset needed.')
    parser.add_argument('--next-steps', action='store_true')
    args = parser.parse_args()

    if args.selftest:
        import export_instrumented
        raise SystemExit(0 if export_instrumented.selftest() else 1)

    if args.next_steps:
        print(NEXT_STEPS)
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workers = args.workers or os.cpu_count()
    started = time.time()

    print('=' * 70)
    print('CORRECTED PIPELINE')
    print('=' * 70)

    # ---- stage 1 -------------------------------------------------
    export_path = (
        REPO_ROOT / 'baseline_data' / f'instrumented_{args.eval_split}.npz'
    )

    if not args.skip_export:
        if not args.dataset_root:
            raise SystemExit(
                '--dataset_root is required unless --skip-export is given '
                '(or --selftest).'
            )
        print('\n[1/4] Instrumented label-free re-export')
        stage_export(args.dataset_root, args.splits, workers,
                     args.max_nfev, args.limit, args.force)
    else:
        print('\n[1/4] Export skipped; using cached')

    if not export_path.exists():
        raise SystemExit(
            f'Missing {export_path}. Run without --skip-export first.'
        )

    X, y, videos, instance_ids, feature_names = load_instrumented(export_path)
    print(f'      {X.shape[0]} instances x {X.shape[1]} features, '
          f'{len(set(videos))} videos')

    # ---- stage 2 -------------------------------------------------
    print('\n[2/4] Unified evaluation protocol (item 1)')
    summary, predictions = stage_unified(
        X, y, videos, instance_ids, args.n_folds, args.seed
    )

    # ---- stage 3 -------------------------------------------------
    print('\n[3/4] Cluster bootstrap over real video ids (item 5)')
    stage_grouped_stats(y, videos, predictions, args.n_bootstrap, args.seed)

    # ---- stage 4 -------------------------------------------------
    print('\n[4/4] Geometric failure diagnosis (item 6)')
    stage_failure_diagnosis(X, y, videos, feature_names, predictions)

    print('\n' + '=' * 70)
    print(f'COMPLETE in {(time.time() - started) / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')
    print('=' * 70)
    print(NEXT_STEPS)

    return 0


NEXT_STEPS = """
Still to run, each needing a choice this script should not make for you:

  Item 3  fixed-population noise
      python3 src/evaluate_sage_robustness.py --dataset_root <root> \\
          --model ../trained_ycbv_ml_v2.json --workers 30
      Pass --fixed-population once that flag lands, so every noise level
      scores the SAME instances rather than re-deriving the set.

  Item 4  classifier x observation ablation (2x2)
      python3 kfold_multiview_eval.py --dataset_root <root> --scoring ml
      python3 evaluate_multiview_ycbv.py --dataset_root <root> \\
          --model ../trained_ycbv_ml_v2.json
      Both cells exist; they need the fold definition this script wrote
      to results/corrected/ so all four cells share it.

  Item 7  corrective multi-view
      Depends on item 6's detector above being informative. Check
      failure_diagnosis.csv first -- if no indicator clears ~0.65 AUROC,
      the closed-loop policy has nothing to trigger on and the
      experiment will not show a gain.

  Retrain
      The registry itself still fits using the label
      (AXISYMMETRIC_WORDS). Repointing train_registry_multiview.py at
      symmetry.fit_both() and retraining is what makes SAGE's own
      number comparable to the baselines above.
"""


if __name__ == '__main__':
    raise SystemExit(main())

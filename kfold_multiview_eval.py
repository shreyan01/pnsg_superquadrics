"""
Video-level K-FOLD cross-validation for the multi-view ML classifier.

The problem this solves: evaluate_multiview_ycbv.py on the fixed val
split only has 13 held-out videos to work with (92 total videos, 79
already spent on train.txt) -- giving n=30 real object instances and a
95% CI of [78.7%, 98.2%], too wide to say anything with confidence.

This script instead ROTATES which videos are held out across N folds.
Every one of the 92 videos gets used as held-out test data in exactly
one fold, while a classifier trained on that fold's OTHER videos scores
it -- no video is ever scored by a classifier that saw it during
training. Pooling predictions across all folds gives you a real n close
to the full 92-video ceiling instead of the fixed split's 13.

IMPORTANT -- this only cross-validates the ML classifier (the part we've
been improving this session), NOT the Welford registry/graph_modes
structure. That's a deliberate simplification, not an oversight:
classify_graph_ml() only ever reads self._ml_classifier -- it never
touches self.graph_modes or the Welford stats at all (check the
registry.py source if you want to confirm). So there's no need to run
the slow multiview Welford training (train_registry_multiview.py) per
fold; each fold just needs a fresh ExtraTreesClassifier trained on that
fold's per-frame exported features, via the same
import_ml_training_data() + rebuild_ml_classifier() path
bake_ml_classifier.py already uses.

COST CONTROL: a full stride=1 per-frame export for each fold's ~73-74
training videos would cost roughly what the original full training
export cost (76 min), FIVE TIMES OVER. Given we already found that
capping box from 16619->2200 examples barely moved accuracy at all
(89.4%->89.0%), there's no evidence that needs the full frame density
either. --train_frame_stride defaults to 5 (1/5 the frames) to keep
total runtime to roughly one fold's worth of the original cost, not
five. Set it to 1 if you have hours to spare and want to rule out any
possibility that stride is costing you something.

Usage:
    python3 kfold_multiview_eval.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --n_folds 5 --scoring ml --workers 30
"""
import argparse
import collections
import csv
import os

# MUST be set before numpy/scipy are imported -- see registry.py's
# module docstring for the real incident this fix is for (a load
# average of 935 on a nominally 30-process run). This line was missing
# here specifically -- registry.py now sets this too, but don't rely on
# import order alone across files; setting it explicitly, this early,
# in every entry-point script is the robust version of this fix.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from registry import Registry
from export_baseline_data import _process_one_frame, _init_worker as _init_export_worker
from ycbv_training.train_registry_multiview import (
    discover_video_ids, discover_video_classes, _aggregate_and_fit, _init_worker as _init_train_worker,
)
from ycbv_training.ycb_pose_aggregation import discover_frames

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def build_folds(dataset_root, n_folds, seed):
    all_videos = sorted(discover_video_ids(dataset_root))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(all_videos))
    folds = [shuffled[i::n_folds] for i in range(n_folds)]   # round-robin, keeps fold sizes balanced
    return all_videos, [sorted(f) for f in folds]


def frame_keys_for_videos(dataset_root, video_ids, stride):
    keys = []
    for vid in video_ids:
        frames = discover_frames(dataset_root, vid)[::stride]
        keys.extend(f'{vid}/{fr}' for fr in frames)
    return keys


def export_features_for_frames(dataset_root, frame_keys, workers, max_nfev, desc):
    work_items = [(dataset_root, fk, False, max_nfev) for fk in frame_keys]
    X, y = [], []
    pbar = tqdm(total=len(work_items), desc=desc, unit='frame') if HAVE_TQDM else None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_export_worker) as ex:
        for frame_key, results, error in ex.map(_process_one_frame, work_items):
            if pbar: pbar.update(1)
            if error:
                continue
            for f, true_word, _cloud in results:
                X.append(f); y.append(true_word)
    if pbar: pbar.close()
    return np.array(X), np.array(y)


def evaluate_videos(dataset_root, video_ids, reg, scoring, frame_stride, max_nfev, workers, desc):
    work_items = []
    for vid in video_ids:
        for cid in discover_video_classes(dataset_root, vid):
            work_items.append((dataset_root, vid, cid, frame_stride, max_nfev))
    rows = []
    pbar = tqdm(total=len(work_items), desc=desc, unit='pair') if HAVE_TQDM else None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_train_worker) as ex:
        for video_id, class_id, vocab_word, graph, error in ex.map(_aggregate_and_fit, work_items):
            if pbar: pbar.update(1)
            if error or graph is None:
                continue
            if scoring == 'ml':
                ranked = reg.classify_graph_ml(graph, top_k=1)
            elif scoring == 'ensembled':
                ranked = reg.classify_graph_ensembled(graph, top_k=1)
            else:
                ranked = reg.classify_graph(graph, top_k=1)
            pred_word = ranked[0][0] if ranked else '__UNKNOWN__'
            confidence = float(ranked[0][1]) if ranked else 0.0
            rows.append({'video': video_id, 'true_label': vocab_word, 'predicted_label': pred_word,
                         'confidence': confidence, 'correct': int(pred_word == vocab_word)})
    if pbar: pbar.close()
    return rows


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - margin, center + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scoring', choices=['joint', 'ensembled', 'ml'], default='ml')
    ap.add_argument('--train_frame_stride', type=int, default=5,
                     help='Stride for per-frame TRAINING export within each fold (default 5, see '
                          'cost note in module docstring). 1 = full density, matches the original '
                          'export exactly, ~5x slower.')
    ap.add_argument('--eval_frame_stride', type=int, default=10,
                     help='Stride for multi-view aggregation of held-out instances (matches '
                          'train_registry_multiview.py default).')
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--out', default='kfold_multiview_predictions.csv')
    args = ap.parse_args()

    from registry import assert_thread_limits_ok
    assert_thread_limits_ok()

    workers = args.workers or os.cpu_count()

    all_videos, folds = build_folds(args.dataset_root, args.n_folds, args.seed)
    print(f'{len(all_videos)} total videos, split into {args.n_folds} folds: '
          f'{[len(f) for f in folds]}')

    all_rows = []
    t_start = time.time()

    for k in range(args.n_folds):
        test_videos = folds[k]
        train_videos = [v for i, f in enumerate(folds) if i != k for v in f]

        print(f'\n{"="*70}\nFOLD {k+1}/{args.n_folds}: {len(train_videos)} train videos, '
              f'{len(test_videos)} test videos\n{"="*70}')

        train_frame_keys = frame_keys_for_videos(args.dataset_root, train_videos, args.train_frame_stride)
        print(f'Exporting training features from {len(train_frame_keys)} frames '
              f'(stride={args.train_frame_stride})...')

        X, y = export_features_for_frames(args.dataset_root, train_frame_keys, workers, args.max_nfev,
                                            desc=f'Fold {k+1} export')
        if len(X) == 0:
            print(f'  WARNING: fold {k+1} got zero training examples, skipping.')
            continue
        print(f'  {len(X)} training examples: {dict(collections.Counter(y))}')

        reg = Registry()
        reg.import_ml_training_data(X, y)
        built = reg.rebuild_ml_classifier()
        if not built:
            print(f'  WARNING: fold {k+1} classifier failed to build (sklearn missing?), skipping.')
            continue

        fold_rows = evaluate_videos(args.dataset_root, test_videos, reg, args.scoring,
                                     args.eval_frame_stride, args.max_nfev, workers,
                                     desc=f'Fold {k+1} eval')
        for r in fold_rows:
            r['fold'] = k
        all_rows.extend(fold_rows)

        fold_correct = sum(r['correct'] for r in fold_rows)
        fold_n = len(fold_rows)
        print(f'  Fold {k+1} result: {fold_correct}/{fold_n} = '
              f'{fold_correct/max(fold_n,1)*100:.1f}%')

    # ---- Pooled results across all folds ----
    print(f'\n{"="*70}\nPOOLED RESULT ACROSS ALL {args.n_folds} FOLDS '
          f'({time.time()-t_start:.0f}s total)\n{"="*70}')

    n_total = len(all_rows)
    n_correct = sum(r['correct'] for r in all_rows)
    print(f'Overall: {n_correct}/{n_total} = {n_correct/max(n_total,1)*100:.1f}%')
    lo, hi = wilson_ci(n_correct, n_total)
    print(f'95% Wilson CI: [{lo*100:.1f}%, {hi*100:.1f}%]')

    per_class_total = collections.defaultdict(int)
    per_class_correct = collections.defaultdict(int)
    for r in all_rows:
        per_class_total[r['true_label']] += 1
        per_class_correct[r['true_label']] += r['correct']

    print('\nPer-class (pooled across folds):')
    for word in sorted(per_class_total):
        c, t = per_class_correct[word], per_class_total[word]
        lo_c, hi_c = wilson_ci(c, t)
        print(f'  {word:10s}: {c:3d}/{t:3d} = {c/t*100:5.1f}%   '
              f'95% CI [{lo_c*100:.1f}%, {hi_c*100:.1f}%]')

    confusion = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in all_rows:
        confusion[r['true_label']][r['predicted_label']] += 1
    print('\nConfusion (true -> predicted), pooled:')
    for true_word in sorted(confusion):
        for pred_word, count in sorted(confusion[true_word].items(), key=lambda kv: -kv[1]):
            if pred_word != true_word:
                print(f'  {true_word} -> {pred_word}: {count} times')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'video', 'true_label', 'predicted_label',
                                                 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nAll {n_total} pooled per-instance predictions saved to {args.out}')
    print('This n is a real, valid held-out number -- every prediction came from a classifier '
          'that never saw that video during its fold\'s training.')


if __name__ == '__main__':
    main()
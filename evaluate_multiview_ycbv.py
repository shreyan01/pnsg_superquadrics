"""
Evaluates the registry using MULTI-VIEW FUSED point clouds instead of
single frames -- directly tests whether the box-collapse we diagnosed
(diagnose_confusions.py: eps1 hitting its fitting bound, a signature of
degraded fits from partial/occluded single views) recovers once multiple
viewpoints of the same real object are fused before classification,
the way training already does.

Reuses train_registry_multiview.py's OWN tested functions
(discover_video_classes, _aggregate_and_fit) rather than reimplementing
aggregation -- the fitting/fusion code is identical to what training
used, just pointed at the val split and feeding classify_graph_* instead
of confirm_graph().

IMPORTANT, read before trusting the headline number: this evaluates one
instance per (video, class) pair, same granularity as training (157
examples from 187 pairs). Your val split will almost certainly collapse
from evaluate_on_ycbv.py's 1109 single-frame instances down to a much
smaller number of real object instances here -- exactly the same
granularity collapse we found in training. A smaller eval set means a
noisier accuracy estimate (wider real uncertainty), which is worth
reporting alongside the number, not just the point estimate. Report
BOTH this number and the single-frame evaluate_on_ycbv.py number --
they answer different, both-legitimate questions ("how good is one
observation" vs "how good is fused, continuous observation").

Usage:
    python3 evaluate_multiview_ycbv.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split val \
        --model trained_ycbv_ml_v2.json \
        --scoring ml \
        --workers 30
"""
import os
# MUST be set before numpy/scipy are imported -- see registry.py's
# module docstring for the real incident this fix is for (a load
# average of 935 on a nominally 30-process run).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from registry import Registry
from ycbv_training.train_registry_multiview import (
    discover_video_classes, discover_video_ids, _aggregate_and_fit, _init_worker,
    AXISYMMETRIC_WORDS,
)
from ycbv_training.ycb_classes import class_id_to_vocab

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val',
                     help='Which split file to read video IDs from (e.g. val, val_sample). '
                          'NOTE: this reads the whole video list, not evaluate_on_ycbv.py\'s '
                          'per-frame sampling -- the two scripts do not necessarily cover the '
                          'exact same set of videos even with the "same" split name if that '
                          'split file was built as a frame subsample. Check the printed video '
                          'count against what you expect.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--scoring', choices=['joint', 'ensembled', 'ml'], default='ml')
    ap.add_argument('--frame_stride', type=int, default=10,
                     help='Same meaning as training: stride when sampling frames WITHIN a '
                          'video for the fused point cloud (not a stride over videos).')
    ap.add_argument('--max_nfev', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    import os
    workers = args.workers or os.cpu_count()

    reg = Registry.load(args.model)
    print(f'Loaded {args.model}: {list(reg.graph_modes.keys())}')
    if args.scoring == 'ml' and getattr(reg, '_ml_classifier', None) is None:
        print('WARNING: --scoring ml requested but this model has no ML classifier -- '
              'falling back to ensembled scoring per-call. Results below are NOT the ml classifier.')

    split_path = os.path.join(args.dataset_root, 'image_sets', f'{args.split}.txt')
    if os.path.exists(split_path):
        video_ids = sorted({line.split('/')[0] for line in open(split_path) if line.strip()})
    else:
        print(f'No {split_path} found -- falling back to scanning all videos in the dataset root.')
        video_ids = discover_video_ids(args.dataset_root)
    print(f'Videos in split "{args.split}": {len(video_ids)}')

    work_items = []
    for vid in video_ids:
        for cid in discover_video_classes(args.dataset_root, vid):
            work_items.append((args.dataset_root, vid, cid, args.frame_stride, args.max_nfev))
    print(f'{len(work_items)} (video, class) pairs to aggregate, fit, and classify (workers={workers})')

    n_total, n_correct, n_skipped = 0, 0, 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    predictions = []

    pbar = tqdm(total=len(work_items), desc='Evaluating', unit='pair') if HAVE_TQDM else None
    started = time.time()

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        for video_id, class_id, vocab_word, graph, error in executor.map(_aggregate_and_fit, work_items):
            if pbar: pbar.update(1)
            if error or graph is None:
                n_skipped += 1
                continue

            if args.scoring == 'ml':
                ranked = reg.classify_graph_ml(graph, top_k=1)
            elif args.scoring == 'ensembled':
                ranked = reg.classify_graph_ensembled(graph, top_k=1)
            else:
                ranked = reg.classify_graph(graph, top_k=1)

            pred_word = ranked[0][0] if ranked else '__UNKNOWN__'
            confidence = float(ranked[0][1]) if ranked else 0.0

            n_total += 1
            per_class_total[vocab_word] += 1
            confusion[vocab_word][pred_word] += 1
            correct = int(pred_word == vocab_word)
            if correct:
                n_correct += 1
                per_class_correct[vocab_word] += 1

            predictions.append({'video': video_id, 'true_label': vocab_word,
                                 'predicted_label': pred_word, 'confidence': confidence, 'correct': correct})

            if pbar:
                pbar.set_postfix(accuracy=f'{n_correct/max(n_total,1)*100:.1f}%', n=n_total, refresh=False)

    if pbar: pbar.close()

    print(f'\n=== Multi-view fused results (n={n_total} real object instances, '
          f'{n_skipped} skipped, {time.time()-started:.0f}s) ===')
    print(f'Overall top-1 accuracy: {n_correct}/{n_total} = {n_correct/max(n_total,1)*100:.1f}%')
    print('\nPer-class accuracy:')
    for word in sorted(per_class_total):
        c, t = per_class_correct[word], per_class_total[word]
        print(f'  {word:10s}: {c:4d}/{t:4d} = {c/t*100:.1f}%')

    print('\nConfusion (true -> predicted):')
    for true_word in sorted(confusion):
        for pred_word, count in sorted(confusion[true_word].items(), key=lambda kv: -kv[1]):
            if pred_word != true_word:
                print(f'  {true_word} -> {pred_word}: {count} times')

    import csv
    out_path = f'multiview_eval_predictions_{args.split}_{args.scoring}.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video', 'true_label', 'predicted_label', 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(predictions)
    print(f'\nPer-instance predictions saved to {out_path}')
    print(f'\nCompare this n={n_total} multi-view number against evaluate_on_ycbv.py\'s n=1109 '
          f'single-frame number -- report both, they measure different, legitimate things.')


if __name__ == '__main__':
    main()
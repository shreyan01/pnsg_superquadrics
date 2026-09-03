"""
Bakes a per-frame feature export (from export_baseline_data.py) into an
existing trained model's ML classifier, WITHOUT touching its Welford
stats / graph structure / provenance. Fixes the real data-starvation
problem found in the first --scoring ml run: multiview training only
calls confirm_graph() once per (video, class) pair (157 total examples,
bowl at just 5), while the classifier needs the same per-frame volume
export_baseline_data.py already produces for val_sample (1109 examples).

Usage:
    # 1. Export per-frame features from the TRAIN split (mirrors the
    #    val_sample export already in baseline_data/)
    python3 export_baseline_data.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --out_dir ./baseline_data --split train --workers 30

    # 2. Bake them into your already-trained model
    python3 bake_ml_classifier.py \
        --model trained_ycbv_ml.json \
        --features baseline_data/features_train.npz \
        --out trained_ycbv_ml_v2.json

    # Optional: cap majority classes (e.g. for the class-imbalance
    # ablation -- see registry.py's rebuild_ml_classifier comment for
    # the real result this was testing: capping barely moved accuracy,
    # 89.4%->89.0%, ruling out raw imbalance as SAGE's dominant residual
    # weakness).
    python3 bake_ml_classifier.py \
        --model trained_ycbv_ml.json \
        --features baseline_data/features_train.npz \
        --out trained_ycbv_ml_v3.json \
        --max_per_class 2200
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import collections
import numpy as np
from registry import Registry


def cap_per_class(X, y, max_per_class, rng):
    """Random subsample any class over max_per_class. Real motivation:
    class_weight='balanced' reweights training LOSS, but with a 14.8:1
    raw imbalance (box=16619 vs bowl=1124 in the first --split train
    export), that reweighting alone wasn't enough -- box AUC was 0.989
    (near-perfect ranking) but precision only 0.805, box recall 1.000:
    the classic signature of a majority class still dominating the vote
    despite loss reweighting. Capping the raw counts directly, on top of
    class_weight, is the standard second lever for this."""
    y = np.asarray(y)
    keep_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_idx.extend(idx.tolist())
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Existing trained model (from train_registry_multiview.py)')
    ap.add_argument('--features', required=True, help='features_train.npz from export_baseline_data.py --split train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_per_class', type=int, default=None,
                     help='Cap each class to at most this many examples before training, on top of '
                          'class_weight=balanced. Try ~2x your smallest class as a starting point.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    reg = Registry.load(args.model)
    print(f'Loaded {args.model}: {list(reg.graph_modes.keys())}')
    print(f'Existing sparse reservoir total (video-level, from multiview training): '
          f'{sum(len(gm.part_modes["dominant"].raw_examples) for gms in reg.graph_modes.values() for gm in gms if "dominant" in gm.part_modes)}')

    d = np.load(args.features, allow_pickle=True)
    X, y = d['X'], d['y']
    print(f'Loaded {len(X)} per-frame examples from {args.features}: {dict(collections.Counter(y))}')

    if args.max_per_class:
        rng = np.random.default_rng(args.seed)
        X, y = cap_per_class(X, y, args.max_per_class, rng)
        print(f'Capped to max {args.max_per_class}/class -> {len(X)} examples: {dict(collections.Counter(y))}')

    reg.import_ml_training_data(X, y)
    built = reg.rebuild_ml_classifier()
    if not built:
        print('WARNING: rebuild_ml_classifier() returned False -- check sklearn is installed '
              '(pip install scikit-learn --break-system-packages) and that X/y were non-empty.')
    else:
        print(f'ML classifier rebuilt on {reg._ml_classifier_n_examples} total examples '
              '(sparse reservoir + bulk import combined).')

    reg.save(args.out)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
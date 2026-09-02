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
"""
import argparse
import numpy as np
from registry import Registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Existing trained model (from train_registry_multiview.py)')
    ap.add_argument('--features', required=True, help='features_train.npz from export_baseline_data.py --split train')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    reg = Registry.load(args.model)
    print(f'Loaded {args.model}: {list(reg.graph_modes.keys())}')
    print(f'Existing sparse reservoir total (video-level, from multiview training): '
          f'{sum(len(gm.part_modes["dominant"].raw_examples) for gms in reg.graph_modes.values() for gm in gms if "dominant" in gm.part_modes)}')

    d = np.load(args.features, allow_pickle=True)
    X, y = d['X'], d['y']
    import collections
    print(f'Importing {len(X)} per-frame examples from {args.features}: {dict(collections.Counter(y))}')

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
"""
SAGE's own sample-efficiency curve, and only that -- you already have the
baseline classifiers' curve (k-NN/SVM) from task3_sample_efficiency.py's
earlier run. This skips that whole combined baseline+SAGE machinery and
just calls the same, already-tested functions
(collect_training_graphs / build_registry / evaluate_registry) directly,
so you get a clean, minimal run with real progress bars throughout.

n=10 and n=20 are not attempted here -- bowl only has 5 total confirmed
(video, class) pairs in the full train split, so those budgets always
fail; this script sticks to what's actually reachable (n=1,2,5).

Usage:
    python3 sage_sample_efficiency.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --workers 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task3_sample_efficiency as t3
import sage_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--val_split', default='val_sample')
    ap.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 5])
    ap.add_argument('--frame_stride', type=int, default=10)
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--scoring', choices=['joint', 'ensembled', 'ml'], default='ml')
    ap.add_argument('--out', default='sage_sample_efficiency.json')
    args = ap.parse_args()

    import os
    workers = args.workers or os.cpu_count()

    available, reason = sage_pipeline.check_available()
    if not available:
        print("Cannot run:\n" + reason)
        raise SystemExit(1)

    from registry import assert_thread_limits_ok
    assert_thread_limits_ok()

    training, evaluation, Registry = sage_pipeline.load_sage_modules()

    print(f"Collecting training graphs (up to n={max(args.sizes)} per category)...")
    graphs = t3.collect_training_graphs(
        training_module=training,
        dataset_root=args.dataset_root,
        split=args.split,
        max_per_class=max(args.sizes),
        frame_stride=args.frame_stride,
        max_nfev=args.max_nfev,
        workers=workers,
        seed=t3.RANDOM_STATE,
    )
    print({word: len(g) for word, g in graphs.items()})

    results = []
    for n in args.sizes:
        registry, counts = t3.build_registry(Registry, training, graphs, n)
        if registry is None:
            print(f"\nn={n}: SKIPPED -- not enough confirmed examples in every category "
                  f"(have {({w: len(g) for w, g in graphs.items()})}).")
            continue

        model_path = t3.MODEL_DIR / f"sage_only_n{n}.json"
        t3.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        registry.save(str(model_path))
        print(f"\nn={n}: trained on {counts}, saved -> {model_path.name}")
        print(f"  evaluating on {args.val_split}...")

        metrics, predictions = t3.evaluate_registry(
            evaluation_module=evaluation,
            model_path=model_path,
            dataset_root=args.dataset_root,
            split=args.val_split,
            max_frames=args.max_frames,
            max_nfev=args.max_nfev,
            workers=workers,
            scoring=args.scoring,
        )
        print(f"  n={n}: overall={metrics['overall_accuracy']*100:.2f}%  "
              f"balanced={metrics['balanced_accuracy']*100:.2f}%  n_eval={metrics['n_evaluated']}")
        results.append({'n_per_class': n, **metrics})

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}\nSAGE SAMPLE-EFFICIENCY SUMMARY\n{'='*70}")
    for r in results:
        print(f"  n={r['n_per_class']:2d}: overall={r['overall_accuracy']*100:.2f}%  "
              f"balanced={r['balanced_accuracy']*100:.2f}%")
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
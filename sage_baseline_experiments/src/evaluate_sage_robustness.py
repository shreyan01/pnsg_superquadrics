"""
SAGE's own robustness to point downsampling and Gaussian sensor noise --
the missing half of Task 4. task4_robustness.py explicitly only
re-evaluates PointNet ("It does NOT re-evaluate SAGE itself... re-scoring
SAGE under noise needs the fitting pipeline and the dataset"). This
script IS that: it re-runs the real fitting pipeline (fit_one_instance,
same code export_baseline_data.py uses) on DEGRADED point clouds, then
queries the same trained ExtraTrees classifier via
reg._ml_classifier.predict() directly on the resulting feature vector --
bypassing the graph-wrapper step, since fit_one_instance's output IS
already the canonicalize() vector classify_graph_ml would extract from
a graph's dominant node anyway. Same math, no graph object needed.

Point-count levels (1024, 512, 256, 128, 64) and noise sigmas (0, 1.5,
2.0, 2.5, 3.0, 3.5, 4.0 mm) match Task 4's PointNet sweep exactly, so the
two are DIRECTLY comparable -- report them side by side, whichever way
it comes out. This is a real head-to-head, not a hedged one.

Usage:
    python3 evaluate_sage_robustness.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split val_sample \
        --model trained_ycbv_ml_v2.json \
        --workers 30
"""
import argparse
import collections
import csv
import os

# MUST be set before numpy/scipy are imported -- see registry.py's
# module docstring for the real incident this fix is for (a load
# average of 935 on a nominally 30-process run).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from registry import Registry
from export_baseline_data import fit_one_instance, _init_worker
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

POINT_LEVELS = [1024, 512, 256, 128, 64]
NOISE_SIGMAS_MM = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

_worker_reg = None


def _init_robustness_worker(model_path):
    global _worker_reg
    _init_worker()
    _worker_reg = Registry.load(model_path)


def _degrade_cloud(cloud, target_points, noise_sigma_m, rng):
    """Subsample to target_points (if cloud is larger; smaller clouds are
    left as-is, same convention Task 4 uses for PointNet -- it pads
    rather than fails on undersized clouds, but SAGE's fitter needs real
    points, not padding, so undersized clouds are just kept at their
    real size here). Then add isotropic Gaussian noise, same convention
    as Task 4's sigma-in-meters sweep."""
    if target_points is not None and len(cloud) > target_points:
        idx = rng.choice(len(cloud), size=target_points, replace=False)
        cloud = cloud[idx]
    if noise_sigma_m > 0:
        cloud = cloud + rng.normal(0, noise_sigma_m, size=cloud.shape)
    return cloud


def _process_one_frame_degraded(args_tuple):
    dataset_root, frame_key, target_points, noise_sigma_m, max_nfev, seed = args_tuple
    rng = np.random.default_rng(seed)
    results = []
    try:
        for vocab_word, cloud, _color in iter_frame_objects(dataset_root, frame_key, class_id_to_vocab):
            cloud = _degrade_cloud(cloud, target_points, noise_sigma_m, rng)
            if len(cloud) < 50:   # same floor as extract_object_cloud's min_points-ish guard
                continue
            f = fit_one_instance(cloud, vocab_word, max_nfev=max_nfev)
            if f is None:
                continue
            if _worker_reg._ml_classifier is None:
                continue
            pred = _worker_reg._ml_classifier.predict(f.reshape(1, -1))[0]
            results.append((vocab_word, str(pred)))
        return frame_key, results, None
    except Exception as e:
        return frame_key, [], str(e)


def run_sweep(dataset_root, split, model_path, sweep_kind, values, workers, max_nfev, seed):
    frame_keys = read_split_file(dataset_root, split)
    rows = []
    for value in values:
        target_points = value if sweep_kind == 'points' else None
        noise_sigma_m = (value / 1000.0) if sweep_kind == 'noise' else 0.0

        work_items = [(dataset_root, fk, target_points, noise_sigma_m, max_nfev, seed) for fk in frame_keys]
        n_total, n_correct = 0, 0
        per_class_total = collections.defaultdict(int)
        per_class_correct = collections.defaultdict(int)

        label = f'{value} points' if sweep_kind == 'points' else f'sigma={value}mm'
        pbar = tqdm(total=len(work_items), desc=f'Testing {label}', unit='frame') if HAVE_TQDM else None

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_robustness_worker,
                                  initargs=(model_path,)) as executor:
            for frame_key, results, error in executor.map(_process_one_frame_degraded, work_items):
                if pbar: pbar.update(1)
                if error:
                    continue
                for true_word, pred_word in results:
                    n_total += 1
                    per_class_total[true_word] += 1
                    if pred_word == true_word:
                        n_correct += 1
                        per_class_correct[true_word] += 1
        if pbar: pbar.close()

        acc = n_correct / max(n_total, 1)
        per_class_acc = {w: per_class_correct[w] / per_class_total[w] for w in per_class_total if per_class_total[w] > 0}
        balanced = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

        print(f'{label}: overall={acc*100:.2f}%  balanced={balanced*100:.2f}%  n={n_total}')
        row = {sweep_kind: value, 'overall_accuracy': acc, 'balanced_accuracy': balanced, 'n': n_total}
        for w, a in per_class_acc.items():
            row[f'{w}_accuracy'] = a
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val_sample')
    ap.add_argument('--model', required=True)
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()

    from registry import assert_thread_limits_ok
    assert_thread_limits_ok()

    workers = args.workers or os.cpu_count()

    args.model = os.path.abspath(args.model)
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"--model resolved to '{args.model}', which doesn't exist. "
            f"Check you're not running this from a different directory "
            f"than where the model file actually lives -- pass the full path."
        )

    reg_check = Registry.load(args.model)
    if reg_check._ml_classifier is None:
        print('WARNING: this model has no ML classifier built -- results will be all-zero. '
              'Use a model that went through bake_ml_classifier.py / rebuild_ml_classifier().')
    print(f'Loaded {args.model}: {list(reg_check.graph_modes.keys())}')

    print('\n' + '='*70)
    print('SAGE ROBUSTNESS -- POINT-COUNT DOWNSAMPLING')
    print('='*70)
    t0 = time.time()
    point_rows = run_sweep(args.dataset_root, args.split, args.model, 'points', POINT_LEVELS,
                            workers, args.max_nfev, args.seed)
    print(f'({time.time()-t0:.0f}s)')

    print('\n' + '='*70)
    print('SAGE ROBUSTNESS -- GAUSSIAN NOISE')
    print('='*70)
    t0 = time.time()
    noise_rows = run_sweep(args.dataset_root, args.split, args.model, 'noise', NOISE_SIGMAS_MM,
                            workers, args.max_nfev, args.seed)
    print(f'({time.time()-t0:.0f}s)')

    with open(os.path.join(args.out_dir, 'sage_robustness_points.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in point_rows for k in r}))
        writer.writeheader(); writer.writerows(point_rows)
    with open(os.path.join(args.out_dir, 'sage_robustness_noise.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in noise_rows for k in r}))
        writer.writeheader(); writer.writerows(noise_rows)

    print('\nSaved sage_robustness_points.csv and sage_robustness_noise.csv')
    print('These use the SAME point-count levels and noise sigmas as Task 4\'s PointNet sweep --')
    print('directly comparable, paste both outputs back for the paper\'s real head-to-head.')


if __name__ == '__main__':
    main()
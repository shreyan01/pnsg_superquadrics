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
import sys
from pathlib import Path

# The repo root (two levels up from src/) holds registry.py,
# export_baseline_data.py and ycbv_training/, which this module imports
# flatly below. Without this the docstring's `python3
# evaluate_sage_robustness.py ...` usage fails with
# ModuleNotFoundError: No module named 'registry' -- it only worked when
# imported by task2/task4, which add the root themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """Degrade and score every object in one frame.

    Yields (instance_id, true_word, pred_word) per object, with
    pred_word None when the instance could not be scored. The instance
    id matters: without it there is no way to hold the evaluated
    population fixed across degradation levels, which is what made the
    old curves incomparable (a harsher level silently dropped the
    instances it could not fit, so accuracy was computed over an easier
    population).

    instance_id format matches evaluate_on_ycbv.py: "<video>/<frame>#<n>".
    """
    dataset_root, frame_key, target_points, noise_sigma_m, max_nfev, seed = args_tuple
    rng = np.random.default_rng(seed)
    results = []
    try:
        for obj_index, (vocab_word, cloud, _color) in enumerate(
                iter_frame_objects(dataset_root, frame_key, class_id_to_vocab)):
            instance_id = f'{frame_key}#{obj_index}'
            cloud = _degrade_cloud(cloud, target_points, noise_sigma_m, rng)
            if len(cloud) < 50:   # same floor as extract_object_cloud's min_points-ish guard
                results.append((instance_id, vocab_word, None))
                continue
            f = fit_one_instance(cloud, vocab_word, max_nfev=max_nfev)
            if f is None:
                results.append((instance_id, vocab_word, None))
                continue
            if _worker_reg._ml_classifier is None:
                results.append((instance_id, vocab_word, None))
                continue
            pred = _worker_reg._ml_classifier.predict(f.reshape(1, -1))[0]
            results.append((instance_id, vocab_word, str(pred)))
        return frame_key, results, None
    except Exception as e:
        return frame_key, [], str(e)


def _process_one_cloud(args_tuple):
    """Fit + classify ONE already-loaded cloud.

    Same pipeline as _process_one_frame_degraded, but takes the cloud
    directly instead of reading it from a frame. That lets callers that
    already hold clouds in memory (task2_pointnet.py's exported
    pointclouds_val_sample.npz set) reuse this without touching the
    dataset.

    Unlike the frame-based path, this NEVER silently drops an instance:
    every input returns a row, with `reason` naming why it abstained.
    That is what makes a fixed-population comparison possible.
    """
    cloud, vocab_word, target_points, noise_sigma_m, max_nfev, seed = args_tuple
    rng = np.random.default_rng(seed)
    try:
        cloud = _degrade_cloud(np.asarray(cloud, dtype=np.float64),
                                target_points, noise_sigma_m, rng)
        if len(cloud) < 50:
            return vocab_word, None, 'too_few_points'

        # NOTE: fit_one_instance() picks its fitting strategy from
        # vocab_word -- the ground-truth label. That is the leak the
        # plan's item 2 replaces; kept here so this matches the current
        # pipeline exactly until that lands.
        f = fit_one_instance(cloud, vocab_word, max_nfev=max_nfev)
        if f is None:
            return vocab_word, None, 'implausible_fit'
        if _worker_reg._ml_classifier is None:
            return vocab_word, None, 'no_ml_classifier'

        pred = _worker_reg._ml_classifier.predict(f.reshape(1, -1))[0]
        return vocab_word, str(pred), None
    except Exception as e:
        return vocab_word, None, f'error: {type(e).__name__}: {e}'


def evaluate_clouds(clouds, labels, model_path, workers, max_nfev=1500,
                    target_points=None, noise_sigma_m=0.0, seed=0):
    """Evaluate a trained SAGE registry on in-memory clouds.

    Returns (metrics, per_instance_rows). `metrics` reports the
    abstained population separately from accuracy rather than folding it
    into the denominator, so `n_evaluated + n_abstained == len(labels)`
    always holds.
    """
    model_path = os.path.abspath(str(model_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAGE model not found: {model_path}")

    work_items = [
        (np.asarray(cloud), str(word), target_points, noise_sigma_m, max_nfev, seed)
        for cloud, word in zip(clouds, labels)
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=workers,
                              initializer=_init_robustness_worker,
                              initargs=(model_path,)) as executor:
        for instance_id, (true_word, pred_word, reason) in enumerate(
                executor.map(_process_one_cloud, work_items)):
            rows.append({'instance_id': instance_id, 'true_label': true_word,
                         'predicted_label': pred_word,
                         'abstained': int(pred_word is None),
                         'reason': reason or ''})

    scored = [r for r in rows if not r['abstained']]
    n_total = len(scored)
    n_correct = sum(r['true_label'] == r['predicted_label'] for r in scored)

    per_class_total = collections.defaultdict(int)
    per_class_correct = collections.defaultdict(int)
    for r in scored:
        per_class_total[r['true_label']] += 1
        if r['true_label'] == r['predicted_label']:
            per_class_correct[r['true_label']] += 1

    per_class_acc = {w: per_class_correct[w] / per_class_total[w]
                     for w in per_class_total if per_class_total[w] > 0}

    reasons = collections.Counter(r['reason'] for r in rows if r['abstained'])

    metrics = {
        'overall_accuracy': n_correct / max(n_total, 1),
        'balanced_accuracy': float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0,
        'n_evaluated': n_total,
        'n_abstained': len(rows) - n_total,
        'n_input': len(rows),
        'abstain_reasons': dict(reasons),
    }
    for w, a in per_class_acc.items():
        metrics[f'{w}_accuracy'] = a

    return metrics, rows


def run_sweep(dataset_root, split, model_path, sweep_kind, values, workers,
              max_nfev, seed, fixed_population=True):
    """Sweep one degradation axis.

    fixed_population=True (the default) scores the SAME set of instances
    at every level: the set that fitted successfully at the FIRST value
    in `values`, which is the undegraded baseline. An instance that
    fails at a harsher level counts as an ERROR, not as an absence.

    Why that is the default, and why the old behaviour is now opt-in:
    re-deriving the population per level lets a method silently drop the
    instances it can no longer fit, so accuracy gets computed over an
    easier and easier subset as conditions worsen. That flatters
    robustness exactly where a method struggles most -- a method that
    abstained on everything hard would look perfectly robust. The
    abstention rate is reported as its own series instead, which is the
    honest way to show the same information.

    `values` must therefore start at the undegraded condition
    (1024 points, or sigma=0), which both POINT_LEVELS and
    NOISE_SIGMAS_MM already do.
    """
    frame_keys = read_split_file(dataset_root, split)
    rows = []
    population = None       # instance ids fixed at the first level

    for value in values:
        target_points = value if sweep_kind == 'points' else None
        noise_sigma_m = (value / 1000.0) if sweep_kind == 'noise' else 0.0

        work_items = [(dataset_root, fk, target_points, noise_sigma_m, max_nfev, seed) for fk in frame_keys]

        scored = {}          # instance_id -> (true, pred)
        abstained = set()

        label = f'{value} points' if sweep_kind == 'points' else f'sigma={value}mm'
        pbar = tqdm(total=len(work_items), desc=f'Testing {label}', unit='frame') if HAVE_TQDM else None

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_robustness_worker,
                                  initargs=(model_path,)) as executor:
            for frame_key, results, error in executor.map(_process_one_frame_degraded, work_items):
                if pbar: pbar.update(1)
                if error:
                    continue
                for instance_id, true_word, pred_word in results:
                    if pred_word is None:
                        abstained.add(instance_id)
                    else:
                        scored[instance_id] = (true_word, pred_word)
        if pbar: pbar.close()

        if fixed_population and population is None:
            # First level defines the population for every later level.
            population = set(scored)
            print(f'  population fixed at {len(population)} instances '
                  f'(from "{label}"); every later level scores this same set')

        evaluated = sorted(population) if fixed_population else sorted(scored)

        n_total = len(evaluated)
        n_correct = 0
        n_lost = 0
        per_class_total = collections.defaultdict(int)
        per_class_correct = collections.defaultdict(int)

        for instance_id in evaluated:
            entry = scored.get(instance_id)
            if entry is None:
                # In the fixed population but unfittable at this level:
                # an error, not an absence.
                n_lost += 1
                continue
            true_word, pred_word = entry
            per_class_total[true_word] += 1
            if pred_word == true_word:
                n_correct += 1
                per_class_correct[true_word] += 1

        acc = n_correct / max(n_total, 1)
        per_class_acc = {w: per_class_correct[w] / per_class_total[w]
                          for w in per_class_total if per_class_total[w] > 0}
        balanced = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

        print(f'{label}: overall={acc*100:.2f}%  balanced={balanced*100:.2f}%  '
              f'n={n_total}  lost_at_this_level={n_lost}  abstained={len(abstained)}')

        row = {sweep_kind: value, 'overall_accuracy': acc,
               'balanced_accuracy': balanced, 'n': n_total,
               'n_lost_at_level': n_lost, 'n_abstained': len(abstained),
               'fixed_population': int(bool(fixed_population))}
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
    ap.add_argument('--per-level-population', dest='fixed_population',
                     action='store_false', default=True,
                     help='Re-derive the evaluated set at every degradation '
                          'level (the OLD behaviour). Off by default: it lets '
                          'a method silently drop the instances it can no '
                          'longer fit, so accuracy is computed over an easier '
                          'population as conditions worsen.')
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
                            workers, args.max_nfev, args.seed, args.fixed_population)
    print(f'({time.time()-t0:.0f}s)')

    print('\n' + '='*70)
    print('SAGE ROBUSTNESS -- GAUSSIAN NOISE')
    print('='*70)
    t0 = time.time()
    noise_rows = run_sweep(args.dataset_root, args.split, args.model, 'noise', NOISE_SIGMAS_MM,
                            workers, args.max_nfev, args.seed, args.fixed_population)
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
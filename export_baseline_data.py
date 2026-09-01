"""
Exports data for baseline comparison (Experiment #1: does a learned
classifier beat the registry on the SAME features, and how much do we
give up vs. a raw-point-cloud embedding model).

Produces TWO files per split (train / val_sample):
  features_<split>.npz  -- small, always exported
      X: (N, 13) float array -- the exact feature vector our registry uses
      y: (N,) string array -- true category label
      feature_names: (13,) string array -- column names, in order

  pointclouds_<split>.npz  -- larger, only if --include_pointclouds
      clouds: object array of (Mi, 3) float arrays (variable length per instance)
      y: (N,) string array -- same labels, same order

Usage:
    python3 export_baseline_data.py \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --out_dir ./baseline_data \
        --include_pointclouds --workers 30
"""
import os
# MUST be set before numpy/scipy are imported -- otherwise each worker
# process independently multithreads its own BLAS calls on top of the
# already-parallel process pool (found the hard way earlier tonight:
# a real load average of 935 on a nominally 30-process run).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from registry import canonicalize, FEATURE_KEYS
from superquadric import fit_superquadric, is_physically_plausible
from iterative_segment import iterative_two_part_segment
from radius_profile import compute_radial_profile
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab

AXISYMMETRIC_WORDS = {'can'}   # MUST match train_registry_multiview.py / evaluate_on_ycbv.py


def _init_worker():
    # OpenCV has its own internal thread pool, separate from BLAS -- see
    # the BLAS fix above for the full explanation of why this matters.
    import cv2
    cv2.setNumThreads(1)


def fit_one_instance(cloud, vocab_word, max_nfev=1500):
    """Returns the 13D feature vector, or None if the fit was
    implausible -- same logic as the real training/eval scripts, so
    the exported features exactly match what the registry itself is
    trained/evaluated on."""
    axisym = vocab_word in AXISYMMETRIC_WORDS
    if axisym:
        fitted, _ = fit_superquadric(cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
                                      min_size_multiplier=0.05, position_margin_multiplier=2.0,
                                      axisymmetric=True)
        if not is_physically_plausible(fitted):
            return None
        taper = compute_radial_profile(cloud, fitted)
        f = canonicalize(fitted, color_features=None, taper_features=taper)
    else:
        params_a, params_b, assignment = iterative_two_part_segment(cloud, verbose=False, max_nfev=max_nfev)
        if not is_physically_plausible(params_a):
            return None
        f = canonicalize(params_a, color_features=None, taper_features=None)
    return f


def _process_one_frame(args_tuple):
    dataset_root, frame_key, include_pointclouds, max_nfev = args_tuple
    results = []
    try:
        for true_word, cloud, color_features in iter_frame_objects(dataset_root, frame_key, class_id_to_vocab):
            f = fit_one_instance(cloud, true_word, max_nfev=max_nfev)
            if f is None:
                continue
            cloud_out = cloud.astype(np.float32) if include_pointclouds else None
            results.append((f, true_word, cloud_out))
        return frame_key, results, None
    except Exception as e:
        return frame_key, [], str(e)


def export_split(dataset_root, split_name, out_dir, include_pointclouds, workers, max_nfev=1500):
    features, labels, clouds = [], [], []

    frame_keys = read_split_file(dataset_root, split_name)
    work_items = [(dataset_root, fk, include_pointclouds, max_nfev) for fk in frame_keys]

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = [executor.submit(_process_one_frame, item) for item in work_items]
        completed = as_completed(futures)
        iterator = tqdm(completed, total=len(futures), desc='Exporting', unit='frame') if HAVE_TQDM else completed

        for future in iterator:
            frame_key, results, error = future.result()
            if error:
                msg = f'  [frame {frame_key}] skipped: {error}'
                (iterator.write(msg) if HAVE_TQDM else print(msg))
                continue
            for f, true_word, cloud_out in results:
                features.append(f)
                labels.append(true_word)
                if include_pointclouds:
                    clouds.append(cloud_out)

    features = np.array(features, dtype=np.float64)
    labels = np.array(labels, dtype=object)

    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, f'features_{split_name}.npz')
    np.savez_compressed(feat_path, X=features, y=labels, feature_names=np.array(FEATURE_KEYS, dtype=object))
    print(f'Saved {len(labels)} instances -> {feat_path}')

    if include_pointclouds:
        cloud_path = os.path.join(out_dir, f'pointclouds_{split_name}.npz')
        clouds_arr = np.empty(len(clouds), dtype=object)
        for i, c in enumerate(clouds):
            clouds_arr[i] = c
        np.savez_compressed(cloud_path, clouds=clouds_arr, y=labels)
        print(f'Saved {len(labels)} point clouds -> {cloud_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--out_dir', default='./baseline_data')
    ap.add_argument('--split', default='val_sample',
                     help='Which split to export -- default val_sample matches our reported '
                          'eval numbers, so a baseline trained on train_registry_multiview\'s '
                          'training data and tested on this export is directly comparable.')
    ap.add_argument('--include_pointclouds', action='store_true')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    print(f'Exporting {args.split} (single-frame instances, workers={args.workers})...')
    export_split(args.dataset_root, args.split, args.out_dir, args.include_pointclouds, args.workers)

    readme_path = os.path.join(args.out_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(
f"""BASELINE COMPARISON DATA -- README

features_{args.split}.npz
------------------------
Load with: data = numpy.load('features_{args.split}.npz', allow_pickle=True)
  data['X']             -- (N, 13) float array, the feature vector
  data['y']              -- (N,) string array, true label
  data['feature_names']  -- (13,) string array, column names in order

Feature columns, in order:
  0: a1               -- primary radius/half-width (meters)
  1: a2                -- secondary radius/half-width (meters)
  2: eps1               -- shape exponent, vertical roundness (0=sharp corner, 1=round)
  3: eps2                -- shape exponent, horizontal roundness (0=sharp corner, 1=round)
  4: a3                   -- half-height (meters)
  5: hue                   -- color hue in degrees (0 if unavailable -- not used in this export)
  6: saturation             -- color saturation 0-1 (0 if unavailable)
  7: r_10                    -- radius at 10% of object height (meters, 0 if not axisymmetric)
  8: r_30                     -- radius at 30% of object height
  9: r_50                      -- radius at 50% of object height
  10: r_70                      -- radius at 70% of object height
  11: r_90                       -- radius at 90% of object height
  12: aspect_ratio                -- height / max(a1,a2)

NOTE: hue/saturation are all zero in this export (color intentionally
excluded to isolate pure-geometry comparison -- ping if you want a
color-included version instead). r_10..r_90 are zero for non-round
categories (box, mug, bowl, bottle) since that concept doesn't apply --
this matches exactly how our own registry treats missing features.

Labels (y): 'box', 'mug', 'bowl', 'can', 'bottle'

pointclouds_{args.split}.npz (only if --include_pointclouds was used)
------------------------------------------------------------
Load with: data = numpy.load('pointclouds_{args.split}.npz', allow_pickle=True)
  data['clouds']  -- object array, each element is (Mi, 3) float32 array
                     of real depth-camera points for one object instance
  data['y']        -- (N,) string array, same order as clouds, same labels

Both files use the SAME instance order and SAME split ('{args.split}') as
our own reported evaluation results, so any baseline trained/tested on
this data is directly, fairly comparable to our numbers.
""")
    print(f'Wrote {readme_path}')


if __name__ == '__main__':
    main()
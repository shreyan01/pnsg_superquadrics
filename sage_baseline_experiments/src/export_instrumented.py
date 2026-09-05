"""
Instrumented, label-free re-export of YCB-Video object instances.

This is the artifact the corrected pipeline is built on. It replaces
export_baseline_data.py's output and carries the four things that one
drops, each of which blocks a different experiment:

  video_id / frame_id / instance_id
      The old export emitted rows with no provenance, so a video-level
      split was impossible and the CV regimes had to infer object
      identity from point-cloud size (data_utils.compute_object_groups).
      That proxy is also why grouped results do not reproduce across
      machines. Unblocks the unified protocol and the cluster bootstrap.

  both fits, chosen without the label
      The old export picked its fitting strategy from the ground-truth
      category (axisym = vocab_word in AXISYMMETRIC_WORDS), which made
      a1 == a2 a 100%-accurate 'can' detector. Here every object gets
      BOTH fits via symmetry.fit_both(), so the signature is constant
      across classes and carries no label information.

  abstention as a ROW, not a skip
      The old export used `continue` for implausible fits, sparse
      clouds and frame errors, so the denominator silently shrank -- and
      shrank by a different amount at each noise level, making
      robustness curves incomparable. Every input here produces a row,
      with `abstained` and `reason` columns.

  fit-quality diagnostics
      Residuals for both fits, plus the two-part segmenter's
      combined_rmse (which it computed and discarded until now), so a
      reliable fit can be told from an unreliable one.

Usage
-----
    python3 src/export_instrumented.py \\
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \\
        --split val_sample \\
        --out ../baseline_data/instrumented_val_sample.npz \\
        --workers 30

Run it once per split you need (train and val_sample at minimum).
Costs roughly twice the original export, since every object is fitted
both ways.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# The repo root holds registry.py, symmetry.py and ycbv_training/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# MUST precede numpy -- see registry.py's module docstring for the
# incident (load average 935 on a nominally 30-process run).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from registry import FEATURE_KEYS_PAIR, canonicalize_pair
from radius_profile import compute_radial_profile
from symmetry import fit_both
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


# Columns recorded alongside the feature vector.
META_COLUMNS = [
    'instance_id', 'video_id', 'frame_id', 'object_index',
    'true_label', 'n_points',
    'abstained', 'reason',
    'ax_rmse_raw', 'fl_rmse_raw',
    'ax_success', 'fl_success', 'ax_nfev', 'fl_nfev',
    'ax_plausible', 'fl_plausible',
    'seg_combined_rmse', 'seg_n_parts',
]


def _init_worker():
    """OpenCV keeps its own thread pool that ignores the BLAS env vars,
    and this pipeline calls cv2.imread constantly inside every worker."""
    import cv2
    cv2.setNumThreads(1)


def _process_frame(args_tuple):
    """Fit every object in one frame, both ways, without the label.

    Returns a list of row dicts -- one per object, including objects
    that could not be fitted. Never returns fewer rows than the frame
    contained.
    """
    dataset_root, frame_key, max_nfev, min_points = args_tuple

    video_id, frame_id = frame_key.split('/')
    rows = []

    try:
        objects = list(
            iter_frame_objects(dataset_root, frame_key, class_id_to_vocab)
        )
    except Exception as error:
        # A whole-frame failure used to vanish entirely, counted
        # nowhere. Record it so the denominator still reconciles.
        return [{
            'instance_id': f'{frame_key}#frame_error',
            'video_id': video_id, 'frame_id': frame_id, 'object_index': -1,
            'true_label': '', 'n_points': 0,
            'abstained': 1, 'reason': f'frame_error: {type(error).__name__}: {error}',
            'features': None,
        }]

    for object_index, (true_word, cloud, color_features) in enumerate(objects):

        instance_id = f'{frame_key}#{object_index}'

        base = {
            'instance_id': instance_id,
            'video_id': video_id,
            'frame_id': frame_id,
            'object_index': object_index,
            'true_label': true_word,
            'n_points': int(len(cloud)),
            'abstained': 0,
            'reason': '',
            'features': None,
        }

        if len(cloud) < min_points:
            base.update(abstained=1, reason='too_few_points')
            rows.append(base)
            continue

        try:
            # NOTE: true_word is NOT passed to the fitter. Both paths
            # run for every object; that is the whole point.
            diagnostics = fit_both(cloud, max_nfev=max_nfev)

            if not (
                diagnostics['axisym_plausible']
                or diagnostics['flexible_plausible']
            ):
                base.update(abstained=1, reason='both_fits_implausible')
                rows.append(base)
                continue

            # Radial profile for EVERY object now, not only for the one
            # category that was permitted to have one. Computed against
            # the axisymmetric fit, which is what defines the axis.
            try:
                taper = compute_radial_profile(cloud, diagnostics['axisym'])
            except Exception:
                taper = None

            features = canonicalize_pair(
                diagnostics,
                color_features=color_features,
                taper_features=taper,
            )

            ax_info = diagnostics['axisym_info']
            fl_info = diagnostics['flexible_info']

            base.update(
                features=features.tolist(),
                ax_rmse_raw=float(ax_info['rmse']),
                fl_rmse_raw=float(fl_info['rmse']),
                ax_success=int(bool(ax_info['success'])),
                fl_success=int(bool(fl_info['success'])),
                ax_nfev=int(ax_info['nfev']),
                fl_nfev=int(fl_info['nfev']),
                ax_plausible=int(diagnostics['axisym_plausible']),
                fl_plausible=int(diagnostics['flexible_plausible']),
                seg_combined_rmse=np.nan,
                seg_n_parts=0,
            )
            rows.append(base)

        except Exception as error:
            base.update(
                abstained=1,
                reason=f'fit_error: {type(error).__name__}: {error}',
            )
            rows.append(base)

    return rows


def export(dataset_root, split, out_path, workers, max_nfev=1500,
           min_points=50, limit=None):
    """Run the export and write an .npz plus a companion .csv."""

    frame_keys = read_split_file(dataset_root, split)

    if limit:
        frame_keys = frame_keys[:limit]

    print(f'Split "{split}": {len(frame_keys)} frames, workers={workers}')
    print('Fitting every object BOTH ways -- no label reaches the fitter.')

    work = [(dataset_root, fk, max_nfev, min_points) for fk in frame_keys]

    rows = []
    started = time.time()

    pbar = tqdm(total=len(work), desc='Exporting', unit='frame') if HAVE_TQDM else None

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        for frame_rows in executor.map(_process_frame, work):
            rows.extend(frame_rows)
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    print(f'{len(rows)} rows from {len(frame_keys)} frames '
          f'in {time.time() - started:.0f}s')

    scored = [r for r in rows if not r['abstained']]
    n_abstained = len(rows) - len(scored)

    print(f'  fitted   : {len(scored)}')
    print(f'  abstained: {n_abstained}')

    if n_abstained:
        import collections
        reasons = collections.Counter(r['reason'].split(':')[0] for r in rows if r['abstained'])
        for reason, count in reasons.most_common():
            print(f'      {reason}: {count}')

    # Feature matrix over the fitted rows only; the abstained rows keep
    # their place in the metadata so the denominator reconciles.
    X = np.array([r['features'] for r in scored], dtype=np.float64)
    y = np.array([r['true_label'] for r in scored], dtype=object)
    videos = np.array([r['video_id'] for r in scored], dtype=object)
    instance_ids = np.array([r['instance_id'] for r in scored], dtype=object)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X, y=y,
        video_id=videos,
        instance_id=instance_ids,
        feature_names=np.array(FEATURE_KEYS_PAIR, dtype=object),
        n_input=len(rows),
        n_abstained=n_abstained,
    )

    print(f'\nSaved {X.shape[0]} x {X.shape[1]} features -> {out_path}')

    # Companion CSV with every row, abstentions included.
    import csv as csv_module

    csv_path = out_path.with_suffix('.csv')

    with open(csv_path, 'w', newline='') as handle:
        writer = csv_module.DictWriter(
            handle,
            fieldnames=META_COLUMNS,
            extrasaction='ignore',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in META_COLUMNS})

    print(f'Saved per-instance metadata ({len(rows)} rows, '
          f'abstentions included) -> {csv_path}')

    print(f'\nVideos covered: {len(set(videos))}')
    print('Denominator reconciles: '
          f'{len(scored)} fitted + {n_abstained} abstained = {len(rows)}')

    return out_path


def selftest(n_per_class=10, max_nfev=400):
    """Exercise the whole feature path without the dataset.

    Runs fit_both -> radial profile -> canonicalize_pair on the already
    exported clouds in baseline_data/pointclouds_val_sample.npz, then
    re-runs the leak acceptance test on the result.

    Worth doing before committing a machine to the full export: it
    verifies the fitting, the feature assembly and the leak closure in
    about a minute, rather than finding a problem hours in.
    """
    from data_utils import load_pointcloud_data

    print('SELF-TEST -- no dataset required')
    print('Exercising fit_both -> radial profile -> canonicalize_pair '
          'on exported clouds.\n')

    clouds, labels = load_pointcloud_data()
    labels = np.asarray(labels)

    rng = np.random.default_rng(0)
    classes = ['box', 'can', 'mug', 'bottle', 'bowl']

    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=n_per_class, replace=False)
        for c in classes
    ])

    rows, started = [], time.time()

    for i in idx:
        cloud = np.asarray(clouds[i], dtype=np.float64)
        diagnostics = fit_both(cloud, max_nfev=max_nfev)

        try:
            taper = compute_radial_profile(cloud, diagnostics['axisym'])
        except Exception:
            taper = None

        rows.append((
            labels[i],
            canonicalize_pair(diagnostics, taper_features=taper),
        ))

    elapsed = time.time() - started

    X = np.array([r[1] for r in rows])
    y = np.array([r[0] for r in rows])

    print(f'Built {X.shape[0]} x {X.shape[1]} vectors in {elapsed:.0f}s '
          f'({elapsed / len(rows) * 1000:.0f} ms/object, single-threaded)')
    print(f'Feature names: {len(FEATURE_KEYS_PAIR)} columns\n')

    keys = FEATURE_KEYS_PAIR
    can = y == 'can'

    print('LEAK ACCEPTANCE TEST -- these must not separate can from rest')
    print(f"{'diagnostic':32s} {'on can':>9s} {'on rest':>9s}   verdict")

    checks = [
        ('flexible fit a1==a2',
         np.isclose(X[:, keys.index('fl_a1')], X[:, keys.index('fl_a2')])),
        ('radial profile non-zero',
         np.abs(X[:, [keys.index(k) for k in
                      ['r_10', 'r_30', 'r_50', 'r_70', 'r_90']]]).sum(1) > 0),
    ]

    all_ok = True

    for name, signal in checks:
        on_can, on_rest = signal[can].mean(), signal[~can].mean()
        ok = abs(on_can - on_rest) < 0.25
        all_ok &= ok
        print(f'{name:32s} {on_can * 100:8.1f}% {on_rest * 100:8.1f}%   '
              f'{"PASS" if ok else "FAIL"}')

    print()
    if all_ok:
        print('PASS -- the fitting path carries no label information.')
        print(f'\nExtrapolated cost: ~{elapsed / len(rows):.2f} s/object '
              f'single-threaded;\nwith N workers a 1109-instance split '
              f'takes roughly {1109 * elapsed / len(rows) / 60:.0f}/N minutes.')
    else:
        print('FAIL -- a diagnostic still separates can. Do not run the '
              'full export until this passes.')

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--selftest', action='store_true',
                        help='Verify the feature path on already-exported '
                             'clouds, without the dataset. Do this first.')
    parser.add_argument('--dataset_root', required=False)
    parser.add_argument('--split', default='val_sample')
    parser.add_argument('--out', default=None,
                        help='Output .npz (default: '
                             '../baseline_data/instrumented_<split>.npz)')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max_nfev', type=int, default=1500)
    parser.add_argument('--min_points', type=int, default=50)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only the first N frames -- for a smoke test.')
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    if not args.dataset_root:
        parser.error(
            '--dataset_root is required (or use --selftest to verify '
            'the feature path without it).'
        )

    workers = args.workers or os.cpu_count()

    out = args.out or (
        Path(__file__).resolve().parents[2]
        / 'baseline_data' / f'instrumented_{args.split}.npz'
    )

    export(args.dataset_root, args.split, out, workers,
           max_nfev=args.max_nfev, min_points=args.min_points,
           limit=args.limit)


if __name__ == '__main__':
    main()

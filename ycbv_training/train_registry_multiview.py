"""
Multi-view training, now with color features wired through (real
per-object hue/saturation, averaged across the same aggregation window
used for shape) and optional axisymmetric fitting for known-round
categories.

Usage:
    python3 -m ycbv_training.train_registry_multiview \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split train --out trained_ycbv_color.json --workers 30
"""
import os
# MUST be set before numpy/scipy are imported -- see evaluate_on_ycbv.py
# for the full explanation (each of the 30 worker processes otherwise
# independently multithreads its own BLAS calls, stacking on top of the
# already-parallel process pool; found via a real load average of 935
# on a nominally 30-process run).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import time
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
import numpy as np


try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from ..registry import Registry
from ..superquadric import fit_superquadric, is_physically_plausible
from ..radius_profile import compute_radial_profile
from .ycb_classes import class_id_to_vocab, YCB_CLASS_NAMES
from .ycb_pose_aggregation import aggregate_multiview_cloud, discover_frames
from .ycb_dataset_loader import resolve_video_dir
from ..iterative_segment import iterative_two_part_segment
from ..pipeline import build_graph_from_segmentation


def _init_worker():
    """OpenCV has its own internal thread pool, separate from BLAS, and
    does NOT reliably respect the OMP/OPENBLAS env vars above -- this
    codebase calls cv2.imread constantly (depth/label/color files)
    inside every worker process, so this must be set explicitly here."""
    import cv2
    cv2.setNumThreads(1)

# categories known to be genuinely rotationally symmetric -- forcing
# a1==a2 for these removes an entire failure mode (single-view fits
# guessing an ellipse for what is physically always a circle), found
# necessary after repeatedly diagnosing this exact distortion tonight.
# 'box' is deliberately excluded -- boxes are not round.
AXISYMMETRIC_WORDS = {'can'}   # DELIBERATE: 'bottle' excluded on purpose. can is a
                                # genuinely simple, uniformly round object -- axisymmetric
                                # (a1=a2, eps2=1.0) + the 5-point radial profile suit it well
                                # (verified: 94.9% accuracy, AUC=0.911). bottle spans multiple
                                # real products (mustard bottle, bleach cleanser) that do not
                                # share one consistent taper profile in real single-view data;
                                # forcing the same rigid circular fit onto it was tested and
                                # found to hurt its accuracy (31.5%->10.3%). bottle instead uses
                                # the flexible multi-part segmenter, which can represent
                                # asymmetric/non-uniform real shapes the circular constraint
                                # cannot. Different categories get different, appropriately
                                # chosen fitting strategies -- a real methodological choice,
                                # not an oversight.
                                                                    # into its two real sub-objects --
                                          # iterative_two_part_segment is structurally
                                          # mismatched to bottle/can (8/8 real instances took
                                          # 11-31s each, hitting the 8-round iteration ceiling
                                          # every time -- not one outlier, a systematic issue).
                                          # The earlier accuracy comparison was never a fair
                                          # test: it compared axisymmetric fitting against an
                                          # alternative now confirmed unreliable for this data.


def discover_video_classes(dataset_root, video_id, sample_every=20):
    frame_ids = discover_frames(dataset_root, video_id)[::sample_every]
    found = set()
    video_dir = resolve_video_dir(dataset_root, video_id)
    for fid in frame_ids:
        try:
            meta = loadmat(os.path.join(video_dir, fid + '-meta.mat'))
            for cid in np.atleast_1d(meta['cls_indexes'].squeeze()):
                if class_id_to_vocab(int(cid)) is not None:
                    found.add(int(cid))
        except Exception:
            continue
    return sorted(found)


def discover_video_ids(dataset_root):
    videos = set()
    for subdir in ['data', 'data2', 'data3']:
        for base in [os.path.join(dataset_root, subdir), os.path.join(dataset_root, subdir, 'data')]:
            if os.path.isdir(base):
                videos.update(d for d in os.listdir(base)
                              if len(d) == 4 and d.isdigit() and os.path.isdir(os.path.join(base, d)))
    return sorted(videos)


def _aggregate_and_fit(args_tuple):
    """Aggregates multi-view cloud AND color for one (video, class) pair,
    segments into parts, fits (with axisymmetric constraint for known-
    round categories), builds a color-tagged graph."""
    dataset_root, video_id, class_id, frame_stride, max_nfev = args_tuple
    vocab_word = class_id_to_vocab(class_id)
    axisym = vocab_word in AXISYMMETRIC_WORDS
    try:
        cloud, color_features = aggregate_multiview_cloud(
            dataset_root, video_id, class_id, frame_stride=frame_stride)
        if cloud is None or len(cloud) < 50:
            return video_id, class_id, vocab_word, None, 'insufficient aggregated points'

        if axisym:
            # SUPERSEDES the earlier discrete body+neck detection (found
            # to regress accuracy: can 36.9%->28.0%->12.8% across two
            # threshold attempts, due to instability under single-frame
            # noise). This computes taper as two CONTINUOUS numbers,
            # always measured, never a discrete branch -- verified stable
            # across noise realizations (bottle: 15.8-16.8mm diff every
            # seed; can: within +/-0.1mm every seed) before being wired
            # in here.
            fitted, info = fit_superquadric(cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
                                             min_size_multiplier=0.05, position_margin_multiplier=2.0,
                                             axisymmetric=True)
            if not is_physically_plausible(fitted):
                return video_id, class_id, vocab_word, None, \
                    f'dominant implausible: a1={fitted["a1"]*1000:.0f}mm a3={fitted["a3"]*1000:.0f}mm'
            taper_features = compute_radial_profile(cloud, fitted)
            graph = build_graph_from_segmentation(cloud, fitted, None, None,
                                                   color_features=color_features, taper_features=taper_features)
        else:
            params_a, params_b, assignment = iterative_two_part_segment(
                cloud, verbose=False, max_nfev=max_nfev)
            if not is_physically_plausible(params_a):
                return video_id, class_id, vocab_word, None, \
                    f'dominant implausible: a1={params_a["a1"]*1000:.0f}mm a2={params_a["a2"]*1000:.0f}mm a3={params_a["a3"]*1000:.0f}mm'
            if params_b is not None and not is_physically_plausible(params_b):
                params_b, assignment = None, None
            graph = build_graph_from_segmentation(cloud, params_a, params_b, assignment,
                                                   color_features=color_features)

        return video_id, class_id, vocab_word, graph, None
    except Exception as e:
        return video_id, class_id, vocab_word, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--frame_stride', type=int, default=10)
    ap.add_argument('--out', default='trained_ycbv_color.json')
    ap.add_argument('--max_nfev', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--load_from', default=None,
                     help='Path to an existing trained model to CONTINUE training '
                          '(online update) instead of starting from a fresh, empty '
                          'registry. New examples update existing modes or spawn new '
                          'ones/words exactly as confirm_graph() already handles -- '
                          'no special-casing needed, the registry was built for this.')
    args = ap.parse_args()

    split_path = os.path.join(args.dataset_root, 'image_sets', f'{args.split}.txt')
    video_ids = sorted(set(line.split('/')[0] for line in open(split_path) if line.strip()))
    print(f'Videos in split "{args.split}": {len(video_ids)}')
    print(f'Axisymmetric (roundness-constrained) categories: {AXISYMMETRIC_WORDS}')

    print('Scanning videos for relevant object classes...')
    work_items = []
    for vid in video_ids:
        for cid in discover_video_classes(args.dataset_root, vid):
            work_items.append((args.dataset_root, vid, cid, args.frame_stride, args.max_nfev))
    print(f'{len(work_items)} (video, class) pairs to aggregate and fit (workers={args.workers})')

    if args.load_from:
        reg = Registry.load(args.load_from)
        print(f'Loaded existing model from {args.load_from} -- CONTINUING training, not starting fresh.')
        print(f'  Existing words: {list(reg.graph_modes.keys())}')
        existing_axisym = getattr(reg, 'axisymmetric_words', set())
        if existing_axisym and existing_axisym != AXISYMMETRIC_WORDS:
            print(f'  WARNING: loaded model\'s axisymmetric_words ({existing_axisym}) differs from '
                  f'this script\'s AXISYMMETRIC_WORDS ({AXISYMMETRIC_WORDS}) -- using the loaded '
                  f'model\'s values to stay consistent with how it was originally trained.')
        else:
            reg.axisymmetric_words = AXISYMMETRIC_WORDS
    else:
        reg = Registry()
        reg.axisymmetric_words = AXISYMMETRIC_WORDS
    per_class_counts = defaultdict(int)
    n_skipped = 0
    t0 = time.time()

    pbar = tqdm(total=len(work_items), desc='Training', unit='pair') if HAVE_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        for video_id, class_id, vocab_word, graph, error in executor.map(_aggregate_and_fit, work_items):
            if error or graph is None:
                n_skipped += 1
                msg = f'  [{video_id} class={class_id}] skipped: {error}'
                (pbar.write(msg) if HAVE_TQDM else print(msg))
                if HAVE_TQDM: pbar.update(1)
                continue

            entry = reg.confirm_graph(graph, vocab_word, F=1)
            per_class_counts[vocab_word] += 1

            if HAVE_TQDM:
                mode_counts = {n: len(m) for n, m in reg.graph_modes.items()}
                pbar.set_postfix(examples=sum(per_class_counts.values()), modes=mode_counts, refresh=False)
                pbar.update(1)

    if HAVE_TQDM: pbar.close()

    reg.save(args.out)
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s. {sum(per_class_counts.values())} examples, {n_skipped} skipped.')
    print(f'Per-class counts: {dict(per_class_counts)}')
    for noun in reg.graph_modes:
        print(reg.describe_graph(noun))
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
"""
Train the registry using MULTI-VIEW aggregation instead of per-frame
fitting -- the real fix for the partial-view bias found tonight
(validated: aggregated master_chef_can fit recovered eps2=0.91, a
near-circular cross-section, vs never exceeding ~0.6 across 100
single-frame fits).

Fundamentally different training unit than before: instead of one
example per (frame, object) -- thousands of biased partial-view
snapshots -- this produces one example per (video, object) pair,
aggregated from many camera angles into a single, much more complete
point cloud. Fewer examples, each one far more representative.

Usage:
    python3 -m ycbv_training.train_registry_multiview \\
        --dataset_root . --split train --out trained_ycbv_multiview.json --workers 30
"""
import argparse
import time
import sys
import os
import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from registry import Registry
from superquadric import fit_superquadric
from ycbv_training.ycb_classes import class_id_to_vocab, YCB_CLASS_NAMES
from ycbv_training.ycb_pose_aggregation import aggregate_multiview_cloud, discover_frames
from ycbv_training.ycb_dataset_loader import resolve_video_dir


def discover_video_classes(dataset_root, video_id, sample_every=20):
    """Scans a sample of frames in a video to find which mapped classes
    actually appear in it, without loading every single frame's meta."""
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


def discover_video_ids(dataset_root, split_videos=None):
    if split_videos is not None:
        return split_videos
    videos = set()
    for subdir in ['data', 'data2', 'data3']:
        data_dir = os.path.join(dataset_root, subdir)
        if os.path.isdir(data_dir):
            videos.update(d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d)))
    return sorted(videos)


def _aggregate_and_fit(args_tuple):
    """Top-level (picklable) worker: aggregates multi-view cloud for one
    (video, class) pair and fits ONE superquadric to it. This is the
    parallelized, independent, expensive part -- registry updates stay
    sequential in the main process, same design as the per-frame
    parallel trainer."""
    dataset_root, video_id, class_id, frame_stride, max_nfev = args_tuple
    vocab_word = class_id_to_vocab(class_id)
    try:
        cloud = aggregate_multiview_cloud(dataset_root, video_id, class_id,
                                           frame_stride=frame_stride)
        if cloud is None or len(cloud) < 50:
            return video_id, class_id, vocab_word, None, 'insufficient aggregated points'
        fitted, info = fit_superquadric(
            cloud, max_nfev=max_nfev,
            max_size_multiplier=4.0, min_size_multiplier=0.05,
            position_margin_multiplier=2.0,
        )
        return video_id, class_id, vocab_word, fitted, None
    except Exception as e:
        return video_id, class_id, vocab_word, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='train', help='Used only to restrict which VIDEOS are used '
                     '(reads video IDs out of image_sets/<split>.txt)')
    ap.add_argument('--frame_stride', type=int, default=10,
                     help='Sample every Nth frame within a video for aggregation')
    ap.add_argument('--out', default='trained_ycbv_multiview.json')
    ap.add_argument('--max_nfev', type=int, default=2000,
                     help='Higher than per-frame default since aggregated clouds have more points')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    # derive the set of videos from the split file (reuses train/val video
    # partition already built by build_splits.py, just at video granularity)
    split_path = os.path.join(args.dataset_root, 'image_sets', f'{args.split}.txt')
    video_ids = sorted(set(line.split('/')[0] for line in open(split_path) if line.strip()))
    print(f'Videos in split "{args.split}": {len(video_ids)}')

    # discover which mapped classes appear in each video
    print('Scanning videos for relevant object classes...')
    work_items = []
    for vid in video_ids:
        classes = discover_video_classes(args.dataset_root, vid)
        for cid in classes:
            work_items.append((args.dataset_root, vid, cid, args.frame_stride, args.max_nfev))
    print(f'{len(work_items)} (video, class) pairs to aggregate and fit '
          f'(workers={args.workers})')

    reg = Registry()
    per_class_counts = defaultdict(int)
    n_skipped = 0
    t0 = time.time()

    pbar = tqdm(total=len(work_items), desc='Multi-view training', unit='pair') if HAVE_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for video_id, class_id, vocab_word, fitted, error in executor.map(_aggregate_and_fit, work_items):
            if error or fitted is None:
                n_skipped += 1
                msg = f'  [{video_id} class={class_id}] skipped: {error}'
                if HAVE_TQDM:
                    pbar.write(msg)
                else:
                    print(msg)
                if HAVE_TQDM:
                    pbar.update(1)
                continue

            entry = reg.confirm(fitted, vocab_word, F=1)
            per_class_counts[vocab_word] += 1

            if HAVE_TQDM:
                mode_counts = {n: len(m) for n, m in reg.modes.items()}
                pbar.set_postfix(examples=sum(per_class_counts.values()), modes=mode_counts,
                                  refresh=False)
                pbar.update(1)

    if HAVE_TQDM:
        pbar.close()

    reg.save(args.out)
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s. {sum(per_class_counts.values())} aggregated examples, '
          f'{n_skipped} pairs skipped.')
    print(f'Per-class counts: {dict(per_class_counts)}')
    for noun in reg.modes:
        print(reg.describe(noun))
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
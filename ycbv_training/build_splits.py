"""
Builds train/val split lists. Splits by VIDEO, not by individual frame,
to avoid leaking near-duplicate frames between train and val.

STRATIFICATION (new): a real evaluation run found that a naive random
video-level split put 100% of val "bottle" instances as bleach_cleanser
(class 12) while training was dominated by mustard_bottle (class 5) --
the SAME word covering two visually distinct real objects, split almost
entirely onto opposite sides by chance. This wasn't a lack of real
variation in the underlying data (both sub-types exist in what's already
downloaded) -- it was the random split failing to represent that
variation on both sides. STRATIFY_CLASS_PAIRS below lists known cases
that need explicit handling: at least one video containing each listed
class is forced onto BOTH sides of the split, overriding pure randomness
for just those pinned videos.

Usage:
    python3 build_splits.py --dataset_root /home/nyan/pnsg_superquadrics/ycb_dataset --val_fraction 0.15
"""
import argparse
import os
import glob
import random
import sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ycbv_training.ycb_dataset_loader import resolve_video_dir

DATA_SUBDIRS = ['data', 'data2', 'data3']

# known cases where one vocabulary word covers multiple visually distinct
# YCB sub-objects -- each needs at least one representative video forced
# onto BOTH sides of the split, not left to chance
STRATIFY_CLASS_PAIRS = {
    'bottle': [5, 12],   # 006_mustard_bottle, 021_bleach_cleanser
}


def _looks_like_video_id(name):
    return len(name) == 4 and name.isdigit()


def discover_videos(dataset_root):
    videos = set()
    for subdir in DATA_SUBDIRS:
        for base in [os.path.join(dataset_root, subdir),
                     os.path.join(dataset_root, subdir, 'data')]:
            if not os.path.isdir(base):
                continue
            videos.update(d for d in os.listdir(base)
                          if _looks_like_video_id(d) and os.path.isdir(os.path.join(base, d)))
    return sorted(videos)


def discover_frames(dataset_root, video_id):
    video_dir = resolve_video_dir(dataset_root, video_id)
    pattern = os.path.join(video_dir, '*-meta.mat')
    files = glob.glob(pattern)
    frame_ids = sorted(os.path.basename(f).replace('-meta.mat', '') for f in files)
    return frame_ids


def find_videos_containing_class(dataset_root, class_id, videos, sample_frames=5):
    """Checks a handful of frames per video (not all -- expensive) for
    the presence of a given YCB class ID."""
    matching = []
    for v in videos:
        frames = discover_frames(dataset_root, v)
        if not frames:
            continue
        for fid in frames[:sample_frames]:
            try:
                video_dir = resolve_video_dir(dataset_root, v)
                meta = loadmat(os.path.join(video_dir, fid + '-meta.mat'))
                if class_id in np.atleast_1d(meta['cls_indexes'].squeeze()):
                    matching.append(v)
                    break
            except Exception:
                continue
    return matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--val_fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frame_stride', type=int, default=5)
    ap.add_argument('--no_stratify', action='store_true',
                     help='Disable stratification, revert to pure random video split')
    args = ap.parse_args()

    videos = discover_videos(args.dataset_root)
    print(f'Found {len(videos)} video folders: {videos}')

    rng = random.Random(args.seed)
    shuffled = videos[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_videos = set(shuffled[:n_val])
    train_videos = set(shuffled[n_val:])

    if not args.no_stratify:
        print('\nStratifying known multi-subtype categories...')
        for word, class_ids in STRATIFY_CLASS_PAIRS.items():
            for cid in class_ids:
                matches = find_videos_containing_class(args.dataset_root, cid, videos)
                if not matches:
                    print(f'  WARNING: no videos found containing class {cid} ("{word}") at all')
                    continue
                in_train = [v for v in matches if v in train_videos]
                in_val = [v for v in matches if v in val_videos]
                print(f'  "{word}" class={cid}: found in {len(matches)} video(s) '
                      f'({len(in_train)} train, {len(in_val)} val)')
                if not in_val:
                    # move one matching video from train to val
                    move = matches[0]
                    train_videos.discard(move)
                    val_videos.add(move)
                    print(f'    -> forced video {move} into VAL (was missing entirely)')
                if not in_train:
                    move = matches[-1] if matches[-1] not in val_videos or len(matches) == 1 else matches[0]
                    val_videos.discard(move)
                    train_videos.add(move)
                    print(f'    -> forced video {move} into TRAIN (was missing entirely)')

    train_videos = sorted(train_videos)
    val_videos = sorted(val_videos)
    print(f'\nFinal train videos ({len(train_videos)}): {train_videos}')
    print(f'Final val videos   ({len(val_videos)}): {val_videos}')

    os.makedirs(os.path.join(args.dataset_root, 'image_sets'), exist_ok=True)

    for split_name, split_videos in [('train', train_videos), ('val', val_videos)]:
        lines = []
        for v in split_videos:
            frame_ids = discover_frames(args.dataset_root, v)
            frame_ids = frame_ids[::args.frame_stride]
            lines.extend(f'{v}/{fid}' for fid in frame_ids)
        out_path = os.path.join(args.dataset_root, 'image_sets', f'{split_name}.txt')
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'Wrote {len(lines)} frame keys to {out_path}')


if __name__ == '__main__':
    main()
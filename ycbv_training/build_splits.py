"""
Builds train/val split lists by scanning the actual `data/` folder on
disk, since this download doesn't include the official image_sets/
files. Splits by VIDEO, not by individual frame -- frames within one
video are highly correlated (same objects, near-identical consecutive
poses), so a random frame-level split would leak near-duplicates between
train and val and make the held-out evaluation dishonest. Held-out
videos are genuinely unseen trajectories instead.

Usage:
    python3 build_splits.py --dataset_root /home/nyan/pnsg_superquadrics --val_fraction 0.15
"""
import argparse
import os
import glob
import random

DATA_SUBDIRS = ['data', 'data2', 'data3']


def discover_videos(dataset_root):
    """Scans ALL of data/, data2/, data3/ (whichever exist) and merges
    the video ID lists -- the dataset is intentionally kept split across
    its original three download archives, not merged into one folder."""
    videos = set()
    for subdir in DATA_SUBDIRS:
        data_dir = os.path.join(dataset_root, subdir)
        if not os.path.isdir(data_dir):
            continue
        videos.update(d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)))
    return sorted(videos)


def _resolve_video_dir(dataset_root, video_id):
    for subdir in DATA_SUBDIRS:
        candidate = os.path.join(dataset_root, subdir, video_id)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Video '{video_id}' not found in any of {DATA_SUBDIRS}")


def discover_frames(dataset_root, video_id):
    video_dir = _resolve_video_dir(dataset_root, video_id)
    pattern = os.path.join(video_dir, '*-meta.mat')
    files = glob.glob(pattern)
    frame_ids = sorted(os.path.basename(f).replace('-meta.mat', '') for f in files)
    return frame_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--val_fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frame_stride', type=int, default=5,
                     help='Use every Nth frame (consecutive video frames are '
                          'nearly identical -- no need to train/eval on all of them)')
    args = ap.parse_args()

    videos = discover_videos(args.dataset_root)
    print(f'Found {len(videos)} video folders: {videos}')

    rng = random.Random(args.seed)
    shuffled = videos[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_videos = sorted(shuffled[:n_val])
    train_videos = sorted(shuffled[n_val:])

    print(f'Train videos ({len(train_videos)}): {train_videos}')
    print(f'Val videos   ({len(val_videos)}): {val_videos}')

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
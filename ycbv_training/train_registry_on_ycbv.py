"""
Train the registry on real YCB-Video RGB-D data.

Usage:
    python3 train_registry_on_ycbv.py --dataset_root /path/to/YCB_Video_Dataset \\
        --split train --max_frames 2000 --out trained_ycbv_model.json

This is the real version of the "training loop" you asked about --
each object instance in each training frame is one confirmed example
(F=1, since YCB-Video's labels are ground truth), fed through the SAME
registry.confirm() used all night on synthetic data. No gradients, no
epochs in the neural-net sense -- each example is one closed-form
statistical update (Welford mean/variance, or a mode spawn), exactly as
validated in test_registry_session.py, just now fed real sensor data
instead of self-generated synthetic clouds.
"""
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry import Registry
from superquadric import fit_superquadric
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True, help='Path to extracted YCB_Video_Dataset')
    ap.add_argument('--split', default='train', help='Split file name in image_sets/ (without .txt)')
    ap.add_argument('--max_frames', type=int, default=None, help='Cap frames processed (omit for all)')
    ap.add_argument('--checkpoint_every', type=int, default=200, help='Save registry every N frames')
    ap.add_argument('--out', default='trained_ycbv_model.json')
    ap.add_argument('--max_nfev', type=int, default=1500, help='Fitting iterations per object (speed/precision tradeoff)')
    args = ap.parse_args()

    frame_keys = read_split_file(args.dataset_root, args.split)
    if args.max_frames:
        frame_keys = frame_keys[:args.max_frames]

    print(f'Training on {len(frame_keys)} frames from {args.dataset_root} (split={args.split})')

    reg = Registry()
    n_examples = 0
    n_frames_processed = 0
    per_class_counts = {}
    t0 = time.time()

    for i, frame_key in enumerate(frame_keys):
        try:
            for vocab_word, cloud in iter_frame_objects(args.dataset_root, frame_key, class_id_to_vocab):
                fitted, info = fit_superquadric(
                    cloud, max_nfev=args.max_nfev,
                    max_size_multiplier=4.0, min_size_multiplier=0.05,
                    position_margin_multiplier=2.0,
                )
                entry = reg.confirm(fitted, vocab_word, F=1)
                n_examples += 1
                per_class_counts[vocab_word] = per_class_counts.get(vocab_word, 0) + 1
        except Exception as e:
            print(f'  [frame {frame_key}] skipped due to error: {e}')
            continue

        n_frames_processed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  frame {i+1}/{len(frame_keys)}  examples_so_far={n_examples}  '
                  f'elapsed={elapsed:.0f}s  per_class={per_class_counts}')

        if (i + 1) % args.checkpoint_every == 0:
            reg.save(args.out)
            print(f'  checkpoint saved to {args.out}')

    reg.save(args.out)
    print(f'\nDone. Processed {n_frames_processed} frames, {n_examples} object examples.')
    print(f'Final per-class counts: {per_class_counts}')
    for noun in reg.modes:
        print(reg.describe(noun))
    print(f'\nSaved final model to {args.out}')


if __name__ == '__main__':
    main()

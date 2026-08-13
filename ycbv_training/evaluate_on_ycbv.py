"""
Evaluate a trained registry on held-out real YCB-Video frames.

Usage:
    python3 evaluate_on_ycbv.py --dataset_root /path/to/YCB_Video_Dataset \\
        --split val --model trained_ycbv_model.json --max_frames 500

Reports per-class and overall top-1 accuracy, confusion pairs, and mean
confidence for correct vs incorrect predictions -- the real-data version
of the train_val_harness.py metrics from earlier tonight, now on actual
sensor data instead of synthetic clouds.
"""
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry import Registry
from superquadric import fit_superquadric
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--model', required=True)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--max_nfev', type=int, default=1500)
    args = ap.parse_args()

    reg = Registry.load(args.model)
    print(f'Loaded model: {list(reg.modes.keys())}')

    frame_keys = read_split_file(args.dataset_root, args.split)
    if args.max_frames:
        frame_keys = frame_keys[:args.max_frames]
    print(f'Evaluating on {len(frame_keys)} held-out frames (split={args.split})')

    n_total = 0
    n_correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(int)   # (true, predicted) -> count
    confident_correct = []
    confident_wrong = []

    for i, frame_key in enumerate(frame_keys):
        try:
            for true_word, cloud in iter_frame_objects(args.dataset_root, frame_key, class_id_to_vocab):
                fitted, info = fit_superquadric(
                    cloud, max_nfev=args.max_nfev,
                    max_size_multiplier=4.0, min_size_multiplier=0.05,
                    position_margin_multiplier=2.0,
                )
                ranked = reg.classify(fitted, top_k=1)
                pred_word, confidence = ranked[0] if ranked else (None, 0.0)

                n_total += 1
                per_class_total[true_word] += 1
                correct = (pred_word == true_word)
                if correct:
                    n_correct += 1
                    per_class_correct[true_word] += 1
                    confident_correct.append(confidence)
                else:
                    confusion[(true_word, pred_word)] += 1
                    confident_wrong.append(confidence)
        except Exception as e:
            print(f'  [frame {frame_key}] skipped due to error: {e}')
            continue

        if (i + 1) % 50 == 0:
            print(f'  frame {i+1}/{len(frame_keys)}  running_accuracy='
                  f'{n_correct/max(n_total,1)*100:.1f}%  n={n_total}')

    print(f'\n=== Results ===')
    print(f'Overall top-1 accuracy: {n_correct}/{n_total} = {100*n_correct/max(n_total,1):.1f}%')

    print(f'\nPer-class accuracy:')
    for cls in sorted(per_class_total):
        acc = per_class_correct[cls] / per_class_total[cls]
        print(f'  {cls:12s}: {per_class_correct[cls]:4d}/{per_class_total[cls]:4d} = {acc*100:.1f}%')

    if confusion:
        print(f'\nTop confusions (true -> predicted):')
        for (true_w, pred_w), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f'  {true_w} -> {pred_w}: {count} times')

    if confident_correct:
        print(f'\nMean confidence, correct predictions:   {sum(confident_correct)/len(confident_correct):.4f}')
    if confident_wrong:
        print(f'Mean confidence, incorrect predictions: {sum(confident_wrong)/len(confident_wrong):.4f}')


if __name__ == '__main__':
    main()

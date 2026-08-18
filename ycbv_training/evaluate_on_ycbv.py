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
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from registry import Registry
from superquadric import fit_superquadric
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects, load_frame
from ycbv_training.ycb_classes import class_id_to_vocab


def draw_frame_visualization(dataset_root, frame_key, reg, max_nfev, out_dir):
    """Loads the frame's REAL color image, runs each real object instance
    through the trained registry, and draws the fitted 3D box (derived
    from REAL depth, not a stub) projected back onto the actual photo --
    the visualization that was missing from the original script."""
    color_path = os.path.join(dataset_root, 'data', frame_key + '-color.jpg')
    image = cv2.imread(color_path)
    if image is None:
        return None

    depth_m, label, intrinsic, cls_indexes = load_frame(dataset_root, frame_key)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx_img, cy_img = intrinsic[0, 2], intrinsic[1, 2]

    for true_word, cloud in iter_frame_objects(dataset_root, frame_key, class_id_to_vocab):
        fitted, info = fit_superquadric(
            cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
            min_size_multiplier=0.05, position_margin_multiplier=2.0)
        ranked = reg.classify(fitted, top_k=1)
        pred_word, confidence = ranked[0] if ranked else ('?', 0.0)

        # project the fitted 3D box's extent back into image pixels using
        # the REAL camera intrinsics for this frame -- real geometry, real
        # depth, unlike the RGB-bridge's stubbed version
        from scipy.spatial.transform import Rotation as R
        cxp, cyp, czp = fitted['cx'], fitted['cy'], fitted['cz']
        ex, ey, ez = fitted['a1'], fitted['a2'], fitted['a3']
        rot = R.from_euler('zyx', [fitted['yaw'], fitted['pitch'], fitted['roll']]).as_matrix()
        local_corners = np.array([[sx*ex, sy*ey, sz*ez]
                                   for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
        world_corners = local_corners @ rot.T + np.array([cxp, cyp, czp])
        Z = np.clip(world_corners[:, 2], 0.05, None)
        px = world_corners[:, 0] * fx / Z + cx_img
        py = world_corners[:, 1] * fy / Z + cy_img
        x_min, x_max = int(px.min()), int(px.max())
        y_min, y_max = int(py.min()), int(py.max())

        correct = (pred_word == true_word)
        color = (0, 200, 0) if correct else (0, 0, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        text = f'{pred_word} {confidence*100:.0f}% (true:{true_word})'
        cv2.putText(image, text, (x_min, max(y_min - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, frame_key.replace('/', '_') + '.png')
    cv2.imwrite(out_path, image)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--model', required=True)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--save_images', type=int, default=10,
                     help='Save this many annotated example frames (0 to disable)')
    ap.add_argument('--image_out_dir', default='eval_images')
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

    iterator = tqdm(list(enumerate(frame_keys)), desc='Evaluating', unit='frame') \
        if HAVE_TQDM else enumerate(frame_keys)

    for i, frame_key in iterator:
        if args.save_images and i < args.save_images:
            saved_path = draw_frame_visualization(args.dataset_root, frame_key, reg,
                                                    args.max_nfev, args.image_out_dir)
            if saved_path:
                msg = f'  saved visualization: {saved_path}'
                iterator.write(msg) if HAVE_TQDM else print(msg)
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
            msg = f'  [frame {frame_key}] skipped due to error: {e}'
            iterator.write(msg) if HAVE_TQDM else print(msg)
            continue

        if HAVE_TQDM:
            acc = n_correct / max(n_total, 1) * 100
            iterator.set_postfix(accuracy=f'{acc:.1f}%', n=n_total, refresh=False)
        elif (i + 1) % 50 == 0:
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
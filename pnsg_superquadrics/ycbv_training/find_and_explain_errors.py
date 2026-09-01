"""
Finds real wrong predictions on held-out YCB-Video data and explains each
one mechanistically using the registry's own learned statistics.

Usage:
    python3 -m ycbv_training.find_and_explain_errors \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --model trained_ycbv_multiview_v2.json \
        --split val --n_examples 5
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry import Registry
from iterative_segment import iterative_two_part_segment
from pipeline import build_graph_from_segmentation
from superquadric import is_physically_plausible
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab
from ycbv_training.explain_errors import explain_prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_examples', type=int, default=5,
                     help='Stop after explaining this many wrong predictions')
    ap.add_argument('--max_frames', type=int, default=100,
                     help='How many frames to search through looking for errors')
    args = ap.parse_args()

    reg = Registry.load(args.model)
    frames = read_split_file(args.dataset_root, args.split)[:args.max_frames]

    shown = 0
    for fk in frames:
        if shown >= args.n_examples:
            break
        for true_word, cloud in iter_frame_objects(args.dataset_root, fk, class_id_to_vocab):
            if shown >= args.n_examples:
                break

            params_a, params_b, assignment = iterative_two_part_segment(cloud, verbose=False)

            if not is_physically_plausible(params_a):
                continue  # degenerate fitting failure, not a real classification case -- skip
            if params_b is not None and not is_physically_plausible(params_b):
                params_b, assignment = None, None

            graph = build_graph_from_segmentation(cloud, params_a, params_b, assignment)
            ranked = reg.classify_graph(graph, top_k=1)
            pred_word = ranked[0][0] if ranked else None

            if pred_word == true_word:
                continue  # only explain actual mistakes

            shown += 1
            print(f'\n{"="*70}')
            print(f'WRONG PREDICTION #{shown}  (frame {fk})')
            print(f'{"="*70}')
            print(explain_prediction(graph, reg, true_word, pred_word))

    if shown == 0:
        print(f'No wrong predictions found in the first {args.max_frames} frames -- '
              f'try increasing --max_frames.')
    else:
        print(f'\n\nShowed {shown} real wrong predictions with full mechanistic explanations.')


if __name__ == '__main__':
    main()
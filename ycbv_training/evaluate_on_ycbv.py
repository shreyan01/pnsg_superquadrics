"""
Evaluation with color features and axisymmetric fitting -- must use the
SAME fitting rules as training (axisymmetric for can/bottle, color
always extracted) or the comparison is apples-to-oranges.

Usage:
    python3 -m ycbv_training.evaluate_on_ycbv \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split val --model trained_ycbv_color.json --workers 30
"""
import os
# MUST be set before numpy/scipy are imported anywhere -- these libraries
# read these env vars once at import time to decide how many internal
# BLAS threads to spawn PER PROCESS. Without this, each of the 30 worker
# processes independently tries to multithread its own linear algebra
# (inside every scipy.optimize.least_squares call), stacking on top of
# the already-parallel 30 processes -- found via a real load average of
# 935 on a 30-process run, confirmed via `ps -eLf` showing far more OS
# threads than processes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from registry import Registry
from superquadric import fit_superquadric, is_physically_plausible
from radius_profile import compute_radial_profile
from iterative_segment import iterative_two_part_segment
from pipeline import build_graph_from_segmentation
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab
from ycbv_training import metrics as metrics_module

AXISYMMETRIC_WORDS = {'can'}   # DELIBERATE, matches train_registry_multiview.py's reasoning:
                                # can uses axisymmetric+profile fitting; bottle uses the
                                # flexible multi-part segmenter. See that file's comment
                                # for the full reasoning.

_worker_registry = None


def _init_worker(model_path):
    global _worker_registry
    # OpenCV has its OWN internal thread pool, separate from BLAS, and
    # does NOT reliably respect the OMP/OPENBLAS env vars above -- this
    # is the second, previously-missed source of the same oversubscription
    # problem, since this codebase calls cv2.imread constantly (every
    # depth/label/color file) inside every one of the 30 worker processes.
    import cv2
    cv2.setNumThreads(1)
    _worker_registry = Registry.load(model_path)


def _eval_one_frame(args_tuple):
    dataset_root, frame_key, max_nfev, scoring = args_tuple
    results = []
    try:
        for true_word, cloud, color_features in iter_frame_objects(
                dataset_root, frame_key, class_id_to_vocab):
            axisym = true_word in AXISYMMETRIC_WORDS
            if axisym:
                # SUPERSEDES the discrete neck detector -- see
                # train_registry_multiview.py's comment for the real
                # stability evidence behind this continuous approach
                params_a, info = fit_superquadric(cloud, max_nfev=max_nfev, max_size_multiplier=4.0,
                                                   min_size_multiplier=0.05, position_margin_multiplier=2.0,
                                                   axisymmetric=True)
                if not is_physically_plausible(params_a):
                    results.append((true_word, '__IMPLAUSIBLE__', 0.0, {}))
                    continue
                taper_features = compute_radial_profile(cloud, params_a)
                graph = build_graph_from_segmentation(cloud, params_a, None, None,
                                                       color_features=color_features, taper_features=taper_features)
            else:
                params_a, params_b, assignment = iterative_two_part_segment(
                    cloud, verbose=False, max_nfev=max_nfev)
                if not is_physically_plausible(params_a):
                    results.append((true_word, '__IMPLAUSIBLE__', 0.0, {}))
                    continue
                if params_b is not None and not is_physically_plausible(params_b):
                    params_b, assignment = None, None
                graph = build_graph_from_segmentation(cloud, params_a, params_b, assignment,
                                                       color_features=color_features)

            if scoring == 'ml':
                classify_fn = _worker_registry.classify_graph_ml
            elif scoring == 'ensembled':
                classify_fn = _worker_registry.classify_graph_ensembled
            else:
                classify_fn = _worker_registry.classify_graph
            ranked_full = classify_fn(graph, top_k=len(_worker_registry.graph_modes))
            ranked = ranked_full[:1]
            score_dict = dict(ranked_full)
            pred_word, confidence = ranked[0] if ranked else (None, 0.0)
            results.append((true_word, pred_word, confidence, score_dict))
        return frame_key, results, None
    except Exception as e:
        return frame_key, [], str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--model', required=True)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--scoring', choices=['joint', 'ensembled', 'ml'], default='joint',
                     help='joint = current 7D combined scoring; ensembled = EXPERIMENTAL '
                          'geometry-primary/color-bonus scoring; ml = ExtraTreesClassifier '
                          'trained on raw dominant-part exemplars (see registry.py header comment '
                          'for real accuracy numbers behind this option). Requires a model file '
                          'that was TRAINED after raw_examples was added to Mode -- older files '
                          'will silently fall back to ensembled scoring (see classify_graph_ml).')
    args = ap.parse_args()

    reg = Registry.load(args.model)
    print(f'Loaded model: {list(reg.graph_modes.keys())}')
    if args.scoring == 'ml' and getattr(reg, '_ml_classifier', None) is None:
        print('WARNING: --scoring ml requested but this model has no ML classifier '
              '(no raw_examples found, or sklearn missing). SILENTLY FALLING BACK to '
              'ensembled scoring per-call -- results below are NOT the ml classifier. '
              'Retrain with the current registry.py to populate raw_examples first.')

    frame_keys = read_split_file(args.dataset_root, args.split)
    if args.max_frames:
        frame_keys = frame_keys[:args.max_frames]
    print(f'Evaluating on {len(frame_keys)} frames (split={args.split})')

    n_total, n_correct, n_implausible = 0, 0, 0
    per_class_total, per_class_correct = defaultdict(int), defaultdict(int)
    confusion = defaultdict(int)
    confident_correct, confident_wrong = [], []
    raw_results = []
    full_score_results = []   # (true_word, {class: score, ...}) -- needed for honest AUC

    work_items = [(args.dataset_root, fk, args.max_nfev, args.scoring) for fk in frame_keys]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                              initargs=(args.model,)) as executor:
        futures = [executor.submit(_eval_one_frame, item) for item in work_items]
        completed = as_completed(futures)
        iterator = tqdm(completed, total=len(futures), desc='Evaluating', unit='frame') if HAVE_TQDM else completed

        for future in iterator:
            frame_key, results, error = future.result()
            if error:
                msg = f'  [frame {frame_key}] skipped: {error}'
                (iterator.write(msg) if HAVE_TQDM else print(msg))
                continue
            for true_word, pred_word, confidence, score_dict in results:
                if pred_word == '__IMPLAUSIBLE__':
                    n_implausible += 1
                    continue
                raw_results.append((true_word, pred_word, confidence))
                full_score_results.append((true_word, score_dict))
                n_total += 1
                per_class_total[true_word] += 1
                if pred_word == true_word:
                    n_correct += 1
                    per_class_correct[true_word] += 1
                    confident_correct.append(confidence)
                else:
                    confusion[(true_word, pred_word)] += 1
                    confident_wrong.append(confidence)
            if HAVE_TQDM:
                acc = n_correct / max(n_total, 1) * 100
                iterator.set_postfix(accuracy=f'{acc:.1f}%', n=n_total, refresh=False)

    print(f'\n=== Results ===')
    print(f'Overall top-1 accuracy: {n_correct}/{n_total} = {100*n_correct/max(n_total,1):.1f}%')
    if n_implausible:
        print(f'(excluded {n_implausible} physically-implausible fits)')

    print(f'\nPer-class accuracy:')
    for cls in sorted(per_class_total):
        acc = per_class_correct[cls] / per_class_total[cls]
        print(f'  {cls:12s}: {per_class_correct[cls]:4d}/{per_class_total[cls]:4d} = {acc*100:.1f}%')

    if confusion:
        print(f'\nTop confusions (true -> predicted):')
        for (t, p), c in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f'  {t} -> {p}: {c} times')

    if confident_correct:
        print(f'\nMean confidence, correct: {sum(confident_correct)/len(confident_correct):.4f}')
    if confident_wrong:
        print(f'Mean confidence, incorrect: {sum(confident_wrong)/len(confident_wrong):.4f}')

    classes = sorted(reg.graph_modes.keys())
    metrics_module.print_full_report(raw_results, classes)

    calibrated_results = [(t, p, metrics_module.calibrated_confidence(c)) for t, p, c in raw_results]
    print(f'\n{"="*70}')
    print('CALIBRATED CONFIDENCE (display-only, power=0.5 / sqrt)')
    print('Provably cannot change any prediction -- monotonic transform, verified')
    print('argmax-preserving across 10,000 random test cases. Purely corrects the')
    print('readability of raw scores, which compound toward small values.')
    print('='*70)
    cc_correct = [c for t,p,c in calibrated_results if t==p]
    cc_wrong = [c for t,p,c in calibrated_results if t!=p]
    if cc_correct:
        print(f'Mean calibrated confidence, correct:   {sum(cc_correct)/len(cc_correct):.4f}')
    if cc_wrong:
        print(f'Mean calibrated confidence, incorrect: {sum(cc_wrong)/len(cc_wrong):.4f}')
    cal_ece, cal_bins = metrics_module.expected_calibration_error(calibrated_results)
    print(f'Calibrated ECE: {cal_ece:.4f} (was {metrics_module.expected_calibration_error(raw_results)[0]:.4f} raw)')

    print(f'\n{"="*70}')
    print('ROC-AUC (one-vs-rest per class, macro-averaged)')
    print('NOTE: unlike accuracy, this is known to look misleadingly good under')
    print('class imbalance -- your P/R/F1/mAP above are the more standard,')
    print('honest choice for this data. AUC is reported here for completeness.')
    print('='*70)
    macro_auc, per_class_auc = metrics_module.macro_roc_auc(full_score_results, classes)
    for c, auc in per_class_auc.items():
        print(f'  {c:12s}  AUC={auc:.3f}')
    if macro_auc is not None:
        print(f'macro AUC: {macro_auc:.3f}')


if __name__ == '__main__':
    main()
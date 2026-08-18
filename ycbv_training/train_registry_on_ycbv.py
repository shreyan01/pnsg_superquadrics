"""
Train the registry on real YCB-Video RGB-D data -- parallelized version.

Usage:
    python3 -m ycbv_training.train_registry_on_ycbv \\
        --dataset_root . --split train --max_frames 2000 \\
        --out trained_ycbv_model.json --workers 8

DESIGN NOTE ON PARALLELISM: fitting each object is completely
independent (no shared state), so that part is parallelized across a
process pool -- this is the actual CPU-bound bottleneck (scipy's
least_squares, one call per object). registry.confirm() is NOT
parallelized: mode spawning depends on the ORDER examples arrive in
(each decision depends on the registry's current state, built from every
prior example), so running confirm() out of order or concurrently would
produce different, non-deterministic results from the single-process
version. The pool computes fits in parallel; the main process consumes
them in original frame order and updates the registry sequentially,
exactly as before -- same final model, just faster to get there.

No GPU involved: scipy's optimizer has no GPU backend, and this
workload (many small independent optimizations) isn't the batched-
matmul shape GPUs accelerate. This parallelizes across CPU cores
instead, which is the correct lever for this specific bottleneck.
"""
import argparse
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from registry import Registry
from superquadric import fit_superquadric
from ycbv_training.ycb_dataset_loader import read_split_file, iter_frame_objects
from ycbv_training.ycb_classes import class_id_to_vocab


def fit_frame_worker(args_tuple):
    """Runs in a worker process: loads one frame, extracts + fits every
    mapped object instance. Returns plain (picklable) data only -- no
    Registry object crosses the process boundary, since it shouldn't be
    shared/mutated across processes anyway."""
    dataset_root, frame_key, max_nfev = args_tuple
    results = []
    try:
        for vocab_word, cloud in iter_frame_objects(dataset_root, frame_key, class_id_to_vocab):
            fitted, info = fit_superquadric(
                cloud, max_nfev=max_nfev,
                max_size_multiplier=4.0, min_size_multiplier=0.05,
                position_margin_multiplier=2.0,
            )
            results.append((vocab_word, fitted, info['rmse']))
        return frame_key, results, None
    except Exception as e:
        return frame_key, [], str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--checkpoint_every', type=int, default=200)
    ap.add_argument('--out', default='trained_ycbv_model.json')
    ap.add_argument('--max_nfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=os.cpu_count(),
                     help='CPU processes for parallel fitting (default: all cores)')
    args = ap.parse_args()

    frame_keys = read_split_file(args.dataset_root, args.split)
    if args.max_frames:
        frame_keys = frame_keys[:args.max_frames]

    print(f'Training on {len(frame_keys)} frames from {args.dataset_root} '
          f'(split={args.split}, workers={args.workers})')

    reg = Registry()
    n_examples = 0
    n_frames_processed = 0
    per_class_counts = {}
    t0 = time.time()

    work_items = [(args.dataset_root, fk, args.max_nfev) for fk in frame_keys]

    pbar = tqdm(total=len(frame_keys), desc='Training', unit='frame') if HAVE_TQDM else None

    # submit all work, but consume results IN ORDER (executor.map preserves
    # input order regardless of which worker finishes first) -- this is
    # what keeps registry updates deterministic and identical to the
    # single-process version
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (frame_key, results, error) in enumerate(executor.map(fit_frame_worker, work_items)):
            if error:
                msg = f'  [frame {frame_key}] skipped due to error: {error}'
                pbar.write(msg) if HAVE_TQDM else print(msg)
                if HAVE_TQDM:
                    pbar.update(1)
                continue

            for vocab_word, fitted, rmse in results:
                entry = reg.confirm(fitted, vocab_word, F=1)
                n_examples += 1
                per_class_counts[vocab_word] = per_class_counts.get(vocab_word, 0) + 1

            n_frames_processed += 1

            if HAVE_TQDM:
                mode_counts = {n: len(m) for n, m in reg.modes.items()}
                pbar.set_postfix(examples=n_examples, modes=mode_counts, refresh=False)
                pbar.update(1)
            elif (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f'  frame {i+1}/{len(frame_keys)}  examples_so_far={n_examples}  '
                      f'elapsed={elapsed:.0f}s  per_class={per_class_counts}')

            if (i + 1) % args.checkpoint_every == 0:
                reg.save(args.out)
                msg = f'  checkpoint saved to {args.out}'
                pbar.write(msg) if HAVE_TQDM else print(msg)

    if HAVE_TQDM:
        pbar.close()

    reg.save(args.out)
    elapsed_total = time.time() - t0
    print(f'\nDone in {elapsed_total:.0f}s. Processed {n_frames_processed} frames, {n_examples} object examples.')
    print(f'Final per-class counts: {per_class_counts}')
    for noun in reg.modes:
        print(reg.describe(noun))
    print(f'\nSaved final model to {args.out}')


if __name__ == '__main__':
    main()
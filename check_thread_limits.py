"""
Confirms the thread-oversubscription fix is actually working, before you
trust it on another multi-hour run.

What happened: worker processes were each internally multithreading their
own linear algebra (inside every scipy.optimize.least_squares call) on
TOP of the already-parallel process pool -- e.g. 30 worker processes each
spawning ~30 BLAS threads = up to 900 threads competing for real CPU
cores. That's consistent with everything you saw: 14 hours for 4 folds
that should take ~75 minutes total, and eventually the whole machine
becoming unresponsive under sustained, severe thread-count thrashing.

This script checks two things in under 10 seconds:
  1. Are the environment variables actually set, in THIS process?
  2. When real worker processes run real fitting work, does the OS
     thread count stay sane (roughly = number of workers), or does it
     explode?

Usage:
    python3 check_thread_limits.py --workers 8
"""
import argparse
import os


def check_env_vars():
    import registry
    print("Step 0: which registry.py actually got imported")
    print("-" * 60)
    print(f"  Path: {registry.__file__}")
    has_fix_marker = hasattr(registry, 'RESERVOIR_CAP')  # only exists in this session's patched version
    print(f"  Looks like the patched version (has RESERVOIR_CAP)? {has_fix_marker}")
    if not has_fix_marker:
        print("  ^ THIS IS LIKELY THE ACTUAL PROBLEM. Python found a DIFFERENT")
        print("  registry.py than the one that was patched -- an old copy still")
        print("  sitting somewhere earlier on sys.path, a stale copy in a")
        print("  different directory, or even an unrelated package also named")
        print("  'registry' (check `pip show registry`). Fix: run this script")
        print("  from the exact directory containing the PATCHED registry.py,")
        print("  or check for duplicate copies with:")
        print("      find ~ -name 'registry.py' 2>/dev/null")
        print()

    names = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
    print("Step 1: environment variables in this process")
    print("-" * 60)
    all_good = True
    for name in names:
        value = os.environ.get(name)
        ok = value == '1'
        all_good &= ok
        print(f"  {name:22s} = {value!r:8s}  {'OK' if ok else 'MISSING OR WRONG -- should be 1'}")
    if not all_good and has_fix_marker:
        print("\n  Odd: this IS the patched registry.py, but the env vars still aren't")
        print("  set. Possible cause: something else imported numpy BEFORE this")
        print("  script even started registry.py's import (e.g. a sitecustomize.py,")
        print("  a shell profile pre-importing something, or an IDE/notebook kernel")
        print("  that already had numpy loaded in this same Python process).")
    return all_good


def _worker_job(i):
    # Real work: this is genuinely the same kind of call that was
    # causing the problem -- a nonlinear least-squares fit, not a
    # toy computation.
    import numpy as np
    from scipy.optimize import least_squares

    def resid(params):
        x = np.linspace(0, 1, 200)
        return params[0] * np.exp(-params[1] * x) - np.sin(x * (i + 1))

    for _ in range(20):  # repeat to give the thread-count check time to observe it
        least_squares(resid, x0=[1.0, 1.0], max_nfev=200)
    return os.getpid()


def check_worker_thread_counts(n_workers):
    print(f"\nStep 2: real worker processes doing real fitting work ({n_workers} workers)")
    print("-" * 60)
    import time
    from concurrent.futures import ProcessPoolExecutor
    import psutil

    proc = psutil.Process(os.getpid())
    max_threads_seen = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker_job, i) for i in range(n_workers)]
        child_pids = set()
        while not all(f.done() for f in futures):
            for child in proc.children(recursive=True):
                child_pids.add(child.pid)
                try:
                    max_threads_seen = max(max_threads_seen, child.num_threads())
                except psutil.NoSuchProcess:
                    pass
            time.sleep(0.2)
        for f in futures:
            f.result()

    print(f"  Worker processes spawned: {n_workers}")
    print(f"  Max OS threads seen in any single worker process: {max_threads_seen}")

    if max_threads_seen <= 4:
        print(f"  OK -- each worker stayed near 1 thread, as expected for the fix working.")
        return True
    else:
        print(f"  WARNING -- {max_threads_seen} threads in one worker process is too many "
              f"for CPU-bound scipy work. The fix may not be taking effect (e.g. a different "
              f"numpy/BLAS backend that ignores these particular env vars). Worth "
              f"investigating further before trusting a long run.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    env_ok = check_env_vars()

    try:
        import psutil  # noqa: F401
    except ImportError:
        print("\npsutil not installed -- skipping step 2 (pip install psutil --break-system-packages "
              "if you want the full check). Step 1's result above is still meaningful on its own.")
        return

    workers_ok = check_worker_thread_counts(args.workers)

    print("\n" + "=" * 60)
    if env_ok and workers_ok:
        print("PASS -- looks safe to run a real long job now.")
    else:
        print("FAIL -- do not start another multi-hour run until this is resolved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
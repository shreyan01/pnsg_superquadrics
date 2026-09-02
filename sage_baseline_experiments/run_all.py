"""
Run the whole baseline-experiment pipeline in order.

Usage
-----
    python run_all.py                 # everything
    python run_all.py 1 5             # only tasks 1 and 5
    python run_all.py --list          # show what would run

Order matters. Task 4 reloads the fold models Task 2 trains,
and Task 5 reads the instance-level predictions Tasks 1 and 2
write, so running a later task alone against stale results
will either fail loudly or quietly report yesterday's numbers.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# (key, script, description, approximate runtime)
STEPS = [
    (
        "0",
        "inspect_data.py",
        "Data inspection and sanity checks",
        "seconds",
    ),
    (
        "1",
        "task1_feature_baselines.py",
        "Same-feature baselines (k-NN, SVM)",
        "seconds",
    ),
    (
        "2",
        "task2_pointnet.py",
        "Raw point-cloud PointNet baseline (3 CV regimes)",
        "~36 min on CPU",
    ),
    (
        "3",
        "task3_sample_efficiency.py",
        "Sample-efficiency curve (baselines; add --dataset_root for SAGE)",
        "~1 min",
    ),
    (
        "4",
        "task4_robustness.py",
        "Robustness to downsampling and noise",
        "~5 min",
    ),
    (
        "5",
        "task5_statistics.py",
        "Bootstrap CIs and significance tests",
        "~1 min",
    ),
]


def main():

    arguments = sys.argv[1:]

    if "--list" in arguments:

        print("Available steps:\n")

        for key, script, description, runtime in STEPS:
            print(
                f"  {key}  {script:32s} "
                f"{description}  ({runtime})"
            )

        return 0

    selected = [
        step
        for step in STEPS
        if not arguments or step[0] in arguments
    ]

    if not selected:
        print(
            f"No steps matched {arguments}. "
            f"Try --list."
        )
        return 1

    failures = []

    for key, script, description, _ in selected:

        print("\n" + "=" * 70)
        print(f"STEP {key}: {description}")
        print(f"  {script}")
        print("=" * 70, flush=True)

        started = time.time()

        completed = subprocess.run(
            [sys.executable, str(SRC_DIR / script)],
            cwd=SRC_DIR,
        )

        elapsed = time.time() - started

        if completed.returncode != 0:

            print(
                f"\nStep {key} FAILED "
                f"(exit {completed.returncode}) "
                f"after {elapsed:.1f}s"
            )

            failures.append(key)

            # Later steps consume earlier outputs, so
            # continuing past a failure would report
            # results built on stale files.
            break

        print(
            f"\nStep {key} finished in {elapsed:.1f}s"
        )

    print("\n" + "=" * 70)

    if failures:
        print(f"PIPELINE FAILED at step(s): {failures}")
        return 1

    print("PIPELINE COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

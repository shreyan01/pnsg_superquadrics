# Notebooks

Summary and plots for the five tasks in `co-author_guide.md`. Start with
`00_overview.ipynb`.

| Notebook | Guide task | What it answers |
|---|---|---|
| `00_overview.ipynb` | all | Headline comparison, the leakage correction, the four findings |
| `01_task1_same_feature_baselines.ipynb` | Task 1 | Is the scoring formula the bottleneck, or the 13D feature set? |
| `02_task2_pointnet_raw_points.ipynb` | Task 2 | What does the interpretable representation cost? |
| `03_task3_sample_efficiency.ipynb` | Task 3 | Accuracy vs examples per category (baseline half) |
| `04_task4_robustness.ipynb` | Task 4 | Graceful degradation, or a cliff? |
| `05_task5_statistics.ipynb` | Task 5 | Confidence intervals, and is the can/bottle gap real? |

## Running them

Select the **`Python (experiment)`** kernel. It is the project's own
`experiment/` virtualenv, registered as a Jupyter kernel:

```bash
experiment/Scripts/python.exe -m pip install ipykernel nbformat nbclient
experiment/Scripts/python.exe -m ipykernel install --user \
    --name sage-experiment --display-name "Python (experiment)"
```

Then, from the project root:

```bash
experiment/Scripts/python.exe -m jupyter lab notebooks/
```

The notebooks are committed **with outputs already embedded**, so they can be
read without running anything.

## What they do and do not do

They **summarise and plot saved results**. Every number is read from the CSVs
under `results/`, via `src/results_io.py` — so a figure here can never disagree
with the script that produced it.

They do **not** retrain. If `results/` is missing or stale, regenerate it first:

```bash
experiment/Scripts/python.exe run_all.py
```

`results_io` raises a `FileNotFoundError` naming that command if a file it
needs is absent, rather than failing obscurely.

## Two shared modules

Both live in `src/` alongside the task scripts:

- **`results_io.py`** — loads the saved results, plus SAGE's reported numbers
  and the class supports. Keeps path handling out of the notebooks.
- **`viz_style.py`** — the palette and matplotlib defaults, so every figure
  reads as one system.

### Chart conventions

- **Colour follows the entity.** SAGE keeps one identity in every figure it
  appears in; the two CV regimes keep theirs (blue = object-grouped, orange =
  random). A filter or a re-sort never repaints a series.
- **Three categorical hues, never cycled.** Where a fourth series genuinely
  shares an axis, it takes a neutral that *means* something — PointNet is a
  different model family (raw points) from the three feature-based models —
  rather than a fourth invented hue.
- **Sequential encoding is one hue, light to dark.** Confusion matrices use a
  single blue ramp, never a rainbow.
- **Every chart ships with its table.** The aqua slot sits marginally under
  3:1 contrast on this surface, so the underlying DataFrame is always shown
  alongside — nothing is gated behind colour perception.
- **Narrow ranges say so.** Where an axis does not start at zero (the Task 4
  degradation curves), the text says so directly under the figure.

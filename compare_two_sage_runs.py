"""
Real, paired significance test between two SAGE evaluation runs -- e.g.
the original scoring rule (78.4%) vs the new ML classifier (89.4%) --
using McNemar's test on the instances BOTH runs actually produced a
valid (non-implausible) fit for, matched by instance_id.

Why this needs instance_id, not just two accuracy numbers: McNemar's
test needs to know, PER INSTANCE, whether each model got it right or
wrong -- "92 correct in run A, 105 correct in run B" alone can't tell
you whether the improvement is real or just moved which instances are
correct around. And why not just row order: two separate
ProcessPoolExecutor runs are not guaranteed to complete frames in the
same order, so pairing by row position would silently mismatch
instances between runs -- instance_id (frame_key + object index) is
required for a valid pairing.

Usage:
    python3 -m ycbv_training.evaluate_on_ycbv \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split val_sample --model <OLD_MODEL>.json --scoring joint \
        --predictions_out old_predictions.csv

    python3 -m ycbv_training.evaluate_on_ycbv \
        --dataset_root ~/pnsg_superquadrics/ycb_dataset \
        --split val_sample --model <NEW_MODEL>.json --scoring ml \
        --predictions_out new_predictions.csv

    python3 compare_two_sage_runs.py --a old_predictions.csv --b new_predictions.csv \
        --a_name "original (joint scoring)" --b_name "ExtraTrees (ml scoring)"
"""
import argparse
import numpy as np
import pandas as pd


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - margin, center + margin


def mcnemar_test(a_correct, b_correct):
    """a_correct, b_correct: aligned boolean arrays, same instances, same
    order. Returns (statistic, p_value) using the standard chi-square
    approximation with continuity correction (matches what Task 5 uses
    elsewhere in this project, via statsmodels)."""
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = sm_mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue, (both_correct, a_only, b_only, both_wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='First predictions CSV (needs instance_id column)')
    ap.add_argument('--b', required=True, help='Second predictions CSV (needs instance_id column)')
    ap.add_argument('--a_name', default='A')
    ap.add_argument('--b_name', default='B')
    args = ap.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    for name, df, path in [(args.a_name, df_a, args.a), (args.b_name, df_b, args.b)]:
        if 'instance_id' not in df.columns:
            raise ValueError(
                f"{path} has no instance_id column -- it was produced before "
                f"this fix, or by a different script. Re-run "
                f"evaluate_on_ycbv.py with the current version to get one; "
                f"pairing by row order would silently give a wrong answer."
            )

    print(f'{args.a_name}: {len(df_a)} instances')
    print(f'{args.b_name}: {len(df_b)} instances')

    merged = df_a.merge(df_b, on='instance_id', suffixes=('_a', '_b'), how='inner')
    n_common = len(merged)
    print(f'Instances present in BOTH runs (inner join on instance_id): {n_common}')
    if n_common < len(df_a) or n_common < len(df_b):
        print(f'  ({len(df_a) - n_common} only in {args.a_name}, '
              f'{len(df_b) - n_common} only in {args.b_name} -- likely differing '
              f'physical-plausibility exclusions between the two models/runs. '
              f'Only the common set can be validly paired.')

    mismatched_labels = merged[merged['true_label_a'] != merged['true_label_b']]
    if len(mismatched_labels) > 0:
        raise ValueError(
            f'{len(mismatched_labels)} instances have DIFFERENT true_label between '
            f'the two runs for the same instance_id -- these should be identical '
            f'(same instance, same ground truth). Something is wrong with how '
            f'these CSVs were produced; do not trust a McNemar test on this data '
            f'until this is resolved.'
        )

    a_correct = (merged['true_label_a'] == merged['predicted_label_a']).to_numpy()
    b_correct = (merged['true_label_b'] == merged['predicted_label_b']).to_numpy()

    a_acc = a_correct.mean()
    b_acc = b_correct.mean()
    a_lo, a_hi = wilson_ci(a_correct.sum(), n_common)
    b_lo, b_hi = wilson_ci(b_correct.sum(), n_common)

    print(f'\n{args.a_name}: {a_correct.sum()}/{n_common} = {a_acc*100:.2f}%  '
          f'95% CI [{a_lo*100:.2f}%, {a_hi*100:.2f}%]')
    print(f'{args.b_name}: {b_correct.sum()}/{n_common} = {b_acc*100:.2f}%  '
          f'95% CI [{b_lo*100:.2f}%, {b_hi*100:.2f}%]')

    stat, pvalue, (both_c, a_only, b_only, both_w) = mcnemar_test(a_correct, b_correct)
    print(f'\nMcNemar (paired, same {n_common} instances):')
    print(f'  both correct: {both_c}   {args.a_name} only: {a_only}   '
          f'{args.b_name} only: {b_only}   both wrong: {both_w}')
    print(f'  statistic={stat:.4f}  p={pvalue:.6g}')
    if pvalue < 0.05:
        winner = args.b_name if b_acc > a_acc else args.a_name
        print(f'  -> Significant at p<0.05: {winner} is really better on these '
              f'{n_common} instances, not just a point-estimate difference.')
    else:
        print(f'  -> NOT significant at p<0.05 -- the accuracy difference could '
              f'be noise on this sample.')

    out_path = 'paired_comparison_result.csv'
    merged[['instance_id', 'true_label_a', 'predicted_label_a', 'predicted_label_b']].rename(
        columns={'true_label_a': 'true_label'}
    ).assign(a_correct=a_correct, b_correct=b_correct).to_csv(out_path, index=False)
    print(f'\nPer-instance paired results saved to {out_path}')


if __name__ == '__main__':
    main()
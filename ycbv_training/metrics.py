import numpy as np

def confusion_matrix(results, classes):
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)
    unknown_pred = 0
    for true_label, pred_label, _ in results:
        if true_label not in idx: continue
        i = idx[true_label]
        if pred_label in idx:
            j = idx[pred_label]; mat[i, j] += 1
        else:
            unknown_pred += 1
    return mat, classes, unknown_pred

def precision_recall_f1(results, classes):
    per_class = {}
    tp_total, fp_total, fn_total = 0, 0, 0
    for c in classes:
        tp = sum(1 for t, p, _ in results if t == c and p == c)
        fp = sum(1 for t, p, _ in results if t != c and p == c)
        fn = sum(1 for t, p, _ in results if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        per_class[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp+fn}
        tp_total += tp; fp_total += fp; fn_total += fn
    macro_p = np.mean([v['precision'] for v in per_class.values()])
    macro_r = np.mean([v['recall'] for v in per_class.values()])
    macro_f1 = np.mean([v['f1'] for v in per_class.values()])
    micro_p = tp_total/(tp_total+fp_total) if (tp_total+fp_total)>0 else 0.0
    micro_r = tp_total/(tp_total+fn_total) if (tp_total+fn_total)>0 else 0.0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r)>0 else 0.0
    return {'per_class': per_class, 'macro': {'precision':macro_p,'recall':macro_r,'f1':macro_f1},
            'micro': {'precision':micro_p,'recall':micro_r,'f1':micro_f1}}

def average_precision(results, target_class):
    n_positive = sum(1 for t, p, c in results if t == target_class)
    if n_positive == 0: return None
    predicted_as_target = [(c, t == target_class) for t, p, c in results if p == target_class]
    if not predicted_as_target: return 0.0
    predicted_as_target.sort(key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for conf, is_correct in predicted_as_target:
        if is_correct: tp_cum += 1
        else: fp_cum += 1
        precisions.append(tp_cum/(tp_cum+fp_cum)); recalls.append(tp_cum/n_positive)
    ap = 0.0; prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p*(r-prev_recall); prev_recall = r
    return ap

def mean_average_precision(results, classes):
    aps = {}
    for c in classes:
        ap = average_precision(results, c)
        if ap is not None: aps[c] = ap
    mean_ap = np.mean(list(aps.values())) if aps else 0.0
    return mean_ap, aps

def expected_calibration_error(results, n_bins=10):
    if not results: return 0.0, []
    bin_edges = np.linspace(0, 1, n_bins+1)
    bin_stats = []; ece = 0.0; n_total = len(results)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        in_bin = [(t,p,c) for t,p,c in results if lo<=c<hi or (i==n_bins-1 and c==hi)]
        if not in_bin:
            bin_stats.append({'range':(lo,hi),'n':0,'avg_confidence':None,'accuracy':None}); continue
        avg_conf = np.mean([c for _,_,c in in_bin])
        accuracy = np.mean([1.0 if t==p else 0.0 for t,p,_ in in_bin])
        weight = len(in_bin)/n_total
        ece += weight*abs(avg_conf-accuracy)
        bin_stats.append({'range':(lo,hi),'n':len(in_bin),'avg_confidence':avg_conf,'accuracy':accuracy})
    return ece, bin_stats

def print_full_report(results, classes):
    print('\n' + '='*70); print('STANDARD CLASSIFICATION METRICS'); print('='*70)
    mat, cls_order, unknown = confusion_matrix(results, classes)
    print(f'\nConfusion matrix (rows=true, cols=predicted), {unknown} unknown-predicted excluded:')
    header = '        ' + ''.join(f'{c[:8]:>10s}' for c in cls_order)
    print(header)
    for i, c in enumerate(cls_order):
        row = ''.join(f'{mat[i,j]:>10d}' for j in range(len(cls_order)))
        print(f'{c[:8]:>8s}{row}')
    prf = precision_recall_f1(results, classes)
    print(f'\nPer-class precision / recall / F1 / support:')
    for c, v in prf['per_class'].items():
        print(f'  {c:12s}  P={v["precision"]:.3f}  R={v["recall"]:.3f}  F1={v["f1"]:.3f}  n={v["support"]}')
    print(f'\nMacro-avg:  P={prf["macro"]["precision"]:.3f}  R={prf["macro"]["recall"]:.3f}  F1={prf["macro"]["f1"]:.3f}')
    print(f'Micro-avg:  P={prf["micro"]["precision"]:.3f}  R={prf["micro"]["recall"]:.3f}  F1={prf["micro"]["f1"]:.3f}')
    mean_ap, per_class_ap = mean_average_precision(results, classes)
    print(f'\nPer-class Average Precision (classification-style, NOT detection mAP):')
    for c, ap in per_class_ap.items(): print(f'  {c:12s}  AP={ap:.3f}')
    print(f'mean AP (classification mAP): {mean_ap:.3f}')
    ece, bins = expected_calibration_error(results)
    print(f'\nExpected Calibration Error (ECE): {ece:.4f}  (lower = better calibrated)')
    for b in bins:
        if b['n'] > 0:
            print(f"  conf∈[{b['range'][0]:.1f},{b['range'][1]:.1f})  n={b['n']:3d}  avg_conf={b['avg_confidence']:.3f}  actual_acc={b['accuracy']:.3f}")
    return {'confusion_matrix': mat.tolist(), 'classes': cls_order, 'precision_recall_f1': prf,
            'mean_ap': mean_ap, 'per_class_ap': per_class_ap, 'ece': ece, 'ece_bins': bins}

def roc_auc_one_vs_rest(full_score_results, target_class, all_classes):
    """full_score_results: list of (true_label, {class: score, ...}) --
    needs a score against EVERY class per instance, not just the winner
    (that's why this needs a separate, richer capture pass -- the
    existing top-1-only results can't compute this honestly). Standard
    one-vs-rest ROC-AUC: treat 'is this the target class' as the binary
    label, target_class's own score as the continuous decision value."""
    pairs = [(scores.get(target_class, 0.0), 1 if true == target_class else 0)
             for true, scores in full_score_results]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # AUC undefined without both classes present

    pairs.sort(key=lambda x: -x[0])
    tp, fp = 0, 0
    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0
    for score, y in pairs:
        if y == 1: tp += 1
        else: fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0   # trapezoidal
        tpr_prev, fpr_prev = tpr, fpr
    return auc


def macro_roc_auc(full_score_results, all_classes):
    aucs = {}
    for c in all_classes:
        auc = roc_auc_one_vs_rest(full_score_results, c, all_classes)
        if auc is not None:
            aucs[c] = auc
    macro = np.mean(list(aucs.values())) if aucs else None
    return macro, aucs


def calibrated_confidence(raw_score, power=0.5):
    """DISPLAY-ONLY recalibration -- raises a raw membership score to a
    power < 1, which is monotonic (preserves ordering exactly) and
    therefore CANNOT change which prediction wins argmax, i.e. cannot
    affect accuracy at all. Purely corrects the readability of the
    reported confidence number, which tends toward very small values
    (e.g. 5-7%) as a mechanical consequence of multiplying several
    sub-1 factors together (dominant score x secondary modifier x
    structural penalty x confidence discount) -- power=0.5 (square
    root) is a standard, simple, easily-explained default, not an
    arbitrarily tuned value."""
    return raw_score ** power
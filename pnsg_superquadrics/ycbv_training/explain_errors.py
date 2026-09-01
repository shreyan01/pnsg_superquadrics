"""
Uses the fact that this system IS explainable to actually explain its
own errors, instead of only counting them. For any candidate object and
any two words being compared, shows the exact per-part numeric
breakdown that decided the winner -- which learned mode matched best,
how many standard deviations away each part was, and whether the
losing word lost because of a genuine shape mismatch or a structural
one (missing/extra part). This is the tool the whole "explainable, not
a black box" thesis promises should exist -- so let's actually use it
to find out why accuracy is where it is, not just report the number.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from registry import canonicalize


def explain_word_match(graph, registry, noun):
    """For one candidate structure and one word, returns a detailed,
    human-readable breakdown of the best-matching mode: per-part
    Mahalanobis distance, per-part membership contribution, and
    structural mismatch info. This is exactly what GraphMode.membership()
    computes internally -- this function just makes every step visible
    instead of collapsing it into one number."""
    gms = registry.graph_modes.get(noun, [])
    if not gms:
        return {'noun': noun, 'known': False, 'score': 0.0,
                'detail': f'"{noun}" has no learned structure at all.'}

    candidate_roles = set(n.role for n in graph.nodes)
    node_by_role = {n.role: n for n in graph.nodes}

    best = None
    for gm in gms:
        learned_roles = set(gm.part_modes.keys())
        shared = candidate_roles & learned_roles
        mismatched = candidate_roles ^ learned_roles

        part_details = []
        if shared:
            for role in shared:
                f = canonicalize(node_by_role[role].params)
                mode = gm.part_modes[role]
                dist = mode.mahalanobis(f)
                membership = mode.membership(f)
                part_details.append({
                    'role': role, 'mahalanobis': dist, 'membership': membership,
                    'candidate_shape': f.round(3).tolist(),
                    'learned_mean': mode.mean.round(3).tolist(),
                    'learned_std': mode.std.round(3).tolist(),
                    'n_support': mode.n,
                })
            geo_mean = float(np.prod([p['membership'] for p in part_details]) ** (1.0 / len(part_details)))
        else:
            geo_mean = 0.0

        penalty = 0.4 ** len(mismatched)
        score = geo_mean * penalty

        if best is None or score > best['score']:
            best = {'mode_id': gm.mode_id, 'score': score, 'parts': part_details,
                    'mismatched_roles': list(mismatched), 'penalty': penalty,
                    'mode_n': gm.n}

    return {'noun': noun, 'known': True, **best}


def explain_prediction(graph, registry, true_word, pred_word):
    """Full explanation for one (true, predicted) pair: compares the
    winning word against the correct word directly, part by part, and
    generates a one-line mechanistic summary of why the wrong one won."""
    true_expl = explain_word_match(graph, registry, true_word)
    pred_expl = explain_word_match(graph, registry, pred_word)

    lines = []
    lines.append(f'Candidate structure: {[n.role for n in graph.nodes]} '
                 f'({len(graph.nodes)} part(s) found)')
    for n in graph.nodes:
        lines.append(f'  {n.role}: size=({n.params["a1"]*1000:.0f},{n.params["a2"]*1000:.0f},'
                     f'{n.params["a3"]*1000:.0f})mm eps=({n.params["eps1"]:.2f},{n.params["eps2"]:.2f})')

    lines.append(f'\nTRUE label "{true_word}": score={true_expl["score"]:.4f}')
    if true_expl.get('parts'):
        for p in true_expl['parts']:
            lines.append(f'    {p["role"]}: {p["mahalanobis"]:.2f}sigma from learned mean '
                         f'(n={p["n_support"]} supporting examples), membership={p["membership"]:.4f}')
    if true_expl.get('mismatched_roles'):
        lines.append(f'    structural mismatch: {true_expl["mismatched_roles"]} '
                     f'(penalty x{true_expl["penalty"]:.2f})')

    lines.append(f'\nPREDICTED label "{pred_word}": score={pred_expl["score"]:.4f}')
    if pred_expl.get('parts'):
        for p in pred_expl['parts']:
            lines.append(f'    {p["role"]}: {p["mahalanobis"]:.2f}sigma from learned mean '
                         f'(n={p["n_support"]} supporting examples), membership={p["membership"]:.4f}')
    if pred_expl.get('mismatched_roles'):
        lines.append(f'    structural mismatch: {pred_expl["mismatched_roles"]} '
                     f'(penalty x{pred_expl["penalty"]:.2f})')

    if not true_expl.get('parts') and true_expl.get('mismatched_roles'):
        verdict = (f'"{true_word}" lost primarily due to STRUCTURAL mismatch '
                  f'(candidate parts {[n.role for n in graph.nodes]} do not overlap with '
                  f'what "{true_word}" learned to expect).')
    elif true_expl.get('parts'):
        worst_part = max(true_expl['parts'], key=lambda p: p['mahalanobis'])
        undertrained = ' -- likely undertrained' if worst_part['n_support'] < 5 else ''
        verdict = (f'"{true_word}" lost primarily due to SHAPE mismatch on its '
                  f'"{worst_part["role"]}" part ({worst_part["mahalanobis"]:.2f} sigma off, '
                  f'only n={worst_part["n_support"]} supporting examples{undertrained}).')
    else:
        verdict = f'"{true_word}" has no learned structure to compare against at all.'

    lines.append(f'\nVERDICT: {verdict}')
    return '\n'.join(lines)
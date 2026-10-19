import numpy as np
from superquadric import fit_superquadric
from segmentation import per_point_residual, segment_and_fit

def iterative_two_part_segment(raw_cloud, init_threshold=0.10, cluster_radius=0.012,
                                 min_cluster_size=40, max_iters=8, verbose=True, max_nfev=3000,
                                 axisymmetric=False, return_diagnostics=False):
    """axisymmetric=True applies true-circularity fitting to BOTH parts
    throughout segmentation and refinement -- lets a body+neck bottle
    naturally discover its real two-segment structure instead of being
    forced into one constant-radius primitive.

    return_diagnostics=True appends a 4th return value: a dict with the
    fit-quality signals this function already computes internally.

    Why that matters: combined_rmse was computed on every refinement
    round and used to pick best_state, then thrown away at every return.
    Four of the five categories (box, mug, bowl, bottle) go through this
    path rather than the axisymmetric one, so NO residual information
    reached the caller for them at all -- leaving the pipeline with
    is_physically_plausible() (a bare semi-axis size check) as its only
    fit-quality gate. The diagnostics are needed to tell a reliable fit
    from an unreliable one.

    Kept optional so the existing 3-tuple contract is unchanged; every
    current caller keeps working untouched.

    diagnostics keys:
        combined_rmse   sum of the two parts' rmse at the chosen state
                        (np.inf if refinement never improved on init)
        rmse_a, rmse_b  per-part rmse, or None where not available
        n_parts         1 or 2
        n_iters         refinement rounds actually run
        converged       whether the assignment stopped changing
        stale_exit      whether it stopped because rmse stopped improving
    """
    parts = segment_and_fit(raw_cloud, residual_threshold=init_threshold, cluster_radius=cluster_radius,
                             min_cluster_size=min_cluster_size, verbose=False, max_nfev=max_nfev,
                             axisymmetric=axisymmetric)
    if len(parts) < 2:
        if verbose: print('No second part found even at initialization.')
        if return_diagnostics:
            # segment_and_fit carries per-part 'info' (segmentation.py:43),
            # so the single-part case has a real rmse to report.
            single_info = parts[0].get('info') or {}
            diagnostics = {'combined_rmse': single_info.get('rmse', np.inf),
                            'rmse_a': single_info.get('rmse'), 'rmse_b': None,
                            'n_parts': 1, 'n_iters': 0,
                            'converged': True, 'stale_exit': False}
            return parts[0]['params'], None, None, diagnostics
        return parts[0]['params'], None, None
    params_a = parts[0]['params']; params_b = parts[1]['params']
    assignment = np.zeros(len(raw_cloud), dtype=bool)
    assignment[parts[0]['point_indices']] = True
    best_state = {'params_a': params_a, 'params_b': params_b, 'assignment': assignment.copy(),
                   'combined_rmse': np.inf, 'rmse_a': None, 'rmse_b': None}
    stale_rounds = 0
    n_iters = 0
    converged = False

    def _result(stale_exit):
        if not return_diagnostics:
            return best_state['params_a'], best_state['params_b'], best_state['assignment']
        diagnostics = {'combined_rmse': best_state['combined_rmse'],
                        'rmse_a': best_state['rmse_a'], 'rmse_b': best_state['rmse_b'],
                        'n_parts': 2, 'n_iters': n_iters,
                        'converged': converged, 'stale_exit': stale_exit}
        return best_state['params_a'], best_state['params_b'], best_state['assignment'], diagnostics

    for it in range(max_iters):
        n_iters = it + 1
        resid_a = per_point_residual(raw_cloud, params_a)
        resid_b = per_point_residual(raw_cloud, params_b)
        new_assignment = resid_a <= resid_b
        idx_a = np.where(new_assignment)[0]; idx_b = np.where(~new_assignment)[0]
        if len(idx_a) < 8 or len(idx_b) < 8: break
        params_a, info_a = fit_superquadric(raw_cloud[idx_a], init=params_a, max_size_multiplier=4.0,
                                             min_size_multiplier=0.05, position_margin_multiplier=1.5,
                                             max_nfev=max_nfev, axisymmetric=axisymmetric)
        params_b, info_b = fit_superquadric(raw_cloud[idx_b], init=params_b, max_size_multiplier=4.0,
                                             min_size_multiplier=0.05, position_margin_multiplier=1.5,
                                             max_nfev=max_nfev, axisymmetric=axisymmetric)
        combined_rmse = info_a['rmse'] + info_b['rmse']
        if combined_rmse < best_state['combined_rmse']:
            best_state = {'params_a': params_a, 'params_b': params_b, 'assignment': new_assignment.copy(),
                           'combined_rmse': combined_rmse,
                           'rmse_a': info_a['rmse'], 'rmse_b': info_b['rmse']}
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= 2:
                return _result(stale_exit=True)
        converged = np.array_equal(new_assignment, assignment)
        assignment = new_assignment
        if converged: break
    return _result(stale_exit=False)
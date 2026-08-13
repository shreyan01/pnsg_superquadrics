"""Iterative reassignment refinement for two-part segmentation: alternates
reassigning points to whichever current fit explains them better, and
refitting -- with a divergence guard that reverts to the best-seen state."""
import numpy as np
from superquadric import fit_superquadric
from segmentation import per_point_residual, segment_and_fit


def iterative_two_part_segment(raw_cloud, init_threshold=0.10, cluster_radius=0.012,
                                 min_cluster_size=40, max_iters=8, verbose=True,
                                 max_nfev=3000):
    parts = segment_and_fit(raw_cloud, residual_threshold=init_threshold,
                             cluster_radius=cluster_radius,
                             min_cluster_size=min_cluster_size, verbose=False, max_nfev=max_nfev)
    if len(parts) < 2:
        if verbose:
            print('No second part found even at initialization.')
        return parts[0]['params'], None, None

    params_a = parts[0]['params']
    params_b = parts[1]['params']
    assignment = np.zeros(len(raw_cloud), dtype=bool)
    assignment[parts[0]['point_indices']] = True

    best_state = {'params_a': params_a, 'params_b': params_b,
                   'assignment': assignment.copy(), 'combined_rmse': np.inf}
    stale_rounds = 0

    for it in range(max_iters):
        resid_a = per_point_residual(raw_cloud, params_a)
        resid_b = per_point_residual(raw_cloud, params_b)
        new_assignment = resid_a <= resid_b
        idx_a = np.where(new_assignment)[0]
        idx_b = np.where(~new_assignment)[0]

        if len(idx_a) < 8 or len(idx_b) < 8:
            break

        params_a, info_a = fit_superquadric(raw_cloud[idx_a], init=params_a,
                                             max_size_multiplier=4.0, min_size_multiplier=0.05,
                                             position_margin_multiplier=1.5, max_nfev=max_nfev)
        params_b, info_b = fit_superquadric(raw_cloud[idx_b], init=params_b,
                                             max_size_multiplier=4.0, min_size_multiplier=0.05,
                                             position_margin_multiplier=1.5, max_nfev=max_nfev)

        combined_rmse = info_a['rmse'] + info_b['rmse']
        if verbose:
            changed = int(np.sum(new_assignment != assignment))
            print(f'  iter {it}: n_a={len(idx_a)} n_b={len(idx_b)} reassigned={changed} '
                  f'combined_rmse={combined_rmse:.5f}')

        if combined_rmse < best_state['combined_rmse']:
            best_state = {'params_a': params_a, 'params_b': params_b,
                           'assignment': new_assignment.copy(), 'combined_rmse': combined_rmse}
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= 2:
                return best_state['params_a'], best_state['params_b'], best_state['assignment']

        converged = np.array_equal(new_assignment, assignment)
        assignment = new_assignment
        if converged:
            break

    return best_state['params_a'], best_state['params_b'], best_state['assignment']

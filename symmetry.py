"""
Label-free decision of whether an object should be fitted axisymmetrically.

Why this exists
---------------
The fitting strategy used to be chosen from the ground-truth category:

    axisym = true_word in AXISYMMETRIC_WORDS      # {'can'}

in export_baseline_data.py, evaluate_on_ycbv.py and
train_registry_multiview.py. Because `axisymmetric=True` fixes a2 = a1
and eps2 = 1.0 (superquadric.py:76-88), and only 'can' took that path,
the fitted vector carried a signature of which path was used: on
features_val_sample.npz, `a1 == a2` held for 100% of cans and 0% of
everything else, so a single bit identified 'can' with 100% accuracy.

The physical prior itself is sound -- a can really is a solid of
revolution, and a single view only ever sees part of its circumference,
so an unconstrained fit will happily call that arc an ellipse. The
defect was consulting the answer key to decide when to apply it.

This module decides it from the observation instead. Two signals, both
computed without labels:

  angular_radius_variation
      Direct geometry. Around the object's principal axis, a solid of
      revolution has a radius that depends only on height, not on
      angle. Measured per height band so a tapered object (bottle neck,
      mug) is not penalised for changing radius with height.

  residual_ratio
      Fit both ways and compare normalised residuals: what does
      constraining a2 = a1 cost on this cloud?

MEASURED FINDING -- there is deliberately no selection rule here
----------------------------------------------------------------
The obvious use of `residual_ratio` is as a selector: impose the
constraint when it is cheap. On real single-view clouds that is
**backwards**, so this module does not provide it.

Measured on 100 exported clouds (20 per class, max_nfev=600), median
residual_ratio by class:

    can     3.271      <- the genuinely round class
    box     2.394
    bottle  1.479
    mug     1.185
    bowl    1.013      <- the least round

Constraining a2 = a1 costs the MOST on cans, not the least. The reason
is the one the axisymmetric docstring already gives: a single view sees
only part of a round object's circumference, and an unconstrained fit
hugs that partial arc closely with an ellipse, while a circular fit has
to compromise. So a low residual_ratio means "this cloud is easy to fit
either way", not "this object is round".

Discriminative power for can-vs-rest (AUROC):
    residual_ratio      0.841   (informative, but in the inverse sense)
    angular_variation   0.603   (near useless on partial views)

So `residual_ratio` is worth carrying as a FEATURE -- the classifier can
learn its true sign from data -- but it must not be used to pick the
fitting path. The physical prior genuinely cannot be recovered from a
partial arc, which is precisely why it was hardcoded in the first place.

What replaces the label-conditioned branch is therefore not a smarter
branch but no branch at all: `fit_both()` runs both paths for every
object and exposes both. Since every object then gets both fits, a1 == a2
holds for 100% of objects in the axisymmetric fit and 0% in the flexible
one -- constant across classes, carrying no label information. The leak
closes by construction rather than by a rule that could be wrong.
"""

import numpy as np

from superquadric import fit_superquadric, is_physically_plausible


def _principal_axis(centred):
    """
    The object's dominant axis, as the smallest-variance direction.

    For an upright bottle/can/mug the points spread most in the two
    radial directions taken together, and a solid of revolution is
    symmetric about the axis with the *most* extent for tall objects and
    the least for flat ones. We take the third singular vector, which is
    the surface normal direction for a flat object and the symmetry axis
    for a tall one, then correct by choosing whichever of the three axes
    minimises angular radius variation.
    """

    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)

    return vh


def angular_radius_variation(cloud, n_height_bands=6, n_angle_bins=8,
                              min_points_per_cell=3):
    """
    How much the radius varies with angle, at fixed height.

    Returns a value in [0, inf) where 0 means perfectly axisymmetric.
    Computed for each of the three principal axes; the smallest is
    returned, since we do not know a priori which axis is the symmetry
    axis and a wrong choice makes a round object look irregular.

    Returns np.nan when the cloud is too sparse or too narrowly sampled
    to measure -- common for heavily-occluded single views, and a signal
    in itself that this measure should not be trusted for that instance.
    """

    cloud = np.asarray(cloud, dtype=np.float64)

    if len(cloud) < 30:
        return np.nan

    centred = cloud - cloud.mean(axis=0)

    axes = _principal_axis(centred)

    scores = []

    for axis_index in range(3):

        axis = axes[axis_index]

        # Build an orthonormal frame with `axis` as z.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])

        x_dir = np.cross(axis, helper)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(axis, x_dir)

        height = centred @ axis
        radius = np.hypot(centred @ x_dir, centred @ y_dir)
        angle = np.arctan2(centred @ y_dir, centred @ x_dir)

        if radius.max() <= 0:
            continue

        # Height bands, so a tapered profile is not read as asymmetry.
        band_edges = np.quantile(
            height, np.linspace(0, 1, n_height_bands + 1)
        )

        band_scores = []

        for lo, hi in zip(band_edges[:-1], band_edges[1:]):

            in_band = (height >= lo) & (height <= hi)

            if in_band.sum() < n_angle_bins * min_points_per_cell:
                continue

            band_radius = radius[in_band]
            band_angle = angle[in_band]

            bins = np.floor(
                (band_angle + np.pi) / (2 * np.pi) * n_angle_bins
            ).astype(int)
            bins = np.clip(bins, 0, n_angle_bins - 1)

            means = [
                band_radius[bins == b].mean()
                for b in range(n_angle_bins)
                if (bins == b).sum() >= min_points_per_cell
            ]

            # Need coverage on at least a few distinct angles, or
            # "variation across angle" is not a meaningful quantity.
            if len(means) < 3:
                continue

            means = np.asarray(means)

            if means.mean() <= 0:
                continue

            # Coefficient of variation of the per-angle mean radius.
            band_scores.append(means.std() / means.mean())

        if band_scores:
            scores.append(float(np.mean(band_scores)))

    if not scores:
        return np.nan

    return min(scores)


def normalised_rmse(params, info):
    """
    Fit residual made comparable across objects of different size.

    info['rmse'] is volume-scaled (superquadric.py:28 returns
    (F**(e1/2) - 1) * sqrt(a1*a2*a3)), so a big object gets a bigger
    rmse for the same relative misfit. Dividing by the geometric mean
    semi-axis removes that dependence.
    """

    scale = (
        params["a1"] * params["a2"] * params["a3"]
    ) ** (1.0 / 3.0)

    if not np.isfinite(scale) or scale <= 0:
        return np.nan

    return float(info["rmse"]) / scale


def fit_both(cloud, max_nfev=1500, **fit_kwargs):
    """
    Fit the cloud both ways and report which the data supports.

    Returns a dict with:
        axisym / axisym_info / axisym_nrmse
        flexible / flexible_info / flexible_nrmse
        residual_ratio            axisym_nrmse / flexible_nrmse
        angular_variation         geometric symmetry measure
        axisym_plausible / flexible_plausible

    Nothing here consults a label. Callers choose using
    `looks_axisymmetric()` or by passing the features straight through.
    """

    cloud = np.asarray(cloud, dtype=np.float64)

    defaults = dict(
        max_size_multiplier=4.0,
        min_size_multiplier=0.05,
        position_margin_multiplier=2.0,
    )
    defaults.update(fit_kwargs)

    axisym_params, axisym_info = fit_superquadric(
        cloud, max_nfev=max_nfev, axisymmetric=True, **defaults
    )

    flexible_params, flexible_info = fit_superquadric(
        cloud, max_nfev=max_nfev, axisymmetric=False, **defaults
    )

    axisym_nrmse = normalised_rmse(axisym_params, axisym_info)
    flexible_nrmse = normalised_rmse(flexible_params, flexible_info)

    if (
        flexible_nrmse is None
        or not np.isfinite(flexible_nrmse)
        or flexible_nrmse <= 0
    ):
        ratio = np.nan
    else:
        ratio = axisym_nrmse / flexible_nrmse

    return {
        "axisym": axisym_params,
        "axisym_info": axisym_info,
        "axisym_nrmse": axisym_nrmse,
        "axisym_plausible": bool(is_physically_plausible(axisym_params)),
        "flexible": flexible_params,
        "flexible_info": flexible_info,
        "flexible_nrmse": flexible_nrmse,
        "flexible_plausible": bool(
            is_physically_plausible(flexible_params)
        ),
        "residual_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "angular_variation": angular_radius_variation(cloud),
    }


def looks_axisymmetric(diagnostics, threshold=None):
    """
    Deliberately not implemented -- see the module docstring.

    A residual-ratio threshold was built and measured, and it selects
    the wrong way round on partial single-view clouds (cans have the
    HIGHEST cost for the constraint, not the lowest; AUROC 0.841 in the
    inverse sense). Shipping it would reintroduce the same defect it was
    meant to remove, only harder to spot.

    Use `fit_both()` and pass both fits through as features instead.
    """

    raise NotImplementedError(
        "No label-free selection rule is provided: the residual-ratio "
        "criterion was measured and selects inversely on partial "
        "single-view clouds (see module docstring). Call fit_both() and "
        "carry both fits as features -- that closes the leak by "
        "construction, with no rule that can be wrong."
    )

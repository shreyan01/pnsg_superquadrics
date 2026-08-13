"""
Object categories seeded with a RANGE of plausible parameters per
category, informed by inspecting real reference photos (image search +
vision) rather than one single guessed shape. Still synthetic point
clouds (no real depth sensor data), but now spanning realistic proportion
variation per category instead of one fixed example repeated.

Proportions (height/diameter ratios) are grounded in what's visible
across several real photos per category. Absolute scale is a general-
knowledge estimate, honestly flagged as such.
"""

# each category: list of (a1_range, a3_range) describing diameter/2 and
# height/2 variation observed across real reference examples, plus a
# fixed shape family (eps1, eps2) consistent with that category's form
REAL_WORLD_CATEGORIES = {
    'mug': {
        'a1_range': (0.035, 0.045),   # radius: 7-9cm diameter
        'a3_range': (0.040, 0.055),   # half-height: 8-11cm tall
        'eps1': 0.28, 'eps2': 1.0,    # flattish caps, round cross-section
        'n_variants': 3,
    },
    'bowl': {
        'a1_range': (0.050, 0.070),   # radius: 10-14cm diameter
        'a3_range': (0.020, 0.030),   # half-height: 4-6cm tall -- shallow
        'eps1': 0.5, 'eps2': 1.0,
        'n_variants': 3,
    },
    'candle_jar': {
        'a1_range': (0.028, 0.038),   # radius: 5.5-7.5cm diameter
        'a3_range': (0.030, 0.045),   # half-height: 6-9cm tall
        'eps1': 0.15, 'eps2': 1.0,    # more cylindrical, flat top/bottom
        'n_variants': 3,
    },
}


def make_real_world_cloud(category, variant_idx, seed):
    """Generate one synthetic point cloud whose PROPORTIONS are grounded
    in the real reference photos for this category, sampling within the
    observed range rather than using one fixed value."""
    import numpy as np
    from superquadric import sample_superquadric_surface, add_noise

    spec = REAL_WORLD_CATEGORIES[category]
    rng = np.random.default_rng(seed)
    # spread variants evenly across the observed range instead of random,
    # so small/medium/large examples are all represented
    n = spec['n_variants']
    frac = variant_idx / max(n - 1, 1)
    a1 = spec['a1_range'][0] + frac * (spec['a1_range'][1] - spec['a1_range'][0])
    a3 = spec['a3_range'][0] + frac * (spec['a3_range'][1] - spec['a3_range'][0])

    params = {'a1': a1, 'a2': a1, 'a3': a3, 'eps1': spec['eps1'], 'eps2': spec['eps2'],
              'cx': 0, 'cy': 0, 'cz': a3, 'roll': 0, 'pitch': 0, 'yaw': 0}
    pts = sample_superquadric_surface(params, n_points=2500, rng=rng)
    return add_noise(pts, sigma=0.0015, rng=rng)


def teach_registry_real_world(registry):
    """Bootstraps the registry using the visually-grounded category
    ranges instead of the old single-point synthetic examples."""
    from iterative_segment import iterative_two_part_segment
    from pipeline import build_graph_from_segmentation

    for category, spec in REAL_WORLD_CATEGORIES.items():
        for v in range(spec['n_variants']):
            raw = make_real_world_cloud(category, v, seed=hash((category, v)) % 100000)
            params_a, params_b, assignment = iterative_two_part_segment(raw, verbose=False)
            graph = build_graph_from_segmentation(raw, params_a, params_b, assignment)
            registry.confirm_graph(graph, category, F=1)

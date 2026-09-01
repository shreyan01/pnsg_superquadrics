import numpy as np
from superquadric import sample_superquadric_surface, add_noise
from iterative_segment import iterative_two_part_segment
from pipeline import build_graph_from_segmentation
from object_zoo import CATEGORIES


def make_object_cloud(category_key, offset, seed):
    cat = CATEGORIES[category_key]
    rng = np.random.default_rng(seed)
    clouds = []
    for part_params, n in zip(cat['parts'], cat['point_counts']):
        pts = sample_superquadric_surface(part_params, n_points=n, rng=rng)
        pts = pts + np.array(offset)
        clouds.append(pts)
    raw = np.vstack(clouds) if len(clouds) > 1 else clouds[0]
    raw = add_noise(raw, sigma=0.0015, rng=rng)
    return raw


def teach_registry(registry, category_key, n_examples=5, seed_base=0):
    label = CATEGORIES[category_key].get('label', category_key)
    for i in range(n_examples):
        raw = make_object_cloud(category_key, offset=[0, 0, 0], seed=seed_base + i)
        params_a, params_b, assignment = iterative_two_part_segment(raw, verbose=False)
        graph = build_graph_from_segmentation(raw, params_a, params_b, assignment)
        registry.confirm_graph(graph, label, F=1)

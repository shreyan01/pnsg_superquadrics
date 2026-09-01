"""
Run the full RGB bridge on the actual uploaded photo. Every stage is
labeled REAL or STUBBED per rgb_bridge.py's docstring. The registry is
taught a small vocabulary first (mug, candle, bowl -- plausible for this
scene) using our existing, already-validated synthetic pipeline, then
classify_graph() is used to score each proposed region.
"""
import numpy as np
import cv2
from rgb_bridge import (propose_regions_2d, estimate_depth_with_prior,
                          lift_region_to_pointcloud, reproject_bbox_to_image)
from iterative_segment import iterative_two_part_segment
from pipeline import build_graph_from_segmentation
from registry import Registry
from real_world_priors import teach_registry_real_world

IMG_PATH = '/mnt/user-data/uploads/close-up-coffee-cup-plate-nuts-candle-pink-tulips-home-windowsill-cozy-feminine-scene-symbolizing-wellness-self-447064935.webp'


def main():
    image = cv2.imread(IMG_PATH)
    if image is None:
        print(f'Could not load image at {IMG_PATH}')
        return
    print(f'Loaded image: {image.shape[1]}x{image.shape[0]} px')

    # teach a small vocabulary using our EXISTING, already-validated
    # synthetic pipeline (this part is fully real, unchanged from earlier)
    reg = Registry()
    print('\nTeaching registry with real-photo-grounded priors: mug, bowl, candle_jar...')
    teach_registry_real_world(reg)
    for noun, gms in reg.graph_modes.items():
        print(f'  {noun}: {len(gms)} mode(s), from {sum(gm.n for gm in gms)} examples')

    # Stage 1: REAL classical-CV region proposals
    regions = propose_regions_2d(image, min_area=2000, max_regions=10)
    print(f'\nStage 1 (real, classical CV): found {len(regions)} candidate region(s)')

    output_img = image.copy()
    results = []

    for i, region in enumerate(regions):
        x, y, w, h = region['bbox_px']

        # Stage 2: STUBBED depth, now with a category-appropriate size
        # prior instead of one global guess (see rgb_bridge.py docstring)
        depth_m, guessed_cat = estimate_depth_with_prior(image.shape, region)

        # Stage 3: REAL geometry, operating on stubbed depth input
        cloud = lift_region_to_pointcloud(region, depth_m, image.shape, n_samples=600)
        if len(cloud) < 30:
            continue

        # Stage 4 (reuse): our EXISTING, already-validated 3D pipeline,
        # completely unchanged -- this is the same segmentation/graph/
        # registry code tested all night, just fed a (stub-derived) cloud
        # instead of a synthetic one
        params_a, params_b, assignment = iterative_two_part_segment(
            cloud, verbose=False, max_iters=4, max_nfev=800,
            min_cluster_size=15)
        graph = build_graph_from_segmentation(cloud, params_a, params_b, assignment)
        ranked = reg.classify_graph(graph, top_k=1)
        label, confidence = ranked[0] if ranked else ('?', 0.0)

        results.append({'bbox_px': (x, y, w, h), 'label': label, 'confidence': confidence})

        color = (0, 200, 0) if confidence > 0.3 else (0, 140, 255)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
        text = f'{label} {confidence*100:.0f}%'
        cv2.putText(output_img, text, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f'  region {i}: bbox_px=({x},{y},{w},{h})  2d_preclass={guessed_cat}  stub_depth={depth_m:.2f}m  '
              f'n_pts={len(cloud)}  -> label={label!r} confidence={confidence:.3f}')

    cv2.putText(output_img, 'STUB DEPTH -- classical CV proposals -- not a trained detector',
                (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    out_path = 'rgb_bridge_output.png'
    cv2.imwrite(out_path, output_img)
    print(f'\nSaved {out_path}')
    print(f'\n{len(results)} region(s) processed through the real 3D pipeline.')


if __name__ == '__main__':
    main()

"""
Reconstructs and plots the ACTUAL shape each stored mode believes a
word looks like -- pulling real a1/a2/eps1/eps2/a3 straight from the
registry's learned mean, and generating a synthetic point cloud from
them (same generator used throughout tonight's synthetic tests). This
answers directly: has the model actually learned sensible bottle/can
shapes, or garbage?

Plots radius-vs-height (top-down view + side profile), NOT a 3D
scatter -- avoids relying on mpl_toolkits.mplot3d, which is prone to
breaking under conflicting matplotlib installations (system + pip),
and radius-vs-height is arguably the clearer diagnostic anyway for
directly comparing axisymmetric shapes like bottle/can.

Usage:
    python3 visualize_learned_shapes.py trained_ycbv_split.json \
        mustard_bottle bleach_bottle can
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from registry import Registry
from superquadric import sample_superquadric_surface


def mode_to_cloud(mode, n_points=1200):
    """Reconstructs a synthetic point cloud from a stored Mode's
    learned mean shape parameters (a1, a2, eps1, eps2, a3 -- indices
    0-4 of the 10D feature vector). Uses a canonical upright pose since
    orientation was never part of what's learned."""
    a1, a2, eps1, eps2, a3 = mode.mean[0], mode.mean[1], mode.mean[2], mode.mean[3], mode.mean[4]
    params = {'a1': a1, 'a2': a2, 'eps1': eps1, 'eps2': eps2, 'a3': a3,
              'cx': 0.0, 'cy': 0.0, 'cz': a3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    return sample_superquadric_surface(params, n_points=n_points), params


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_learned_shapes.py <model.json> <word1> [word2] ...")
        sys.exit(1)

    model_path = sys.argv[1]
    words = sys.argv[2:]

    reg = Registry.load(model_path)

    all_modes = []
    for word in words:
        gms = reg.graph_modes.get(word, [])
        if not gms:
            print(f'WARNING: no learned modes for "{word}"')
            continue
        for gm in gms:
            if 'dominant' in gm.part_modes:
                all_modes.append((word, gm))

    if not all_modes:
        print('No modes found to plot.')
        sys.exit(1)

    n_cols = len(all_modes)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, (word, gm) in enumerate(all_modes):
        mode = gm.part_modes['dominant']
        cloud, params = mode_to_cloud(mode)

        # top-down view (X vs Y) -- should look like a filled circle/ring
        # for a genuinely round object, an ellipse if not
        ax_top = axes[0, i]
        ax_top.scatter(cloud[:, 0] * 1000, cloud[:, 1] * 1000, s=2, alpha=0.4)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'{word}\n{gm.mode_id} (n={mode.n})\ntop-down view', fontsize=8)
        ax_top.set_xlabel('mm'); ax_top.set_ylabel('mm')

        # side profile: radial distance vs height -- directly shows the
        # taper/body shape, the actual thing we're trying to compare
        radius = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2) * 1000
        height = cloud[:, 2] * 1000
        ax_side = axes[1, i]
        ax_side.scatter(radius, height, s=2, alpha=0.4)
        ax_side.scatter(-radius, height, s=2, alpha=0.4, color='C0')  # mirror for a full silhouette
        ax_side.set_title(f'a1={params["a1"]*1000:.0f}mm a3={params["a3"]*1000:.0f}mm\n'
                          f'eps=({params["eps1"]:.2f},{params["eps2"]:.2f})', fontsize=8)
        ax_side.set_xlabel('radius (mm)'); ax_side.set_ylabel('height (mm)')
        ax_side.set_aspect('equal')

    plt.tight_layout()
    out_path = 'learned_shapes.png'
    plt.savefig(out_path, dpi=100)
    print(f'\nSaved visualization to {out_path} ({len(all_modes)} modes plotted)')
    print('Top row: top-down view (circle=round, ellipse=not).')
    print('Bottom row: side silhouette (radius vs height) -- directly shows taper/body shape.')


if __name__ == '__main__':
    main()
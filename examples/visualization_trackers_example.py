"""Example visualization trackers for RSSM latents and JEPA/ViT embeddings.

Run from the repository root with:

    python examples/visualization_trackers_example.py

The script writes two interactive HTML files in the current directory when
Plotly is installed:

- rssm_latent_trajectories.html
- jepa_embedding_clusters.html
"""

from __future__ import annotations

import importlib.util

import numpy as np

import torchwm


def make_synthetic_rssm_states() -> np.ndarray:
    """Create two smooth latent trajectories with different offsets."""

    rng = np.random.default_rng(0)
    timesteps = np.linspace(0, 2 * np.pi, 24)
    base = np.stack(
        [
            np.sin(timesteps),
            np.cos(timesteps),
            np.sin(2 * timesteps),
            np.cos(2 * timesteps),
        ],
        axis=-1,
    )
    projection_matrix = rng.normal(size=(4, 32))
    first = base @ projection_matrix + rng.normal(scale=0.02, size=(24, 32))
    second = (base + np.array([0.4, -0.2, 0.1, 0.3])) @ projection_matrix
    second += rng.normal(scale=0.02, size=(24, 32))
    return np.stack([first, second], axis=0)


def make_synthetic_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Create three labeled embedding clusters."""

    rng = np.random.default_rng(1)
    centers = np.array(
        [
            [-2.0, 0.0, 0.5],
            [1.5, 1.0, -0.5],
            [0.5, -1.5, 1.0],
        ]
    )
    projection_matrix = rng.normal(size=(3, 64))
    labels = np.repeat(["agent", "object", "background"], 40)
    low_dim = np.repeat(centers, 40, axis=0) + rng.normal(scale=0.25, size=(120, 3))
    embeddings = low_dim @ projection_matrix + rng.normal(scale=0.05, size=(120, 64))
    return embeddings, labels


def main() -> None:
    if importlib.util.find_spec("plotly") is None:
        raise SystemExit("This example requires plotly. Install it with: pip install plotly")

    rssm_states = make_synthetic_rssm_states()
    rssm_projection = torchwm.project_latent_trajectories(
        rssm_states,
        method="pca",
        n_components=2,
    )
    torchwm.plot_projection(
        rssm_projection,
        title="Synthetic RSSM latent trajectories",
        output_path="rssm_latent_trajectories.html",
    )

    embeddings, labels = make_synthetic_embeddings()
    embedding_projection = torchwm.project_representation_embeddings(
        embeddings,
        labels=labels,
        method="pca",
        n_components=2,
    )
    torchwm.plot_projection(
        embedding_projection,
        title="Synthetic JEPA/ViT embedding clusters",
        output_path="jepa_embedding_clusters.html",
    )

    print("Wrote rssm_latent_trajectories.html")
    print("Wrote jepa_embedding_clusters.html")


if __name__ == "__main__":
    main()

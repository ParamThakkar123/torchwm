import importlib.util

import numpy as np
import pytest

from world_models.utils.visualization import (
    plot_projection,
    project_latent_trajectories,
    project_representation_embeddings,
)


def test_project_latent_trajectories_tracks_time_and_sequence_ids():
    states = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)

    result = project_latent_trajectories(states, n_components=2)

    assert result.coordinates.shape == (8, 2)
    assert result.timesteps.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert result.sequence_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result.explained_variance_ratio.shape == (2,)


def test_project_representation_embeddings_validates_label_count():
    embeddings = np.random.default_rng(0).normal(size=(5, 6))

    with pytest.raises(ValueError, match="one value per embedding"):
        project_representation_embeddings(embeddings, labels=["cat", "dog"])


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn is not installed")
def test_project_representation_embeddings_with_tsne_and_labels():
    embeddings = np.random.default_rng(1).normal(size=(6, 4))
    labels = np.array(["a", "a", "b", "b", "c", "c"])

    result = project_representation_embeddings(
        embeddings,
        labels=labels,
        method="tsne",
        perplexity=50,
        random_state=0,
    )

    assert result.coordinates.shape == (6, 2)
    assert result.labels.tolist() == labels.tolist()


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="plotly is not installed")
def test_plot_projection_writes_html(tmp_path):
    result = project_latent_trajectories(np.random.default_rng(2).normal(size=(4, 3)))
    output_path = tmp_path / "projection.html"

    fig = plot_projection(result, output_path=output_path)

    assert fig is not None
    assert output_path.exists()

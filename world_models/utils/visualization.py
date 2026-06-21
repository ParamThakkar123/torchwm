"""Visualization trackers for latent trajectories and learned embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import importlib

ProjectionMethod = Literal["pca", "tsne"]


@dataclass(frozen=True)
class ProjectionResult:
    """Projected coordinates plus aligned plotting metadata."""

    coordinates: np.ndarray
    labels: np.ndarray | None = None
    sequence_ids: np.ndarray | None = None
    timesteps: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None


def _to_numpy(values: Any) -> np.ndarray:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        torch = importlib.import_module("torch")
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
    return np.asarray(values)


def _flatten_features(values: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = _to_numpy(values)
    if array.ndim < 2:
        raise ValueError("Expected at least 2 dimensions: samples/trajectory and features")
    leading_shape = array.shape[:-1]
    return array.reshape(-1, array.shape[-1]), leading_shape


def _project(
    features: np.ndarray,
    *,
    method: ProjectionMethod,
    n_components: int,
    random_state: int,
    perplexity: float,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    if method == "pca":
        if importlib.util.find_spec("sklearn") is not None:
            from sklearn.decomposition import PCA

            model = PCA(n_components=n_components, random_state=random_state, **kwargs)
            coords = model.fit_transform(features)
            return coords, model.explained_variance_ratio_
        centered = features - features.mean(axis=0, keepdims=True)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:n_components].T
        denom = max(features.shape[0] - 1, 1)
        variances = (singular_values**2) / denom
        total = variances.sum()
        ratio = variances[:n_components] / total if total > 0 else np.zeros(n_components)
        return coords, ratio
    if method == "tsne":
        from sklearn.manifold import TSNE

        if features.shape[0] < 2:
            raise ValueError("t-SNE requires at least two samples")
        safe_perplexity = min(float(perplexity), max(1.0, features.shape[0] - 1.0))
        model = TSNE(
            n_components=n_components,
            perplexity=safe_perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            **kwargs,
        )
        return model.fit_transform(features), None
    raise ValueError(f"Unsupported projection method: {method!r}")


def project_latent_trajectories(
    states: Any,
    *,
    method: ProjectionMethod = "pca",
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float = 30.0,
    **kwargs: Any,
) -> ProjectionResult:
    """Project RSSM latent state trajectories for temporal visualization.

    Args:
        states: Array/Tensor shaped ``(time, features)``, ``(batch, time, features)``,
            or any leading trajectory dimensions followed by a feature dimension.
        method: Dimensionality reduction method, ``"pca"`` or ``"tsne"``.
        n_components: Projection dimensionality, either 2 or 3.
        random_state: Seed used by stochastic reducers.
        perplexity: Requested t-SNE perplexity, clipped for small sample counts.
        **kwargs: Additional keyword arguments forwarded to the reducer.
    """

    features, leading_shape = _flatten_features(states)
    coords, variance = _project(
        features,
        method=method,
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        **kwargs,
    )

    if len(leading_shape) == 1:
        timesteps = np.arange(leading_shape[0])
        sequence_ids = np.zeros(leading_shape[0], dtype=int)
    else:
        num_sequences = int(np.prod(leading_shape[:-1]))
        time_steps = leading_shape[-1]
        timesteps = np.tile(np.arange(time_steps), num_sequences)
        sequence_ids = np.repeat(np.arange(num_sequences), time_steps)

    return ProjectionResult(coords, sequence_ids=sequence_ids, timesteps=timesteps, explained_variance_ratio=variance)


def project_representation_embeddings(
    embeddings: Any,
    labels: Sequence[Any] | np.ndarray | Any | None = None,
    *,
    method: ProjectionMethod = "pca",
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float = 30.0,
    **kwargs: Any,
) -> ProjectionResult:
    """Project JEPA/ViT embeddings to 2D or 3D, optionally colored by labels."""

    features, leading_shape = _flatten_features(embeddings)
    coords, variance = _project(
        features,
        method=method,
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        **kwargs,
    )
    label_array = None if labels is None else _to_numpy(labels).reshape(-1)
    if label_array is not None and label_array.shape[0] != features.shape[0]:
        raise ValueError("labels must contain one value per embedding")
    timesteps = np.arange(features.shape[0]) if len(leading_shape) == 1 else None
    return ProjectionResult(coords, labels=label_array, timesteps=timesteps, explained_variance_ratio=variance)


def plot_projection(
    projection: ProjectionResult,
    *,
    title: str = "Projection",
    output_path: str | Path | None = None,
):
    """Render a 2D/3D projection as a Plotly figure and optionally write it to HTML."""

    import plotly.express as px

    coords = projection.coordinates
    if coords.shape[1] == 2:
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=None if projection.labels is None else projection.labels.astype(str),
            symbol=None if projection.sequence_ids is None else projection.sequence_ids.astype(str),
            hover_data={"timestep": projection.timesteps} if projection.timesteps is not None else None,
            title=title,
            labels={"x": "component 1", "y": "component 2", "color": "label", "symbol": "sequence"},
        )
    elif coords.shape[1] == 3:
        fig = px.scatter_3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            color=None if projection.labels is None else projection.labels.astype(str),
            symbol=None if projection.sequence_ids is None else projection.sequence_ids.astype(str),
            hover_data={"timestep": projection.timesteps} if projection.timesteps is not None else None,
            title=title,
        )
    else:
        raise ValueError("Projection coordinates must be 2D or 3D")
    if output_path is not None:
        fig.write_html(str(output_path))
    return fig

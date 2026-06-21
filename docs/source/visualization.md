# Visualization Trackers

TorchWM includes lightweight projection helpers for inspecting world-model latents
and representation embeddings in notebooks, scripts, or experiment dashboards.
They are designed for two common debugging workflows:

- **Latent dynamics trajectories:** project RSSM states to 2D or 3D so you can
  inspect temporal patterns, trajectory separation, resets, and rollout drift.
- **Representation embeddings:** project JEPA or ViT embeddings with PCA or t-SNE
  and color points by class, environment, episode, or any user-provided category.

The helpers are available from the top-level `torchwm` API and from
`world_models.utils.visualization`.

## Install optional plotting dependencies

PCA projections work with NumPy alone and use scikit-learn when it is available.
For t-SNE and interactive Plotly output, install the standard TorchWM package or
ensure `scikit-learn` and `plotly` are present in your environment:

```bash
pip install torchwm
# or, when developing from source:
pip install scikit-learn plotly
```

## RSSM latent dynamics trajectories

Use `project_latent_trajectories` when you have RSSM state vectors shaped like
`(time, features)`, `(batch, time, features)`, or any leading trajectory shape
followed by a feature dimension. The returned `ProjectionResult` contains
projected coordinates plus aligned `timesteps` and `sequence_ids` metadata.

```python
import numpy as np
import torchwm

# Example shape: two imagined rollouts, six timesteps, 32-dimensional RSSM state.
states = np.random.default_rng(0).normal(size=(2, 6, 32))

projection = torchwm.project_latent_trajectories(
    states,
    method="pca",
    n_components=2,
)

fig = torchwm.plot_projection(
    projection,
    title="RSSM latent trajectories",
    output_path="rssm_latent_trajectories.html",
)
fig.show()
```

For 3D plots, set `n_components=3`:

```python
projection_3d = torchwm.project_latent_trajectories(states, n_components=3)
torchwm.plot_projection(projection_3d, title="RSSM trajectories in 3D")
```

## JEPA/ViT representation embeddings

Use `project_representation_embeddings` for learned representations from JEPA,
ViT, or other encoders. Labels are optional, but when provided they must contain
one value per embedding. Labels are passed through to Plotly for coloring.

```python
import numpy as np
import torchwm

# Example shape: 128 images represented by 768-dimensional ViT/JEPA embeddings.
embeddings = np.random.default_rng(1).normal(size=(128, 768))
labels = np.array(["cat"] * 64 + ["dog"] * 64)

projection = torchwm.project_representation_embeddings(
    embeddings,
    labels=labels,
    method="tsne",
    n_components=2,
    perplexity=30,
    random_state=42,
)

fig = torchwm.plot_projection(
    projection,
    title="JEPA embedding clusters",
    output_path="jepa_embedding_clusters.html",
)
fig.show()
```

For quick deterministic checks, use PCA:

```python
projection = torchwm.project_representation_embeddings(
    embeddings,
    labels=labels,
    method="pca",
)
print(projection.explained_variance_ratio)
```

## Saving and inspecting raw coordinates

If you want to use Matplotlib, Altair, pandas, or custom dashboard tooling, use
the `ProjectionResult` fields directly:

```python
projection = torchwm.project_latent_trajectories(states)

coords = projection.coordinates
for xy, sequence_id, timestep in zip(
    coords,
    projection.sequence_ids,
    projection.timesteps,
    strict=True,
):
    print(sequence_id, timestep, xy)
```

## Complete script

A runnable example is available at `examples/visualization_trackers_example.py`.
It generates synthetic RSSM trajectories and synthetic JEPA/ViT embeddings, then
writes `rssm_latent_trajectories.html` and `jepa_embedding_clusters.html`.

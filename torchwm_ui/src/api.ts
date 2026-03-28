import type { CatalogResponse, FrameResponse, MetricsResponse, StateResponse } from "./types";

export interface Dependency {
  name: string;
  label: string;
  required: boolean;
  installed: boolean;
}

export interface DependenciesResponse {
  dependencies: Dependency[];
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // keep fallback detail
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function fetchCatalog(): Promise<CatalogResponse> {
  return request<CatalogResponse>("/api/catalog");
}

export function fetchState(): Promise<StateResponse> {
  return request<StateResponse>("/api/state");
}

export function fetchMetrics(limit = 400): Promise<MetricsResponse> {
  return request<MetricsResponse>(`/api/metrics?limit=${limit}`);
}

export function fetchFrame(): Promise<FrameResponse> {
  return request<FrameResponse>("/api/frame");
}

export function loadModel(model: string, config: Record<string, unknown>): Promise<StateResponse> {
  return request<StateResponse>("/api/load-model", {
    method: "POST",
    body: JSON.stringify({ model, config })
  });
}

export function loadEnvironment(
  environment: string,
  backend: string = "dm_control",
  config: Record<string, unknown> = {}
): Promise<StateResponse> {
  return request<StateResponse>("/api/load-environment", {
    method: "POST",
    body: JSON.stringify({ environment, backend, config })
  });
}

export function startTraining(config: Record<string, unknown>): Promise<StateResponse> {
  return request<StateResponse>("/api/train/start", {
    method: "POST",
    body: JSON.stringify({ config })
  });
}

export function stopTraining(): Promise<{ stop_requested: boolean } & StateResponse> {
  return request<{ stop_requested: boolean } & StateResponse>("/api/train/stop", {
    method: "POST"
  });
}

export function fetchDependencies(): Promise<DependenciesResponse> {
  return request<DependenciesResponse>("/api/dependencies");
}

export function visualizeLatents(
  latents: string,
  method: "tsne" | "umap" = "tsne",
  labels?: string,
  perplexity = 30,
  nNeighbors = 15,
  shape?: number[]
): Promise<{ html: string }> {
  const params = new URLSearchParams({
    latents,
    method,
    perplexity: perplexity.toString(),
    n_neighbors: nNeighbors.toString(),
  });
  if (labels) {
    params.append("labels", labels);
  }
  if (shape && shape.length > 0) {
    params.append("shape", shape.join(","));
  }
  return request<{ html: string }>(`/api/visualize?${params}`, { method: "POST" });
}

export function fetchLatents(): Promise<{ latents: string }> {
  // fetchLatents needs to handle 204 No Content (latents not ready) specially
  return fetch(`${API_BASE}/api/latents`).then(async (response) => {
    if (response.status === 204) {
      throw new Error("Latents not ready");
    }
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body.detail) detail = body.detail;
      } catch {}
      throw new Error(detail);
    }
    // Some endpoints may return empty body; handle gracefully
    try {
      return (await response.json()) as { latents: string };
    } catch {
      throw new Error("Latents not ready");
    }
  });
}

export function fetchVideo(filename: string): string {
  return `${API_BASE}/api/video/${filename}`;
}

import type { CatalogResponse, FrameResponse, MetricsResponse, StateResponse } from "./types";

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
  config: Record<string, unknown>
): Promise<StateResponse> {
  return request<StateResponse>("/api/load-environment", {
    method: "POST",
    body: JSON.stringify({ environment, config })
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

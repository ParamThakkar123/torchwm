export interface CatalogResponse {
  models: Record<string, { label: string; description: string }>;
  environments_by_model: Record<string, string[]>;
  default_model_configs: Record<string, Record<string, unknown>>;
  default_training_configs: Record<string, Record<string, unknown>>;
}

export interface StateResponse {
  model: string | null;
  environment: string | null;
  status: "idle" | "running" | "completed" | "failed" | "stopped";
  message: string;
  traceback: string | null;
  started_at: number | null;
  finished_at: number | null;
  results_dir: string | null;
  progress: {
    current: number;
    total: number;
    unit: string;
    ratio: number;
  };
}

export interface MetricPoint {
  step: number;
  value: number;
  timestamp: number;
}

export interface MetricsResponse {
  series: Record<string, MetricPoint[]>;
}

export interface FrameResponse {
  image: string | null;
}

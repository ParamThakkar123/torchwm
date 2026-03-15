export interface CatalogResponse {
  models: Record<string, { label: string; description: string }>;
  environments_by_model: Record<string, string[]>;
  env_backends: Record<string, { label: string; description: string; environments: string[] }>;
  default_model_configs: Record<string, Record<string, unknown>>;
  default_training_configs: Record<string, Record<string, unknown>>;
  components: Record<string, ComponentDefinition>;
}

export interface ComponentDefinition {
  label: string;
  description: string;
  category: "environment" | "encoder" | "decoder" | "rssm" | "reward" | "value" | "actor" | "optimizer" | "memory";
  hyperparameters: HyperparameterDefinition[];
  inputs: string[];
  outputs: string[];
  icon: string;
  source_file?: string;
}

export interface HyperparameterDefinition {
  name: string;
  label: string;
  type: "number" | "string" | "boolean" | "select";
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
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
  gif: string | null;
}

export interface Dependency {
  name: string;
  label: string;
  required: boolean;
  installed: boolean;
}

export interface FlowchartNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    component: string;
    label: string;
    config: Record<string, number | string | boolean>;
  };
}

export interface FlowchartEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface FlowchartConfig {
  nodes: FlowchartNode[];
  edges: FlowchartEdge[];
}

export interface StartFlowchartTrainingRequest {
  flowchart: FlowchartConfig;
  config: Record<string, unknown>;
}

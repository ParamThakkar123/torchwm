import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  fetchCatalog,
  fetchFrame,
  fetchMetrics,
  fetchState,
  loadEnvironment,
  loadModel,
  startTraining,
  stopTraining
} from "./api";
import type { CatalogResponse, MetricPoint, MetricsResponse, StateResponse } from "./types";

function parseObjectInput(input: string): Record<string, unknown> {
  const parsed = JSON.parse(input);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Config must be a JSON object.");
  }
  return parsed as Record<string, unknown>;
}

function formatEpochTime(epoch: number | null): string {
  if (epoch === null) {
    return "N/A";
  }
  return new Date(epoch * 1000).toLocaleString();
}

function SparklineCard({ name, points }: { name: string; points: MetricPoint[] }) {
  const data = points.map((point) => ({ step: point.step, value: point.value }));
  return (
    <article className="metric-card">
      <header>
        <h4>{name}</h4>
        <span>{points.length} pts</span>
      </header>
      <div className="metric-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <Line
              type="monotone"
              dataKey="value"
              stroke="#0f766e"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [state, setState] = useState<StateResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse>({ series: {} });
  const [frame, setFrame] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState("dreamer");
  const [selectedEnvironment, setSelectedEnvironment] = useState("");
  const [modelConfigText, setModelConfigText] = useState("{}");
  const [trainingConfigText, setTrainingConfigText] = useState("{}");
  const [activeMetric, setActiveMetric] = useState("");

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const environments = useMemo(() => {
    if (!catalog) {
      return [];
    }
    return catalog.environments_by_model[selectedModel] ?? [];
  }, [catalog, selectedModel]);

  const sortedMetricEntries = useMemo(
    () =>
      Object.entries(metrics.series).sort((a, b) =>
        a[0].localeCompare(b[0], "en", { sensitivity: "base" })
      ),
    [metrics.series]
  );

  const activeMetricData = useMemo(() => {
    return (metrics.series[activeMetric] ?? []).map((point) => ({
      step: point.step,
      value: point.value
    }));
  }, [activeMetric, metrics.series]);

  async function refreshState() {
    const [latestState, latestMetrics, latestFrame] = await Promise.all([
      fetchState(),
      fetchMetrics(500),
      fetchFrame()
    ]);
    setState(latestState);
    setMetrics(latestMetrics);
    setFrame(latestFrame.image);
  }

  useEffect(() => {
    const initialize = async () => {
      try {
        const [catalogResponse, currentState] = await Promise.all([fetchCatalog(), fetchState()]);
        setCatalog(catalogResponse);
        setState(currentState);

        const defaultModel = currentState.model ?? Object.keys(catalogResponse.models)[0] ?? "dreamer";
        setSelectedModel(defaultModel);

        const modelDefaults = catalogResponse.default_model_configs[defaultModel] ?? {};
        const trainingDefaults = catalogResponse.default_training_configs[defaultModel] ?? {};
        setModelConfigText(JSON.stringify(modelDefaults, null, 2));
        setTrainingConfigText(JSON.stringify(trainingDefaults, null, 2));

        const envList = catalogResponse.environments_by_model[defaultModel] ?? [];
        const currentEnv = currentState.environment ?? envList[0] ?? "";
        setSelectedEnvironment(currentEnv);
      } catch (error) {
        setErrorMessage((error as Error).message);
      }
    };
    void initialize();
  }, []);

  useEffect(() => {
    if (!catalog) {
      return;
    }

    const envList = catalog.environments_by_model[selectedModel] ?? [];
    if (!envList.includes(selectedEnvironment)) {
      setSelectedEnvironment(envList[0] ?? "");
    }

    const modelDefaults = catalog.default_model_configs[selectedModel] ?? {};
    const trainingDefaults = catalog.default_training_configs[selectedModel] ?? {};
    setModelConfigText(JSON.stringify(modelDefaults, null, 2));
    setTrainingConfigText(JSON.stringify(trainingDefaults, null, 2));
  }, [catalog, selectedModel]);

  useEffect(() => {
    if (sortedMetricEntries.length === 0) {
      setActiveMetric("");
      return;
    }
    const stillExists = sortedMetricEntries.some(([name]) => name === activeMetric);
    if (!stillExists) {
      setActiveMetric(sortedMetricEntries[0][0]);
    }
  }, [activeMetric, sortedMetricEntries]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshState().catch((error: Error) => {
        setErrorMessage(error.message);
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  async function runAction(action: () => Promise<unknown>) {
    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      await action();
      await refreshState();
    } catch (error) {
      setErrorMessage((error as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleLoadModel() {
    let parsed: Record<string, unknown>;
    try {
      parsed = parseObjectInput(modelConfigText);
    } catch (error) {
      setErrorMessage((error as Error).message);
      return;
    }
    await runAction(() => loadModel(selectedModel, parsed));
  }

  async function handleLoadEnvironment() {
    if (!selectedEnvironment) {
      setErrorMessage("Select an environment first.");
      return;
    }
    await runAction(() => loadEnvironment(selectedEnvironment, {}));
  }

  async function handleStartTraining() {
    let parsed: Record<string, unknown>;
    try {
      parsed = parseObjectInput(trainingConfigText);
    } catch (error) {
      setErrorMessage((error as Error).message);
      return;
    }
    await runAction(() => startTraining(parsed));
  }

  async function handleStopTraining() {
    await runAction(() => stopTraining());
  }

  const progressPercent = Math.round(((state?.progress.ratio ?? 0) * 1000) / 10);
  const statusClass = `status-pill status-${state?.status ?? "idle"}`;

  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden />

      <header className="topbar reveal">
        <div>
          <h1>TorchWM Studio</h1>
          <p>React control plane for model loading, environment loading, and training telemetry.</p>
        </div>
        <span className={statusClass}>{state?.status ?? "idle"}</span>
      </header>

      <main className="layout">
        <section className="control-panel card reveal">
          <h2>Run Control</h2>

          <label>
            Model
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {catalog &&
                Object.entries(catalog.models).map(([key, model]) => (
                  <option key={key} value={key}>
                    {model.label}
                  </option>
                ))}
            </select>
          </label>

          <label>
            Model Config (JSON)
            <textarea
              value={modelConfigText}
              onChange={(event) => setModelConfigText(event.target.value)}
              rows={7}
            />
          </label>
          <button type="button" onClick={() => void handleLoadModel()} disabled={isSubmitting}>
            Load Model
          </button>

          <label>
            Environment
            <select
              value={selectedEnvironment}
              onChange={(event) => setSelectedEnvironment(event.target.value)}
            >
              {environments.map((env) => (
                <option key={env} value={env}>
                  {env}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void handleLoadEnvironment()} disabled={isSubmitting}>
            Load Environment
          </button>

          <label>
            Training Config (JSON)
            <textarea
              value={trainingConfigText}
              onChange={(event) => setTrainingConfigText(event.target.value)}
              rows={10}
            />
          </label>

          <div className="actions">
            <button type="button" className="primary" onClick={() => void handleStartTraining()}>
              Start Training
            </button>
            <button type="button" className="danger" onClick={() => void handleStopTraining()}>
              Stop
            </button>
          </div>

          {errorMessage && <p className="error-message">{errorMessage}</p>}
        </section>

        <section className="dashboard">
          <article className="card status-card reveal">
            <h2>Status</h2>
            <p>{state?.message ?? "No session loaded yet."}</p>
            <p className="mono">Model: {state?.model ?? "none"}</p>
            <p className="mono">Environment: {state?.environment ?? "none"}</p>
            <p className="mono">Started: {formatEpochTime(state?.started_at ?? null)}</p>
            <p className="mono">Finished: {formatEpochTime(state?.finished_at ?? null)}</p>
            <p className="mono">Artifacts: {state?.results_dir ?? "n/a"}</p>

            <div className="progress-shell" role="progressbar" aria-valuenow={progressPercent}>
              <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
            </div>
            <p className="mono progress-label">
              {state?.progress.current ?? 0} / {state?.progress.total ?? 0} {state?.progress.unit}
            </p>
          </article>

          <article className="card chart-card reveal">
            <header>
              <h2>Tensorboard-Like Scalars</h2>
              <select value={activeMetric} onChange={(event) => setActiveMetric(event.target.value)}>
                {sortedMetricEntries.map(([name]) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </header>
            <div className="main-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activeMetricData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d7ccb9" />
                  <XAxis dataKey="step" stroke="#8a7f68" />
                  <YAxis stroke="#8a7f68" />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#ea580c"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </article>

          <article className="card frame-card reveal">
            <h2>Environment Progress</h2>
            {frame ? (
              <img src={frame} alt="Latest environment rollout frame" />
            ) : (
              <div className="frame-placeholder">No preview frame yet.</div>
            )}
          </article>

          <div className="metric-grid">
            {sortedMetricEntries.map(([name, points]) => (
              <SparklineCard key={name} name={name} points={points} />
            ))}
          </div>

          {state?.traceback && (
            <article className="card traceback-card">
              <h2>Failure Traceback</h2>
              <pre>{state.traceback}</pre>
            </article>
          )}
        </section>
      </main>
    </div>
  );
}

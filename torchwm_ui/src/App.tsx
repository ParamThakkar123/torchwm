import { useEffect, useMemo, useState } from "react";
import { Toaster, toast } from "react-hot-toast";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  fetchCatalog,
  fetchDependencies,
  fetchFrame,
  fetchLatents,
  fetchMetrics,
  fetchState,
  fetchVideo,
  loadEnvironment,
  loadModel,
  startTraining,
  stopTraining,
  visualizeLatents
} from "./api";
import type { CatalogResponse, Dependency, MetricPoint, MetricsResponse, StateResponse } from "./types";

function formatNumber(num: number): string {
  if (Math.abs(num) >= 1000000) return (num / 1000000).toFixed(2) + "M";
  if (Math.abs(num) >= 1000) return (num / 1000).toFixed(2) + "K";
  return num.toFixed(4);
}

interface ConfigField {
  name: string;
  label: string;
  type: "number" | "boolean" | "string";
}

const DREAMER_CONFIG_FIELDS: ConfigField[] = [
  { name: "total_steps", label: "Total Steps", type: "number" },
  { name: "seed_steps", label: "Seed Steps", type: "number" },
  { name: "update_steps", label: "Update Steps", type: "number" },
  { name: "collect_steps", label: "Collect Steps", type: "number" },
  { name: "test_interval", label: "Test Interval", type: "number" },
  { name: "enable_wandb", label: "Enable WandB", type: "boolean" },
  { name: "wandb_api_key", label: "WandB API Key", type: "string" },
  { name: "wandb_project", label: "WandB Project", type: "string" },
  { name: "wandb_entity", label: "WandB Entity", type: "string" },
  { name: "enable_tensorboard", label: "Enable TensorBoard", type: "boolean" },
];

const PLANET_CONFIG_FIELDS: ConfigField[] = [
  { name: "epochs", label: "Epochs", type: "number" },
  { name: "warmup_episodes", label: "Warmup Episodes", type: "number" },
  { name: "steps_per_epoch", label: "Steps/Epoch", type: "number" },
  { name: "batch_size", label: "Batch Size", type: "number" },
  { name: "horizon", label: "Horizon", type: "number" },
  { name: "beta", label: "Beta", type: "number" },
];

function MetricCard({ name, points }: { name: string; points: MetricPoint[] }) {
  const data = points.slice(-30).map((p) => ({ step: p.step, value: p.value }));
  const latestValue = points.length > 0 ? points[points.length - 1].value : undefined;
  const color = name.includes("loss") ? "#f43f5e" : name.includes("reward") ? "#10b981" : name.includes("actor") ? "#8b5cf6" : name.includes("value") ? "#f59e0b" : "#00d4ff";

  return (
    <div className="mini-metric">
      <div className="mini-metric-header">
        <span className="mini-metric-name">{name}</span>
        <span className="mini-metric-value" style={{ color }}>{latestValue !== undefined ? formatNumber(latestValue) : "—"}</span>
      </div>
      <div className="mini-metric-chart">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id={`g-${name.replace(/[^a-z]/g, "")}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill={`url(#g-${name.replace(/[^a-z]/g, "")})`} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [state, setState] = useState<StateResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse>({ series: {} });
  const [frame, setFrame] = useState<string | null>(null);
  const [gif, setGif] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState("dreamerv2");
  const [selectedBackend, setSelectedBackend] = useState("dm_control");
  const [selectedEnvironment, setSelectedEnvironment] = useState("");
  const [trainingConfig, setTrainingConfig] = useState<Record<string, unknown>>({});
  const [activeMetric, setActiveMetric] = useState("");
  const [activeTab, setActiveTab] = useState<"all" | "loss" | "reward" | "eval" | "viz">("all");
  const [isFullscreenPreview, setIsFullscreenPreview] = useState(false);
  const [previewMode, setPreviewMode] = useState<"image" | "gif">("image");

const [vizHtml, setVizHtml] = useState<string>("");
const [latentsInput, setLatentsInput] = useState("");
const [labelsInput, setLabelsInput] = useState("");
const [vizMethod, setVizMethod] = useState<"tsne" | "umap">("tsne");
const [videoFilename, setVideoFilename] = useState("test_stream.mp4");

const [latentsFetched, setLatentsFetched] = useState(false);

const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dependencies, setDependencies] = useState<Dependency[]>([]);
  const [showDependencies, setShowDependencies] = useState(false);

  const configFields = selectedModel === "planet" ? PLANET_CONFIG_FIELDS : DREAMER_CONFIG_FIELDS;
  const envBackends = catalog?.env_backends ?? {};
  const availableEnvironments = envBackends[selectedBackend]?.environments ?? [];
  const sortedMetricEntries = useMemo(() => Object.entries(metrics.series).sort((a, b) => a[0].localeCompare(b[0])), [metrics.series]);

  const filteredMetrics = useMemo(() => {
    if (activeTab === "all") return sortedMetricEntries;
    if (activeTab === "loss") return sortedMetricEntries.filter(([n]) => n.includes("loss"));
    if (activeTab === "reward") return sortedMetricEntries.filter(([n]) => n.includes("reward") && !n.startsWith("eval"));
    if (activeTab === "eval") return sortedMetricEntries.filter(([n]) => n.startsWith("eval"));
    return sortedMetricEntries;
  }, [sortedMetricEntries, activeTab]);

  const activeMetricData = useMemo(() => {
    if (activeTab === "loss") {
      const lossMetrics = sortedMetricEntries.filter(([n]) => n.includes("loss"));
      if (lossMetrics.length > 0) {
        const steps = new Set<number>();
        lossMetrics.forEach(([, points]) => points.forEach(p => steps.add(p.step)));
        const sortedSteps = Array.from(steps).sort((a, b) => a - b);
        return sortedSteps.map(step => {
          const entry: Record<string, number> = { step };
          lossMetrics.forEach(([name, points]) => {
            const point = points.find(p => p.step === step);
            if (point?.value !== undefined) {
              entry[name.replace(/\//g, '_')] = point.value;
            }
          });
          return entry;
        });
      }
    }
    return (metrics.series[activeMetric] ?? []).map((p) => ({ step: p.step, value: p.value }));
  }, [activeTab, activeMetric, metrics.series, sortedMetricEntries]);

  const latestMetrics = useMemo(() => { const r: Record<string, number> = {}; for (const [n, p] of Object.entries(metrics.series)) { if (p.length > 0) r[n] = p[p.length - 1].value; } return r; }, [metrics.series]);

  async function refreshState() {
    try {
      const [s, m, f] = await Promise.all([fetchState(), fetchMetrics(200), fetchFrame()]);
      setState(s); setMetrics(m); setFrame(f.image); setGif(f.gif);
    } catch { /* ignore */ }
  }

  useEffect(() => {
    const init = async () => {
      try {
        const [cat, st] = await Promise.all([fetchCatalog(), fetchState()]);
        setCatalog(cat); setState(st);
        const model = st.model ?? Object.keys(cat.models)[0] ?? "dreamerv2";
        setSelectedModel(model);

        const backends = Object.keys(cat.env_backends);
        setSelectedBackend(backends[0] ?? "dm_control");

        const envs = cat.env_backends[backends[0]]?.environments ?? [];
        setSelectedEnvironment(st.environment ?? envs[0] ?? "");

        const defaults = cat.default_training_configs[model] ?? {};
        setTrainingConfig(Object.fromEntries(Object.entries(defaults).map(([k, v]) => [k, Number(v)])));
        const m = await fetchMetrics(200);
        setMetrics(m);
      } catch (e) { setErrorMessage((e as Error).message); }
    };
    void init();
  }, []);

  useEffect(() => {
    if (!catalog) return;
    const defaults = catalog.default_training_configs[selectedModel] ?? {};
    setTrainingConfig(Object.fromEntries(Object.entries(defaults).map(([k, v]) => [k, Number(v)])));
  }, [catalog, selectedModel]);

  useEffect(() => {
    if (!catalog) return;
    const envs = catalog.env_backends[selectedBackend]?.environments ?? [];
    if (!envs.includes(selectedEnvironment)) setSelectedEnvironment(envs[0] ?? "");
  }, [catalog, selectedBackend]);

  useEffect(() => {
    if (sortedMetricEntries.length === 0) { setActiveMetric(""); return; }
    if (!sortedMetricEntries.some(([n]) => n === activeMetric)) setActiveMetric(sortedMetricEntries[0][0]);
  }, [activeMetric, sortedMetricEntries]);

useEffect(() => {
    let interval = 5000;
    if (state?.status === "running") interval = 500;
    const t = setInterval(refreshState, interval);
    return () => clearInterval(t);
  }, [state?.status]);

  useEffect(() => {
    const fetchDeps = async () => {
      try {
        const deps = await fetchDependencies();
        setDependencies(deps.dependencies);
      } catch { /* ignore */ }
    };
    void fetchDeps();
  }, []);

  useEffect(() => {
    if (activeTab === "viz" && !latentsFetched) {
      fetchLatents()
        .then((res) => {
          setLatentsInput(res.latents);
          setLatentsFetched(true);
          toast.success("Latents loaded automatically!");
        })
        .catch((e) => {
          setErrorMessage((e as Error).message);
        });
    }
  }, [activeTab, latentsFetched]);

  const handleConfigChange = (key: string, value: unknown) => {
    setTrainingConfig(prev => ({ ...prev, [key]: value }));
  };

  async function runAction(action: () => Promise<unknown>) {
    setIsSubmitting(true); setErrorMessage(null);
    try { await action(); await refreshState(); }
    catch (e) { setErrorMessage((e as Error).message); }
    finally { setIsSubmitting(false); }
  }

const handleLoadModel = () => {
    toast.loading("Loading model...", { id: "loadModel" });
    runAction(() => loadModel(selectedModel, {}))
      .then(() => toast.success(`Model "${selectedModel}" loaded successfully!`, { id: "loadModel" }))
      .catch(() => {});
  };
  const handleLoadEnv = () => {
    if (!selectedEnvironment) {
      setErrorMessage("Select env");
      toast.error("Please select an environment", { id: "loadEnv" });
      return;
    }
    toast.loading("Loading environment...", { id: "loadEnv" });
    runAction(() => loadEnvironment(selectedEnvironment, selectedBackend, {}))
      .then(() => toast.success(`Environment "${selectedEnvironment}" loaded successfully!`, { id: "loadEnv" }))
      .catch(() => {});
  };
  const handleStart = () => {
    toast.loading("Starting training...", { id: "startTraining" });
    runAction(() => startTraining(trainingConfig))
      .then(() => toast.success("Training started!", { id: "startTraining" }))
      .catch(() => {});
  };
  const handleStop = () => runAction(() => stopTraining());

  const progressPercent = Math.round(((state?.progress.ratio ?? 0) * 1000) / 10);

return (
    <div className="app">
      <Toaster position="top-right" />
      <header className="header">
        <div className="header-left">
          <h1>TorchWM Studio</h1>
          <span className={`status-badge status-${state?.status ?? "idle"}`}>{state?.status ?? "idle"}</span>
        </div>
<div className="header-right">
          <div className="dependencies-wrapper">
            <button
              className="dependencies-toggle"
              onClick={() => setShowDependencies(!showDependencies)}
            >
              <span className="deps-icon">📦</span>
              <span className="deps-label">Dependencies</span>
              <span className={`deps-arrow ${showDependencies ? 'open' : ''}`}>▼</span>
            </button>
            {showDependencies && (
              <div className="dependencies-dropdown">
                <div className="deps-header">Training Dependencies</div>
                <div className="deps-list">
                  {dependencies.map(dep => (
                    <div key={dep.name} className={`dep-item ${dep.installed ? 'installed' : 'missing'}`}>
                      <span className="dep-status">{dep.installed ? '✓' : '✗'}</span>
                      <span className="dep-label">{dep.label}</span>
                      {dep.required && !dep.installed && <span className="dep-required">Required</span>}
                    </div>
                  ))}
                </div>
                <div className="deps-summary">
                  {dependencies.filter(d => d.installed).length} / {dependencies.length} installed
                </div>
              </div>
            )}
          </div>
          <div className="progress-mini"><div className="progress-mini-bar" style={{ width: `${progressPercent}%` }} /></div>
          <span className="progress-text">{progressPercent}%</span>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <div className="sidebar-section">
            <label>Model</label>
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {catalog && Object.entries(catalog.models).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
            <button onClick={handleLoadModel} disabled={isSubmitting}>Load Model</button>
          </div>

          <div className="sidebar-section">
            <label>Environment</label>
            <select value={selectedBackend} onChange={e => setSelectedBackend(e.target.value)}>
              {catalog && Object.entries(catalog.env_backends).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
            <select value={selectedEnvironment} onChange={e => setSelectedEnvironment(e.target.value)}>
              {availableEnvironments.map(e => <option key={e} value={e}>{e}</option>)}
            </select>
            <button onClick={handleLoadEnv} disabled={isSubmitting}>Load Env</button>
          </div>

          <div className="sidebar-section">
            <label>Training Config</label>
            <div className="config-fields">
              {configFields.map(field => (
                <div key={field.name} className="config-field">
                  <span className="config-label">{field.label}</span>
                  {field.type === "boolean" ? (
                    <input
                      type="checkbox"
                      checked={Boolean(trainingConfig[field.name] ?? false)}
                      onChange={e => handleConfigChange(field.name, e.target.checked)}
                    />
                  ) : field.type === "string" ? (
                    <input
                      type="text"
                      value={String(trainingConfig[field.name] ?? "")}
                      onChange={e => handleConfigChange(field.name, e.target.value)}
                    />
                  ) : (
                    <input
                      type="number"
                      value={Number(trainingConfig[field.name] ?? 0)}
                      onChange={e => handleConfigChange(field.name, Number(e.target.value))}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-actions">
            <button className="btn-primary" onClick={handleStart} disabled={isSubmitting}>Start</button>
            <button className="btn-danger" onClick={handleStop} disabled={isSubmitting}>Stop</button>
          </div>

          {errorMessage && <div className="error">{errorMessage}</div>}
        </aside>

        <div className="content">
          <div className="stats-row">
            <div className="stat-box"><span className="stat-label">Step</span><span className="stat-value">{state?.progress.current ?? 0} / {state?.progress.total ?? 0}</span></div>
            <div className="stat-box"><span className="stat-label">Model Loss</span><span className="stat-value stat-loss">{latestMetrics["train/model_loss"]?.toFixed(4) ?? "—"}</span></div>
            <div className="stat-box"><span className="stat-label">Actor Loss</span><span className="stat-value stat-actor">{latestMetrics["train/actor_loss"]?.toFixed(4) ?? "—"}</span></div>
            <div className="stat-box"><span className="stat-label">Value Loss</span><span className="stat-value stat-value">{latestMetrics["train/value_loss"]?.toFixed(4) ?? "—"}</span></div>
            <div className="stat-box"><span className="stat-label">Train Reward</span><span className="stat-value stat-reward">{latestMetrics["train/avg_reward"]?.toFixed(2) ?? "—"}</span></div>
          </div>

          <div className="main-chart">
            <div className="chart-header">
              <div className="chart-tabs">
                <button className={`chart-tab ${activeTab === "all" ? "active" : ""}`} onClick={() => setActiveTab("all")}>All</button>
                <button className={`chart-tab ${activeTab === "loss" ? "active" : ""}`} onClick={() => setActiveTab("loss")}>Loss</button>
                <button className={`chart-tab ${activeTab === "reward" ? "active" : ""}`} onClick={() => setActiveTab("reward")}>Reward</button>
                <button className={`chart-tab ${activeTab === "eval" ? "active" : ""}`} onClick={() => setActiveTab("eval")}>Eval</button>
                <button className={`chart-tab ${activeTab === "viz" ? "active" : ""}`} onClick={() => setActiveTab("viz")}>Viz</button>
              </div>
              {activeTab !== "loss" && (
                <select value={activeMetric} onChange={e => setActiveMetric(e.target.value)}>
                  {sortedMetricEntries.map(([n]) => <option key={n} value={n}>{n}</option>)}
                </select>
              )}
            </div>
            {activeTab === "viz" ? (
              <div className="viz-content">
                 <div className="viz-inputs">
                   <div className="viz-field">
                     <label>Latents</label>
                     <textarea
                       value={latentsInput}
                       readOnly
                       placeholder="Latents will be loaded automatically when available"
                     />
                   </div>
                  <div className="viz-field">
                    <label>Labels (base64 numpy array, optional)</label>
                    <textarea
                      value={labelsInput}
                      onChange={e => setLabelsInput(e.target.value)}
                      placeholder="Enter base64 encoded labels array"
                    />
                  </div>
                  <div className="viz-field">
                    <label>Method</label>
                    <select value={vizMethod} onChange={e => setVizMethod(e.target.value as "tsne" | "umap")}>
                      <option value="tsne">t-SNE</option>
                      <option value="umap">UMAP</option>
                    </select>
                  </div>
                  <button
                    onClick={async () => {
                      try {
                        const res = await visualizeLatents(latentsInput, vizMethod, labelsInput);
                        setVizHtml(res.html);
                      } catch (e) {
                        setErrorMessage((e as Error).message);
                      }
                    }}
                  >
                    Generate Visualization
                  </button>
                </div>
                <div className="viz-display">
                  {vizHtml ? (
                    <div dangerouslySetInnerHTML={{ __html: vizHtml }} />
                  ) : (
                    <div className="viz-placeholder">No visualization generated</div>
                  )}
                </div>
                <div className="video-section">
                  <div className="viz-field">
                    <label>Video Filename</label>
                    <input
                      type="text"
                      value={videoFilename}
                      onChange={e => setVideoFilename(e.target.value)}
                      placeholder="e.g., test_stream.mp4"
                    />
                  </div>
                  <video controls width="100%" height="300">
                    <source src={fetchVideo(videoFilename)} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={activeMetricData}>
                  <defs>
                    <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} /><stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} /></linearGradient>
                    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} /><stop offset="95%" stopColor="#f43f5e" stopOpacity={0} /></linearGradient>
                    <linearGradient id="actorGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} /><stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} /></linearGradient>
                    <linearGradient id="valueGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} /><stop offset="95%" stopColor="#f59e0b" stopOpacity={0} /></linearGradient>
                    <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.3} /><stop offset="95%" stopColor="#10b981" stopOpacity={0} /></linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="step" stroke="#606078" fontSize={11} />
                  <YAxis stroke="#606078" fontSize={11} />
                  <Tooltip contentStyle={{ background: "#1c1c26", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px" }} />
                  {activeTab === "loss" ? (
                    <>
                      <Area type="monotone" dataKey="train_model_loss" stroke="#f43f5e" strokeWidth={2} fill="url(#lossGrad)" isAnimationActive={false} name="Model Loss" />
                      <Area type="monotone" dataKey="train_actor_loss" stroke="#8b5cf6" strokeWidth={2} fill="url(#actorGrad)" isAnimationActive={false} name="Actor Loss" />
                      <Area type="monotone" dataKey="train_value_loss" stroke="#f59e0b" strokeWidth={2} fill="url(#valueGrad)" isAnimationActive={false} name="Value Loss" />
                      <Legend />
                    </>
                  ) : (
                    <Area type="monotone" dataKey="value" stroke="#00d4ff" strokeWidth={2} fill="url(#chartGrad)" isAnimationActive={false} />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="preview-row">
            <div className={`preview-box ${isFullscreenPreview ? "fullscreen" : ""}`}>
              <div className="preview-header">
                <h3>Environment Preview</h3>
                <div className="preview-controls">
                  <button
                    className={`preview-mode-btn ${previewMode === "image" ? "active" : ""}`}
                    onClick={() => setPreviewMode("image")}
                    title="Image"
                  >
                    IMG
                  </button>
                  <button
                    className={`preview-mode-btn ${previewMode === "gif" ? "active" : ""}`}
                    onClick={() => setPreviewMode("gif")}
                    title="GIF"
                    disabled={!gif}
                  >
                    GIF
                  </button>
                  <button
                    className="fullscreen-btn"
                    onClick={() => setIsFullscreenPreview(!isFullscreenPreview)}
                    title={isFullscreenPreview ? "Exit fullscreen" : "Fullscreen"}
                  >
                    {isFullscreenPreview ? "⊠" : "⛶"}
                  </button>
                </div>
              </div>
              {previewMode === "gif" && gif ? (
                <img src={gif} alt="preview gif" />
              ) : frame ? (
                <img src={frame} alt="preview" />
              ) : (
                <div className="preview-placeholder">No preview</div>
              )}
            </div>
            {!isFullscreenPreview && (
              <div className="metrics-grid">
                {filteredMetrics.slice(0, 8).map(([n, p]) => <MetricCard key={n} name={n} points={p} />)}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

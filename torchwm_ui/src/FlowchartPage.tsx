import { useCallback, useState, useMemo, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Edge,
  Node,
  BackgroundVariant,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Toaster, toast } from "react-hot-toast";
import {
  fetchCatalog,
  fetchState,
  startFlowchartTraining,
  stopTraining,
  fetchComponentSource,
} from "./api";
import type { CatalogResponse, StateResponse, ComponentDefinition } from "./types";

const defaultNodeStyle = {
  background: "rgba(20, 20, 28, 0.95)",
  border: "1px solid rgba(255, 255, 255, 0.06)",
  borderRadius: "10px",
  padding: "12px",
  minWidth: "180px",
};

const nodeColors: Record<string, string> = {
  environment: "linear-gradient(135deg, #10b981, #059669)",
  encoder: "linear-gradient(135deg, #00d4ff, #0891b2)",
  decoder: "linear-gradient(135deg, #a855f7, #7c3aed)",
  rssm: "linear-gradient(135deg, #f59e0b, #d97706)",
  reward: "linear-gradient(135deg, #f43f5e, #e11d48)",
  value: "linear-gradient(135deg, #f43f5e, #e11d48)",
  actor: "linear-gradient(135deg, #8b5cf6, #6d28d9)",
  optimizer: "linear-gradient(135deg, #06b6d4, #0891b2)",
  memory: "linear-gradient(135deg, #6366f1, #4f46e5)",
};

const nodeIcons: Record<string, string> = {
  environment: "🌍",
  encoder: "📷",
  decoder: "🖼️",
  rssm: "🔄",
  reward: "🎯",
  value: "💰",
  actor: "🎮",
  optimizer: "⚡",
  memory: "💾",
};

interface FlowchartPageProps {
  onNavigate: (page: "training" | "flowchart") => void;
}

interface ComponentItem extends ComponentDefinition {
  key: string;
}

type FlowchartNode = Node;
type FlowchartEdge = Edge;

export default function FlowchartPage({ onNavigate }: FlowchartPageProps) {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [state, setState] = useState<StateResponse | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState<Record<string, number>>({});
  const [showCode, setShowCode] = useState(false);
  const [sourceCode, setSourceCode] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<string | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);

  const components = useMemo(() => {
    if (!catalog?.components) {
      return {
        environments: [] as ComponentItem[],
        encoders: [] as ComponentItem[],
        decoders: [] as ComponentItem[],
        rssm: [] as ComponentItem[],
        rewards: [] as ComponentItem[],
        values: [] as ComponentItem[],
        actors: [] as ComponentItem[],
        optimizers: [] as ComponentItem[],
        memories: [] as ComponentItem[],
      };
    }

    const comps = Object.entries(catalog.components);
    return {
      environments: comps.filter(([, c]) => c.category === "environment").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      encoders: comps.filter(([, c]) => c.category === "encoder").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      decoders: comps.filter(([, c]) => c.category === "decoder").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      rssm: comps.filter(([, c]) => c.category === "rssm").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      rewards: comps.filter(([, c]) => c.category === "reward").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      values: comps.filter(([, c]) => c.category === "value").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      actors: comps.filter(([, c]) => c.category === "actor").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      optimizers: comps.filter(([, c]) => c.category === "optimizer").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
      memories: comps.filter(([, c]) => c.category === "memory").map(([k, v]) => ({ key: k, ...v })) as ComponentItem[],
    };
  }, [catalog]);

  const componentDefaults = useMemo(() => {
    const defaults: Record<string, Record<string, number | string | boolean>> = {};
    if (catalog?.components) {
      Object.entries(catalog.components).forEach(([key, comp]) => {
        defaults[key] = {};
        comp.hyperparameters.forEach((hp) => {
          defaults[key][hp.name] = hp.default;
        });
      });
    }
    return defaults;
  }, [catalog]);

  useEffect(() => {
    const init = async () => {
      try {
        const [cat, st] = await Promise.all([fetchCatalog(), fetchState()]);
        setCatalog(cat);
        setState(st);
        setIsTraining(st.status === "running");

        const defaults = cat.default_training_configs[Object.keys(cat.models)[0] ?? "dreamerv2"] ?? {};
        setTrainingConfig(Object.fromEntries(Object.entries(defaults).map(([k, v]) => [k, Number(v)])));
      } catch (e) {
        console.error("Failed to initialize:", e);
      }
    };
    init();
  }, []);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: "#00d4ff", strokeWidth: 2 } }, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback(async (_: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setShowCode(false);

    const componentKey = node.data.component as string;
    if (componentKey && catalog?.components?.[componentKey]) {
      const comp = catalog.components[componentKey];
      if (comp.source_file) {
        setSourceLoading(true);
        try {
          const response = await fetchComponentSource(componentKey);
          setSourceCode(response.source);
          setSourceFile(response.file);
        } catch (e) {
          console.error("Failed to fetch source:", e);
          setSourceCode(null);
          setSourceFile(null);
        }
        setSourceLoading(false);
      } else {
        setSourceCode(null);
        setSourceFile(null);
      }
    }
  }, [catalog]);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const addNode = useCallback((componentKey: string, component: ComponentDefinition) => {
    const newNode: Node = {
      id: `${componentKey}-${Date.now()}`,
      type: "default",
      position: { x: Math.random() * 400 + 100, y: Math.random() * 300 + 100 },
      data: {
        component: componentKey,
        label: component.label,
        config: componentDefaults[componentKey] ?? {},
      },
      style: defaultNodeStyle as React.CSSProperties,
    };
    setNodes((nds) => [...nds, newNode]);
  }, [setNodes, componentDefaults]);

  const updateNodeConfig = useCallback((nodeId: string, config: Record<string, number | string | boolean>) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: { ...node.data, config },
          };
        }
        return node;
      })
    );
    setSelectedNode((prev) =>
      prev?.id === nodeId ? { ...prev, data: { ...prev.data, config } } : prev
    );
  }, [setNodes]);

  const deleteSelectedNode = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
      setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
      setSelectedNode(null);
    }
  }, [selectedNode, setNodes, setEdges]);

  const clearFlowchart = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
  }, [setNodes, setEdges]);

  const handleTrain = async () => {
    if (nodes.length === 0) {
      toast.error("Add components to the flowchart first");
      return;
    }

    const flowchart = {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.component,
        position: n.position,
        data: { component: n.data.component, label: n.data.label, config: n.data.config },
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
      })),
    };

    toast.loading("Starting training...", { id: "train" });
    try {
      await startFlowchartTraining(flowchart as any, trainingConfig);
      toast.success("Training started!", { id: "train" });
      setIsTraining(true);
      onNavigate("training");
    } catch (e) {
      toast.error((e as Error).message, { id: "train" });
    }
  };

  const handleStop = async () => {
    try {
      await stopTraining();
      setIsTraining(false);
      toast.success("Training stopped");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const selectedComponent = selectedNode && catalog?.components?.[selectedNode.data.component as string];

  const renderComponentItem = (item: ComponentItem) => (
    <div
      key={item.key}
      className="component-item"
      onClick={() => addNode(item.key, item)}
    >
      <div className={`component-item-icon ${item.category}`}>
        {nodeIcons[item.category] || "📦"}
      </div>
      <div className="component-item-info">
        <div className="component-item-name">{item.label}</div>
        <div className="component-item-desc">{item.description}</div>
      </div>
    </div>
  );

  return (
    <div className="flowchart-page">
      <Toaster position="top-right" />

      <header className="flowchart-page-header">
        <h1>TorchWM Studio - Flowchart Builder</h1>
        <button className="nav-tab" onClick={() => onNavigate("training")}>← Back to Training</button>
      </header>

      <div className="flowchart-page-main">
        <aside className="flowchart-sidebar">
          <div className="flowchart-sidebar-header">
            <h2>Component Library</h2>
            <p>Click to add components</p>
          </div>

          <div className="component-palette">
            {components.environments.length > 0 && (
              <div className="component-category">
                <div className="component-category-title">Environments</div>
                {components.environments.map(renderComponentItem)}
            </div>
          )}

          {components.encoders.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Encoders</div>
              {components.encoders.map(renderComponentItem)}
            </div>
          )}

          {components.rssm.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">State Model (RSSM)</div>
              {components.rssm.map(renderComponentItem)}
            </div>
          )}

          {components.decoders.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Decoders</div>
              {components.decoders.map(renderComponentItem)}
            </div>
          )}

          {components.actors.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Actors</div>
              {components.actors.map(renderComponentItem)}
            </div>
          )}

          {components.values.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Value Networks</div>
              {components.values.map(renderComponentItem)}
            </div>
          )}

          {components.rewards.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Reward Models</div>
              {components.rewards.map(renderComponentItem)}
            </div>
          )}

          {components.optimizers.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Optimizers</div>
              {components.optimizers.map(renderComponentItem)}
            </div>
          )}

          {components.memories.length > 0 && (
            <div className="component-category">
              <div className="component-category-title">Memory</div>
              {components.memories.map(renderComponentItem)}
            </div>
          )}

          {Object.keys(components).every(k => components[k as keyof typeof components].length === 0) && (
            <>
              <div className="component-category">
                <div className="component-category-title">Environments</div>
                <div className="component-item" onClick={() => addNode("dmc_env", { label: "DM Control", description: "DeepMind Control Suite environments", category: "environment", hyperparameters: [], inputs: [], outputs: [], icon: "🌍" })}>
                  <div className="component-item-icon env">🌍</div>
                  <div className="component-item-info">
                    <div className="component-item-name">DM Control</div>
                    <div className="component-item-desc">DeepMind Control Suite</div>
                  </div>
                </div>
                <div className="component-item" onClick={() => addNode("gym_env", { label: "Gym", description: "OpenAI Gym/Gymnasium environments", category: "environment", hyperparameters: [], inputs: [], outputs: [], icon: "🎲" })}>
                  <div className="component-item-icon env">🎲</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Gym</div>
                    <div className="component-item-desc">OpenAI Gym environments</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">Encoders</div>
                <div className="component-item" onClick={() => addNode("dreamer_encoder", { label: "CNN Encoder", description: "Convolutional encoder for images", category: "encoder", hyperparameters: [], inputs: [], outputs: [], icon: "📷" })}>
                  <div className="component-item-icon encoder">📷</div>
                  <div className="component-item-info">
                    <div className="component-item-name">CNN Encoder</div>
                    <div className="component-item-desc">Convolutional encoder for images</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">State Model (RSSM)</div>
                <div className="component-item" onClick={() => addNode("rssm", { label: "RSSM", description: "Recurrent State-Space Model", category: "rssm", hyperparameters: [], inputs: [], outputs: [], icon: "🔄" })}>
                  <div className="component-item-icon rssm">🔄</div>
                  <div className="component-item-info">
                    <div className="component-item-name">RSSM</div>
                    <div className="component-item-desc">Recurrent State-Space Model</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">Decoders</div>
                <div className="component-item" onClick={() => addNode("dreamer_decoder", { label: "CNN Decoder", description: "Deconvolutional decoder for reconstructions", category: "decoder", hyperparameters: [], inputs: [], outputs: [], icon: "🖼️" })}>
                  <div className="component-item-icon decoder">🖼️</div>
                  <div className="component-item-info">
                    <div className="component-item-name">CNN Decoder</div>
                    <div className="component-item-desc">Deconvolutional decoder</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">Reward Model</div>
                <div className="component-item" onClick={() => addNode("dreamer_reward", { label: "Reward Predictor", description: "Network for predicting rewards", category: "reward", hyperparameters: [], inputs: [], outputs: [], icon: "🎯" })}>
                  <div className="component-item-icon reward">🎯</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Reward Predictor</div>
                    <div className="component-item-desc">Network for predicting rewards</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">Policy & Value</div>
                <div className="component-item" onClick={() => addNode("dreamer_actor", { label: "Actor", description: "Stochastic actor for action selection", category: "actor", hyperparameters: [], inputs: [], outputs: [], icon: "🎮" })}>
                  <div className="component-item-icon actor">🎮</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Actor</div>
                    <div className="component-item-desc">Stochastic actor network</div>
                  </div>
                </div>
                <div className="component-item" onClick={() => addNode("dreamer_value", { label: "Value Network", description: "Value network for state estimation", category: "value", hyperparameters: [], inputs: [], outputs: [], icon: "💰" })}>
                  <div className="component-item-icon value">💰</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Value Network</div>
                    <div className="component-item-desc">Value estimation network</div>
                  </div>
                </div>
              </div>

              <div className="component-category">
                <div className="component-category-title">Memory & Optimizer</div>
                <div className="component-item" onClick={() => addNode("replay_memory", { label: "Replay Memory", description: "Experience replay buffer", category: "memory", hyperparameters: [], inputs: [], outputs: [], icon: "💾" })}>
                  <div className="component-item-icon memory">💾</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Replay Memory</div>
                    <div className="component-item-desc">Experience replay buffer</div>
                  </div>
                </div>
                <div className="component-item" onClick={() => addNode("adam_optimizer", { label: "Adam Optimizer", description: "Adaptive moment estimation optimizer", category: "optimizer", hyperparameters: [], inputs: [], outputs: [], icon: "⚡" })}>
                  <div className="component-item-icon optimizer">⚡</div>
                  <div className="component-item-info">
                    <div className="component-item-name">Adam Optimizer</div>
                    <div className="component-item-desc">Adaptive optimization</div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </aside>

      <div className="flowchart-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          fitView
          snapToGrid
          snapGrid={[15, 15]}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(255,255,255,0.05)" />
          <Controls />
          <MiniMap
            nodeColor={(node) => nodeColors[node.data?.component as string] || "#6366f1"}
            maskColor="rgba(8, 8, 12, 0.8)"
          />
          <Panel position="top-right">
            <div className="flowchart-toolbar">
              <button onClick={clearFlowchart}> Clear</button>
              {selectedNode && (
                <button onClick={deleteSelectedNode} style={{ color: "#f43f5e" }}>
                  Delete
                </button>
              )}
              {isTraining ? (
                <button className="train-btn" onClick={handleStop} style={{ background: "linear-gradient(135deg, #f43f5e, #e11d48)" }}>
                  Stop Training
                </button>
              ) : (
                <button className="train-btn" onClick={handleTrain}>
                  Train Model
                </button>
              )}
            </div>
          </Panel>
        </ReactFlow>
      </div>

      <aside className="hyperparameter-panel">
        <div className="hyperparameter-header">
          <h2>{selectedNode ? (selectedNode.data as any).label : "Configuration"}</h2>
          <div className="hyperparameter-header-actions">
            {selectedNode && sourceFile && (
              <button
                className="view-code-btn"
                onClick={() => setShowCode(!showCode)}
              >
                {showCode ? "⚙️ Settings" : "📄 View Code"}
              </button>
            )}
            {selectedNode && (
              <button className="hyperparameter-close" onClick={() => setSelectedNode(null)}>
                ✕
              </button>
            )}
          </div>
        </div>

        <div className="hyperparameter-content">
          {showCode && sourceCode ? (
            <div className="code-viewer">
              <div className="code-viewer-header">
                <span className="code-file">{sourceFile}</span>
              </div>
              <pre className="code-content">{sourceCode}</pre>
            </div>
          ) : selectedNode && selectedComponent ? (
            <>
              <div className="hyperparameter-group">
                <div className="hyperparameter-group-title">Component Settings</div>
                {selectedComponent.hyperparameters.map((hp) => (
                  <div key={hp.name} className="hyperparameter-field">
                    <label>{hp.label}</label>
                    {hp.type === "number" ? (
                      <div className="hyperparameter-field-range">
                        <input
                          type="range"
                          min={hp.min ?? 0}
                          max={hp.max ?? 1000}
                          step={hp.step ?? 1}
                          value={((selectedNode.data as any).config[hp.name] as number) ?? hp.default}
                          onChange={(e) =>
                            updateNodeConfig(selectedNode.id, {
                              ...(selectedNode.data as any).config,
                              [hp.name]: Number(e.target.value),
                            })
                          }
                        />
                        <span>{String((selectedNode.data as any).config[hp.name] ?? hp.default)}</span>
                      </div>
                    ) : hp.type === "select" ? (
                      <select
                        value={((selectedNode.data as any).config[hp.name] as string) ?? hp.default}
                        onChange={(e) =>
                          updateNodeConfig(selectedNode.id, {
                            ...(selectedNode.data as any).config,
                            [hp.name]: e.target.value,
                          })
                        }
                      >
                        {hp.options?.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : hp.type === "boolean" ? (
                      <input
                        type="checkbox"
                        checked={((selectedNode.data as any).config[hp.name] as boolean) ?? hp.default}
                        onChange={(e) =>
                          updateNodeConfig(selectedNode.id, {
                            ...(selectedNode.data as any).config,
                            [hp.name]: e.target.checked,
                          })
                        }
                      />
                    ) : (
                      <input
                        type="text"
                        value={String((selectedNode.data as any).config[hp.name] ?? hp.default)}
                        onChange={(e) =>
                          updateNodeConfig(selectedNode.id, {
                            ...(selectedNode.data as any).config,
                            [hp.name]: e.target.value,
                          })
                        }
                      />
                    )}
                  </div>
                ))}
              </div>

              <div className="hyperparameter-group">
                <div className="hyperparameter-group-title">Connections</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  <div>Inputs: {selectedComponent.inputs.join(", ") || "None"}</div>
                  <div style={{ marginTop: "0.25rem" }}>Outputs: {selectedComponent.outputs.join(", ") || "None"}</div>
                </div>
              </div>
            </>
          ) : (
            <div className="hyperparameter-empty">
              <div className="hyperparameter-empty-icon">🔧</div>
              <p>Select a component to configure its hyperparameters</p>
            </div>
          )}

          {!selectedNode && (
            <>
              <div className="hyperparameter-group">
                <div className="hyperparameter-group-title">Training Settings</div>
                {Object.entries(trainingConfig).map(([key, value]) => (
                  <div key={key} className="hyperparameter-field">
                    <label>{key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}</label>
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setTrainingConfig((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                    />
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </aside>
      </div>
    </div>
  );
}

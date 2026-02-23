import { expect, Page, test } from "@playwright/test";

type JsonRecord = Record<string, unknown>;

interface MockBackend {
  requests: {
    loadModel: JsonRecord[];
    loadEnvironment: JsonRecord[];
    startTraining: JsonRecord[];
    stopTraining: number;
  };
}

const FRAME_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAsMB9Fwrq6QAAAAASUVORK5CYII=";

function json(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    body: JSON.stringify(body)
  };
}

function parseBody(payload: string | null): JsonRecord {
  if (!payload) {
    return {};
  }
  return JSON.parse(payload) as JsonRecord;
}

async function setupApiMock(page: Page): Promise<MockBackend> {
  const catalog = {
    models: {
      dreamer: { label: "Dreamer", description: "Dreamer agent" },
      jepa: { label: "JEPA", description: "JEPA agent" }
    },
    environments_by_model: {
      dreamer: ["Pendulum-v1", "CartPole-v1"],
      jepa: ["ImageFolder"]
    },
    default_model_configs: {
      dreamer: { latent_size: 32 },
      jepa: { embed_dim: 256 }
    },
    default_training_configs: {
      dreamer: { total_steps: 1_000 },
      jepa: { epochs: 10 }
    }
  };

  const state = {
    model: "dreamer",
    environment: "Pendulum-v1",
    status: "idle",
    message: "Ready.",
    traceback: null,
    started_at: null,
    finished_at: null,
    results_dir: null,
    progress: {
      current: 0,
      total: 10,
      unit: "steps",
      ratio: 0
    }
  };

  const metrics = {
    series: {
      loss: [
        { step: 1, value: 1.2, timestamp: 1_709_000_001 },
        { step: 2, value: 0.8, timestamp: 1_709_000_002 }
      ],
      reward: [
        { step: 1, value: 3, timestamp: 1_709_000_001 },
        { step: 2, value: 5, timestamp: 1_709_000_002 }
      ]
    }
  };

  const requests = {
    loadModel: [] as JsonRecord[],
    loadEnvironment: [] as JsonRecord[],
    startTraining: [] as JsonRecord[],
    stopTraining: 0
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const { pathname } = url;

    if (pathname === "/api/catalog" && request.method() === "GET") {
      await route.fulfill(json(catalog));
      return;
    }

    if (pathname === "/api/state" && request.method() === "GET") {
      if (state.status === "running" && state.progress.current < state.progress.total) {
        state.progress.current += 1;
        state.progress.ratio = state.progress.current / state.progress.total;
        state.message = `Training in progress (${state.progress.current}/${state.progress.total})`;
      }
      await route.fulfill(json(state));
      return;
    }

    if (pathname === "/api/metrics" && request.method() === "GET") {
      await route.fulfill(json(metrics));
      return;
    }

    if (pathname === "/api/frame" && request.method() === "GET") {
      await route.fulfill(json({ image: FRAME_IMAGE }));
      return;
    }

    if (pathname === "/api/load-model" && request.method() === "POST") {
      const payload = parseBody(request.postData());
      requests.loadModel.push(payload);
      state.model = String(payload.model ?? state.model);
      state.message = `Loaded model: ${state.model}`;
      await route.fulfill(json(state));
      return;
    }

    if (pathname === "/api/load-environment" && request.method() === "POST") {
      const payload = parseBody(request.postData());
      requests.loadEnvironment.push(payload);
      state.environment = String(payload.environment ?? state.environment);
      state.message = `Loaded environment: ${state.environment}`;
      await route.fulfill(json(state));
      return;
    }

    if (pathname === "/api/train/start" && request.method() === "POST") {
      const payload = parseBody(request.postData());
      requests.startTraining.push(payload);
      state.status = "running";
      state.started_at = Math.floor(Date.now() / 1000);
      state.finished_at = null;
      state.message = "Training started.";
      state.progress.current = 0;
      state.progress.ratio = 0;
      await route.fulfill(json(state));
      return;
    }

    if (pathname === "/api/train/stop" && request.method() === "POST") {
      requests.stopTraining += 1;
      state.status = "stopped";
      state.finished_at = Math.floor(Date.now() / 1000);
      state.message = "Stop requested.";
      await route.fulfill(json({ ...state, stop_requested: true }));
      return;
    }

    await route.fulfill(json({ detail: `Unhandled route: ${pathname}` }, 404));
  });

  return { requests };
}

test.describe("TorchWM Studio E2E", () => {
  test("loads defaults and renders dashboard widgets", async ({ page }) => {
    await setupApiMock(page);
    await page.goto("/");
    const modelSelect = page.locator('label:has-text("Model") select');
    const environmentSelect = page.locator('label:has-text("Environment") select');

    await expect(page.getByRole("heading", { name: "TorchWM Studio" })).toBeVisible();
    await expect(modelSelect).toHaveValue("dreamer");
    await expect(environmentSelect).toHaveValue("Pendulum-v1");
    await expect(page.locator(".status-pill")).toContainText("idle");
    await expect(page.getByText("No preview frame yet.")).not.toBeVisible();
    await expect(page.getByAltText("Latest environment rollout frame")).toBeVisible();
    await expect(page.getByRole("heading", { name: "loss" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "reward" })).toBeVisible();
  });

  test("validates JSON and handles model/environment button actions", async ({ page }) => {
    const backend = await setupApiMock(page);
    await page.goto("/");
    const modelSelect = page.locator('label:has-text("Model") select');
    const environmentSelect = page.locator('label:has-text("Environment") select');

    await modelSelect.selectOption("jepa");
    await expect(environmentSelect).toHaveValue("ImageFolder");

    await page.getByLabel("Model Config (JSON)").fill("[]");
    await page.getByRole("button", { name: "Load Model" }).click();
    await expect(page.getByText("Config must be a JSON object.")).toBeVisible();

    await page.getByLabel("Model Config (JSON)").fill('{"embed_dim": 512}');
    await page.getByRole("button", { name: "Load Model" }).click();
    await expect(page.locator(".status-card")).toContainText("Loaded model: jepa");
    expect(backend.requests.loadModel.at(-1)).toEqual({
      model: "jepa",
      config: { embed_dim: 512 }
    });

    await page.getByRole("button", { name: "Load Environment" }).click();
    await expect(page.locator(".status-card")).toContainText("Loaded environment: ImageFolder");
    expect(backend.requests.loadEnvironment.at(-1)).toEqual({
      environment: "ImageFolder",
      config: {}
    });
  });

  test("starts and stops training via button clicks", async ({ page }) => {
    const backend = await setupApiMock(page);
    await page.goto("/");

    await page.getByLabel("Training Config (JSON)").fill('{"total_steps": 4}');
    await page.getByRole("button", { name: "Start Training" }).click();

    await expect(page.locator(".status-pill")).toContainText("running");
    await expect(page.locator(".status-card")).toContainText("Training in progress");
    expect(backend.requests.startTraining.at(-1)).toEqual({
      config: { total_steps: 4 }
    });

    await page.getByRole("button", { name: "Stop" }).click();
    await expect(page.locator(".status-pill")).toContainText("stopped");
    await expect(page.locator(".status-card")).toContainText("Stop requested.");
    expect(backend.requests.stopTraining).toBeGreaterThan(0);
  });
});

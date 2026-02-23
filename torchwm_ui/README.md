# TorchWM React UI

This folder contains a React + TypeScript frontend for controlling TorchWM model training.

## 1. Start the backend API

From the repository root:

```bash
uvicorn world_models.ui.server:app --reload
```

Backend defaults:
- API base URL: `http://127.0.0.1:8000`
- Frontend CORS origin: `http://127.0.0.1:5173`

## 2. Start the frontend

From `torchwm_ui/`:

```bash
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Optional API URL override

If your backend runs elsewhere:

```bash
set VITE_API_URL=http://127.0.0.1:9000
npm run dev
```

## UI end-to-end tests (browser automation)

These tests open the UI and automate interactions (typing in JSON fields, clicking
buttons, changing dropdowns), similar to Selenium workflows.

From `torchwm_ui/`:

```bash
npm install
npx playwright install chromium
npm run test:e2e
```

Notes:
- Tests are in `torchwm_ui/e2e/app.spec.ts`.
- API calls are mocked in-browser, so the backend server is not required.

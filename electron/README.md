# TorchWM Desktop

Desktop application for training world models on Windows, macOS, and Linux.

## Architecture

- **Electron App**: Desktop wrapper that provides native window controls, system integration
- **Python Backend**: FastAPI server that runs PyTorch training (same as web version)
- **React UI**: Connects to the Python backend via HTTP API

When you train a model in the desktop app, it uses the same backend as the web app, so you can:
1. Train in the desktop app
2. View training progress in the web browser (if on the same network)

## Prerequisites

- Python 3.13+
- Node.js 18+
- pip

## Install Dependencies

### 1. Install Python dependencies

```bash
# Install torchwm (from source)
pip install -e .

# Or from PyPI
pip install torchwm
```

### 2. Install Electron dependencies

```bash
cd electron
npm install
```

## Running the Desktop App

### Development Mode

```bash
cd electron
npm run dev
```

This will:
1. Start the Python backend server on port 8000
2. Open the Electron app window
3. Load the React UI

### Building for Distribution

```bash
# Build for current platform
npm run build

# Build for specific platforms
npm run build:win   # Windows
npm run build:mac   # macOS
npm run build:linux # Linux
```

The built application will be in `electron/dist/`.

## Connecting Web App to Desktop Backend

If you want to view training on your phone/other devices:

1. Start the desktop app
2. Find your computer's IP address (e.g., `192.168.1.x`)
3. Open `http://192.168.1.x:5173` in your browser

Note: The desktop app must have network access enabled in your firewall.

## Project Structure

```
electron/
├── main.js          # Electron main process
├── preload.js       # Secure IPC bridge
├── package.json     # Electron dependencies
└── dist/            # Built application
```

## Troubleshooting

### Backend won't start

Make sure Python and torchwm are installed:
```bash
python --version
pip show torchwm
```

### Port already in use

The backend uses port 8000. If it's busy, you may need to stop other services.

### Training is slow

World model training requires GPU. Make sure you have CUDA (NVIDIA) or Metal (Apple Silicon) installed with PyTorch.

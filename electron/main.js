const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let mainWindow = null;
let pythonProcess = null;

const isDev = !app.isPackaged;
const SERVER_PORT = 8000;
const UI_PORT = 5173;

function getDistPath() {
  if (isDev) {
    return null;
  }
  return path.join(__dirname, '../torchwm_ui/dist');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'icon.png'),
    show: false,
    backgroundColor: '#1a1a2e'
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (isDev) {
    mainWindow.loadURL(`http://127.0.0.1:${UI_PORT}`);
    mainWindow.webContents.openDevTools();
  } else {
    const distPath = getDistPath();
    if (distPath && fs.existsSync(distPath)) {
      mainWindow.loadFile(path.join(distPath, 'index.html'));
    } else {
      console.error('Dist folder not found:', distPath);
    }
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonServer() {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

    pythonProcess = spawn(pythonCmd, [
      '-m', 'uvicorn',
      'world_models.ui.server:app',
      '--host', '0.0.0.0',
      '--port', String(SERVER_PORT),
      '--reload'
    ], {
      cwd: path.join(__dirname, '..'),
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: true,
      env: {
        ...process.env,
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'ELECTRON_RUN': 'true'
      }
    });

    let started = false;

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('[TorchWM Backend]', output);

      if (!started && output.includes('Application startup complete')) {
        started = true;
        setTimeout(() => resolve(), 1000);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.log('[TorchWM Backend]', output);
    });

    pythonProcess.on('error', (err) => {
      console.error('Failed to start backend:', err);
      reject(err);
    });

    pythonProcess.on('close', (code) => {
      console.log('Backend closed with code:', code);
      pythonProcess = null;
    });

    setTimeout(() => {
      if (!started) resolve();
    }, 8000);
  });
}

function stopPythonServer() {
  if (pythonProcess) {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
    } else {
      pythonProcess.kill('SIGTERM');
    }
    pythonProcess = null;
  }
}

function checkPython() {
  return new Promise((resolve) => {
    const cmd = process.platform === 'win32' ? 'python --version' : 'python3 --version';
    require('child_process').exec(cmd, (error, stdout) => {
      resolve({ available: !error, version: stdout.trim() });
    });
  });
}

function checkTorchwm() {
  return new Promise((resolve) => {
    require('child_process').exec('pip show torchwm', (error) => {
      resolve(!error);
    });
  });
}

function checkBackendRunning() {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${SERVER_PORT}/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.end();
  });
}

app.whenReady().then(async () => {
  console.log('Starting TorchWM Desktop...');

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('start-backend', async () => {
  try {
    const running = await checkBackendRunning();
    if (running) {
      return { success: true, message: 'Backend already running' };
    }
    await startPythonServer();
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-backend', async () => {
  stopPythonServer();
  return { success: true };
});

ipcMain.handle('get-backend-status', async () => {
  const running = await checkBackendRunning();
  return { running, port: SERVER_PORT };
});

ipcMain.handle('check-environment', async () => {
  const python = await checkPython();
  const torchwm = await checkTorchwm();
  const backendRunning = await checkBackendRunning();
  return {
    python,
    torchwm: torchwm ? 'installed' : 'not installed',
    backendRunning
  };
});

ipcMain.handle('get-server-url', () => {
  return `http://127.0.0.1:${SERVER_PORT}`;
});

ipcMain.handle('open-external', async (event, url) => {
  await shell.openExternal(url);
});

ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('select-file', async (event, filters) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: filters || []
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('open-web-ui', async () => {
  const running = await checkBackendRunning();
  if (running) {
    await shell.openExternal(`http://127.0.0.1:${UI_PORT}`);
  }
});

const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');

let mainWindow = null;
let pythonProcess = null;

const isDev = !app.isPackaged;
const SERVER_PORT = 8000;
const UI_PORT = 5173;

log.transports.file.level = 'info';
autoUpdater.logger = log;

function getWebUIPath() {
  if (isDev) {
    return path.join(__dirname, '../torchwm_ui/dist');
  }
  return path.join(process.resourcesPath, 'web-ui');
}

function setupAutoUpdater() {
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on('checking-for-update', () => {
    log.info('Checking for updates...');
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'checking' });
    }
  });

  autoUpdater.on('update-available', (info) => {
    log.info('Update available:', info.version);
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'available', version: info.version });
    }
  });

  autoUpdater.on('update-not-available', () => {
    log.info('No updates available');
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'not-available' });
    }
  });

  autoUpdater.on('download-progress', (progress) => {
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'downloading', percent: progress.percent });
    }
  });

  autoUpdater.on('update-downloaded', (info) => {
    log.info('Update downloaded:', info.version);
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'downloaded', version: info.version });
    }
  });

  autoUpdater.on('error', (err) => {
    log.error('Auto-updater error:', err);
    if (mainWindow) {
      mainWindow.webContents.send('update-status', { status: 'error', error: err.message });
    }
  });
}

function checkForUpdates() {
  if (!isDev) {
    autoUpdater.checkForUpdates().catch(err => {
      log.error('Error checking for updates:', err);
    });
  }
}

function createWindow() {
  const iconPath = isDev
    ? path.join(__dirname, 'icon.png')
    : path.join(process.resourcesPath, 'app.asar', 'icon.png');

  const windowOptions = {
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    show: false,
    backgroundColor: '#1a1a2e'
  };

  if (fs.existsSync(iconPath)) {
    windowOptions.icon = iconPath;
  }

  mainWindow = new BrowserWindow(windowOptions);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    if (!isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  if (isDev) {
    const webUIPath = getWebUIPath();
    console.log('Loading web UI from:', webUIPath);
    if (fs.existsSync(webUIPath)) {
      mainWindow.loadFile(path.join(webUIPath, 'index.html'));
    } else {
      console.error('Web UI not found at:', webUIPath);
      mainWindow.loadURL('data:text/html,<html><body><h1>Error: Web UI not found</h1></body></html>');
    }
  } else {
    const webUIPath = getWebUIPath();
    console.log('Loading web UI from:', webUIPath);
    if (fs.existsSync(webUIPath)) {
      mainWindow.loadFile(path.join(webUIPath, 'index.html'));
    } else {
      console.error('Web UI not found at:', webUIPath);
      mainWindow.loadURL('data:text/html,<html><body><h1>Error: Web UI not found</h1></body></html>');
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
  console.log('Is packaged:', app.isPackaged);

  try {
    createWindow();
  } catch (error) {
    console.error('Error creating window:', error);
  }

  console.log('Starting backend...');
  try {
    await startPythonServer();
    console.log('Backend started successfully');
  } catch (error) {
    console.error('Failed to start backend:', error);
  }

  setupAutoUpdater();

  if (!isDev) {
    setTimeout(() => {
      checkForUpdates();
    }, 3000);
  }

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

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection:', reason);
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

ipcMain.handle('check-for-updates', async () => {
  if (isDev) {
    return { success: false, error: 'Updates are not available in development mode' };
  }
  try {
    await autoUpdater.checkForUpdates();
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('download-update', async () => {
  if (isDev) {
    return { success: false, error: 'Updates are not available in development mode' };
  }
  try {
    await autoUpdater.downloadUpdate();
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('install-update', () => {
  autoUpdater.quitAndInstall();
});

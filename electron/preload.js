const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  startBackend: () => ipcRenderer.invoke('start-backend'),
  stopBackend: () => ipcRenderer.invoke('stop-backend'),
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  checkEnvironment: () => ipcRenderer.invoke('check-environment'),
  getServerUrl: () => ipcRenderer.invoke('get-server-url'),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  selectFile: (filters) => ipcRenderer.invoke('select-file', filters),
  openWebUI: () => ipcRenderer.invoke('open-web-ui'),

  onTrainingUpdate: (callback) => {
    ipcRenderer.on('training-update', (event, data) => callback(data));
  },
  onLogMessage: (callback) => {
    ipcRenderer.on('log-message', (event, data) => callback(data));
  }
});

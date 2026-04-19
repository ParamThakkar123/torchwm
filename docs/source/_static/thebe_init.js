// Initialize Thebe using config injected by Sphinx `thebe_config`
window.addEventListener('load', async () => {
  try {
    if (!window.thebe) {
      console.warn('Thebe not loaded yet');
      return;
    }

    // Use thebe_config injected by conf.py via Sphinx thebe extension
    const cfg = window.thebe_config || {};
    // Provide sane defaults if missing
    const defaultCfg = {
      requestKernel: true,
      binderOptions: {
        repo: 'paramthakkar123/torchwm',
        ref: 'main',
        binderUrl: 'https://mybinder.org',
      },
    };

    const finalCfg = Object.assign({}, defaultCfg, cfg || {});

    // Initialize Thebe
    window.thebe.bootstrap(finalCfg);
  } catch (err) {
    console.warn('Thebe init failed', err);
  }
});

import { getConfig } from './api.js';
import { renderConfig } from './renderers.js';
import { initRuns } from './runs.js';
import { initTheme } from './theme.js';

initTheme();
initRuns();
getConfig().then(renderConfig).catch(() => {
  document.getElementById('env-model').textContent = 'unavailable';
});

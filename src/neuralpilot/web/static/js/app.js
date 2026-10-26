import { getConfig } from './api.js';
import { initAuth } from './auth.js';
import { initHistory, refreshRuns } from './history.js';
import { renderConfig } from './renderers.js';
import { initRuns, openRun } from './runs.js';
import { initTheme } from './theme.js';

initTheme();
initRuns();
initHistory(openRun);
initAuth(refreshRuns);
getConfig().then(renderConfig).catch(() => {
  document.getElementById('env-model').textContent = 'unavailable';
});

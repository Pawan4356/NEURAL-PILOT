import { getConfig } from './api.js';
import { initAuth } from './auth.js';
import { initHistory, refreshRuns } from './history.js';
import { renderConfig } from './renderers.js';
import { initReport } from './report.js';
import { initRuns, openRun } from './runs.js';
import { initShell } from './shell.js';
import { initTheme } from './theme.js';
import { $ } from './dom.js';

initTheme();
initShell();
initRuns();
initReport();
initHistory(openRun);
initAuth(refreshRuns);

getConfig().then(renderConfig).catch(() => {
  $('env-model').textContent = 'unavailable';
});

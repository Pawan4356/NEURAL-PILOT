import { getConfig } from './api.js';
import { initAuth } from './auth.js';
import { initHistory, refreshRuns } from './history.js';
import { renderConfig } from './renderers.js';
import { initReport } from './report.js';
import { initRuns, openRun, syncFormLimits } from './runs.js';
import { initShell } from './shell.js';
import { initTheme } from './theme.js';
import { $ } from './dom.js';

/** Both #app and #auth start hidden, so a throw anywhere in the wiring below
 *  leaves an empty page with nothing to go on. Say what broke instead, in
 *  styles of its own: whatever failed may well be the stylesheet. */
function reportStartupFailure(error) {
  const box = document.createElement('div');
  box.setAttribute('role', 'alert');
  box.style.cssText =
    'margin:16vh auto 0;max-width:34rem;padding:20px 22px;border:1px solid #e0788a;' +
    'border-radius:10px;background:#181c22;color:#e7eaee;font:15px/1.6 system-ui,sans-serif';
  box.innerHTML =
    '<b>The console could not start.</b><br>' +
    'Reload with <b>Ctrl+Shift+R</b> — this is usually a script left over from an ' +
    'older version of the page.';
  const detail = document.createElement('p');
  detail.style.cssText = 'margin:12px 0 0;color:#8a93a2;font:12px/1.5 ui-monospace,monospace';
  detail.textContent = String(error?.message || error);
  box.appendChild(detail);
  document.body.appendChild(box);
}

try {
  initTheme();
  initShell();
  initRuns();
  initReport();
  initHistory(openRun);
  initAuth(refreshRuns);
} catch (error) {
  reportStartupFailure(error);
  throw error;
}

getConfig()
  .then((cfg) => {
    renderConfig(cfg);
    syncFormLimits();
  })
  .catch(() => {
    $('env-model').textContent = 'unavailable';
  });

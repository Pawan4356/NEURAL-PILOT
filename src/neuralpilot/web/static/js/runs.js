import { createRun, getRun, submitClarification } from './api.js';
import { $, escapeHtml, hide, show } from './dom.js';
import { refreshRuns, setCurrentRun } from './history.js';
import { renderRun } from './renderers.js';
import { resetRunState, state, stopPolling } from './state.js';
import { setCrumb, showScreen } from './shell.js';

const POLL_MS = 900;
// A run in one of these states will not change on its own, so polling stops.
const SETTLED = ['finalized', 'failed', 'interrupted', 'awaiting_clarification'];

// ---------------------------------------------------------------------------
// New run form
// ---------------------------------------------------------------------------

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
}

/** Show the dataset's column names so the user can check the file before starting. */
async function previewColumns(file) {
  try {
    const header = (await file.slice(0, 8192).text()).split(/\r?\n/)[0];
    const columns = header.split(',').map((c) => c.replace(/^"|"$/g, '').trim()).filter(Boolean);
    const shown = columns.slice(0, 10).map((c) => `<code>${escapeHtml(c)}</code>`).join('');
    const more = columns.length > 10 ? `<code class="more">+${columns.length - 10} more</code>` : '';
    $('cols').innerHTML = `<span class="small" style="margin-right:4px">${columns.length} columns</span>${shown}${more}`;
    show('cols');
  } catch (error) {
    hide('cols'); // the preview is a convenience; the server still checks the file
  }
}

function takeFile(file) {
  $('drop').classList.add('is-filled');
  $('drop-t').textContent = file.name;
  $('drop-s').textContent = `${formatSize(file.size)}, ready to upload`;
  hide('new-error');
  previewColumns(file);
}

function setDropzoneDragging(on) {
  $('drop').classList.toggle('is-over', on);
}

function toggleSuggestion(button) {
  const goal = $('goal-input');
  const add = button.dataset.add;
  const on = button.getAttribute('aria-pressed') === 'true';
  goal.value = on
    ? goal.value.replace(` ${add}`, '').replace(add, '').trim()
    : `${goal.value.trim()} ${add}`.trim();
  button.setAttribute('aria-pressed', String(!on));
}

function showNewError(message) {
  $('new-error-t').textContent = message;
  show('new-error');
}

async function startRun(event) {
  event.preventDefault();
  const file = $('file').files[0];
  const goal = $('goal-input').value.trim();
  if (!file) return showNewError('Choose a CSV dataset first.');
  if (!goal) return showNewError('Describe what you want to predict.');

  hide('new-error');
  $('start-btn').disabled = true;
  try {
    const run = await createRun(file, goal);
    openRunScreen(run.run_id, goal);
    startPolling();
    refreshRuns();
  } catch (error) {
    showNewError(error.message);
  } finally {
    $('start-btn').disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Run screen
// ---------------------------------------------------------------------------

function clearRunScreen() {
  $('feed').innerHTML = '';
  hide('ask');
  hide('ask-error');
  hide('feed-foot');
  hide('run-error');
}

/** Put a run on the run screen, ready for polling. */
function openRunScreen(runId, goal) {
  resetRunState();
  state.currentRunId = runId;
  clearRunScreen();
  $('run-id').textContent = `#${runId.slice(0, 6)}`;
  $('run-goal').textContent = goal;
  $('report-goal').textContent = goal;
  setCrumb(`Run #${runId.slice(0, 6)}`);
  setCurrentRun(runId);
  showScreen('run');
}

/** Open a run from the rail. A run that is already finished goes straight to its
 *  report; the status is read from the server, not from the (possibly older) list row. */
export function openRun(run) {
  openRunScreen(run.run_id, run.goal);
  state.showReportWhenReady = true;
  startPolling();
}

/** Back to a blank form, dropping whatever run was open. */
export function startNewRun() {
  stopPolling();
  resetRunState();
  clearRunScreen();
  setCurrentRun(null);
  setCrumb('New run');
  showScreen('new');
}

async function fetchStatus() {
  try {
    const data = await getRun(state.currentRunId);
    renderRun(data);
    if (state.showReportWhenReady) {
      // Decided by the first answer only: a run still in progress when it was opened
      // stays on the run screen, even if it finishes later.
      if (data.status === 'finalized') showScreen('report');
      state.showReportWhenReady = false;
    }
    if (SETTLED.includes(data.status)) {
      stopPolling();
      refreshRuns();
    }
  } catch (error) {
    $('run-error-t').textContent = error.message;
    show('run-error');
  }
}

function startPolling() {
  stopPolling();
  state.pollHandle = setInterval(fetchStatus, POLL_MS);
  fetchStatus();
}

async function submitAnswer(event) {
  event.preventDefault();
  const answer = $('ask-in').value.trim();
  if (!answer) return;

  hide('ask-error');
  try {
    await submitClarification(state.currentRunId, answer);
    hide('ask');
    startPolling();
  } catch (error) {
    $('ask-error-t').textContent = error.message;
    show('ask-error');
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

export function initRuns() {
  $('drop').onclick = () => $('file').click();
  $('drop').onkeydown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') $('file').click();
  };
  ['dragenter', 'dragover'].forEach((name) =>
    $('drop').addEventListener(name, (event) => {
      event.preventDefault();
      setDropzoneDragging(true);
    }),
  );
  ['dragleave', 'drop'].forEach((name) =>
    $('drop').addEventListener(name, (event) => {
      event.preventDefault();
      setDropzoneDragging(false);
    }),
  );
  $('drop').addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $('file').files = transfer.files;
    takeFile(file);
  });
  $('file').onchange = (event) => {
    if (event.target.files[0]) takeFile(event.target.files[0]);
  };

  document.querySelectorAll('#suggest button').forEach((button) => {
    button.onclick = () => toggleSuggestion(button);
  });
  $('new-form').onsubmit = startRun;
  $('ask-form').onsubmit = submitAnswer;
  $('view-report-btn').onclick = () => showScreen('report');
  $('new-run-btn').onclick = startNewRun;
  $('brand-link').onclick = (event) => {
    event.preventDefault();
    startNewRun();
  };
}

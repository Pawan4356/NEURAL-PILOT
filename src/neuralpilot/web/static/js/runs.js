import { cancelRun, createRun, getRun, retryRun, submitClarification } from './api.js';
import { $, escapeHtml, hide, show } from './dom.js';
import { refreshRuns, setCurrentRun } from './history.js';
import { renderRun } from './renderers.js';
import { resetRunState, state, stopPolling } from './state.js';
import { closeRail, setCrumb, showScreen } from './shell.js';

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
  updateStartEnabled();
}

/** Start run needs both a dataset and a goal. */
function updateStartEnabled() {
  $('start-btn').disabled = !$('file').files[0] || !$('goal-input').value.trim();
}

/** Empty the new-run form: no dataset, no goal, default time limit. */
function resetNewForm() {
  $('new-form').reset();
  $('file').value = '';
  $('drop').classList.remove('is-filled');
  $('drop-t').textContent = 'Drop a CSV here, or browse';
  $('drop-s').textContent = 'One file, with a header row';
  $('cols').innerHTML = '';
  hide('cols');
  hide('new-error');
  document.querySelectorAll('#suggest button').forEach((button) => button.setAttribute('aria-pressed', 'false'));
  budgetMin = budgetBounds()[0];
  stepBudget(0);
  iterations = state.config?.max_iterations ?? 5;
  stepIterations(0);
  updateStartEnabled();
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

// Time limit, in minutes. The bounds come from /api/config once it has loaded.
const BUDGET_STEP_MIN = 15;
let budgetMin = 15;

function budgetBounds() {
  const cfg = state.config || {};
  return [(cfg.run_budget_min_s ?? 900) / 60, (cfg.run_budget_max_s ?? 7200) / 60];
}

function formatMinutes(total) {
  const h = Math.floor(total / 60);
  const m = total % 60;
  if (!h) return `${m} min`;
  return m ? `${h} h ${m} min` : `${h} h`;
}

function stepBudget(direction) {
  const [lo, hi] = budgetBounds();
  budgetMin = Math.min(hi, Math.max(lo, budgetMin + direction * BUDGET_STEP_MIN));
  $('budget-val').textContent = formatMinutes(budgetMin);
  $('budget-dec').disabled = budgetMin <= lo;
  $('budget-inc').disabled = budgetMin >= hi;
}

// Iteration cap. The bounds and default come from /api/config once it has loaded.
let iterations = 5;

function iterationBounds() {
  const cfg = state.config || {};
  return [cfg.max_iterations_min ?? 2, cfg.max_iterations_max ?? 10];
}

/** The form was drawn before /api/config arrived: apply its bounds and default. */
export function syncFormLimits() {
  iterations = state.config?.max_iterations ?? iterations;
  stepIterations(0);
  stepBudget(0);
}

function stepIterations(direction) {
  const [lo, hi] = iterationBounds();
  iterations = Math.min(hi, Math.max(lo, iterations + direction));
  $('iter-val').textContent = String(iterations);
  $('iter-dec').disabled = iterations <= lo;
  $('iter-inc').disabled = iterations >= hi;
}

/** A short, cartoonish launch: the button squashes and springs, then puffs and
 *  stars burst out of it and a rocket shoots off. Purely decorative. */
function playLaunch(button) {
  button.classList.remove('is-launching');
  void button.offsetWidth; // restart the animation on a quick second click
  button.classList.add('is-launching');
  const bits = ['✦', '★', '', '', '✦', '', '★', ''];
  bits.forEach((glyph, i) => {
    const bit = document.createElement('span');
    bit.className = glyph ? 'launch-bit is-star' : 'launch-bit';
    bit.textContent = glyph;
    const angle = (i / bits.length) * Math.PI * 2 + Math.random() * 0.4;
    const dist = 46 + Math.random() * 28;
    bit.style.setProperty('--dx', `${Math.cos(angle) * dist}px`);
    bit.style.setProperty('--dy', `${Math.sin(angle) * dist * 0.7}px`);
    bit.style.setProperty('--rot', `${Math.round(Math.random() * 360)}deg`);
    button.appendChild(bit);
  });
  const rocket = document.createElement('span');
  rocket.className = 'launch-rocket';
  rocket.textContent = '🚀';
  rocket.setAttribute('aria-hidden', 'true');
  button.appendChild(rocket);
  setTimeout(() => {
    button.querySelectorAll('.launch-bit, .launch-rocket').forEach((el) => el.remove());
    button.classList.remove('is-launching');
  }, 1000);
}

async function startRun(event) {
  event.preventDefault();
  const file = $('file').files[0];
  const goal = $('goal-input').value.trim();
  if (!file) return showNewError('Choose a CSV dataset first.');
  if (!goal) return showNewError('Describe what you want to predict.');

  hide('new-error');
  $('start-btn').disabled = true;
  playLaunch($('start-btn'));
  try {
    // Let the launch play out before the screen changes.
    const [run] = await Promise.all([
      createRun(file, goal, budgetMin, iterations),
      new Promise((resolve) => setTimeout(resolve, 650)),
    ]);
    openRunScreen(run.run_id, goal);
    startPolling();
    refreshRuns();
  } catch (error) {
    showNewError(error.message);
  } finally {
    updateStartEnabled();
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
  hide('retry-btn');
  hide('stop-btn');
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

/** Back to a blank form, dropping whatever run was open. Already on the form,
 *  this keeps what has been entered so far. */
export function startNewRun() {
  if (!document.querySelector('[data-screen="new"]').hidden) {
    closeRail();
    return;
  }
  resetNewForm();
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

/** Ask the run to stop after the trial in flight. Whatever it found so far is
 *  still scored, saved and reported, so this is a finish, not a failure. */
async function stopRun() {
  state.stopRequested = true;
  $('stop-btn').disabled = true;
  $('stop-btn').textContent = 'Stopping...';
  try {
    await cancelRun(state.currentRunId);
  } catch (error) {
    state.stopRequested = false;
    $('run-error-t').textContent = error.message;
    show('run-error');
  }
}

/** Run the same dataset and goal again, as a new run. */
async function retry() {
  const goal = $('run-goal').textContent; // the retry keeps the goal, with a new run ID
  $('retry-btn').disabled = true;
  try {
    const run = await retryRun(state.currentRunId);
    openRunScreen(run.run_id, goal);
    startPolling();
    refreshRuns();
  } catch (error) {
    $('run-error-t').textContent = error.message;
  } finally {
    $('retry-btn').disabled = false;
  }
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
  $('budget-dec').onclick = () => stepBudget(-1);
  $('budget-inc').onclick = () => stepBudget(1);
  $('iter-dec').onclick = () => stepIterations(-1);
  $('iter-inc').onclick = () => stepIterations(1);
  $('goal-input').addEventListener('input', updateStartEnabled);
  resetNewForm();
  $('new-form').onsubmit = startRun;
  $('ask-form').onsubmit = submitAnswer;
  $('stop-btn').onclick = stopRun;
  $('retry-btn').onclick = retry;
  $('view-report-btn').onclick = () => showScreen('report');
  $('new-run-btn').onclick = startNewRun;
  $('brand-link').onclick = (event) => {
    event.preventDefault();
    startNewRun();
  };
}

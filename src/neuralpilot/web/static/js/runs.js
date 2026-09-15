import { createRun, getRun, submitClarification } from './api.js';
import { $, hide, show } from './dom.js';
import { renderRun } from './renderers.js';
import { resetRunState, state, stopPolling } from './state.js';

function setFile(file) {
  if (!file) return;

  $('dropzone').classList.add('filled');
  $('drop-title').textContent = file.name;
  $('drop-sub').textContent = `${(file.size / 1024).toFixed(0)} KB · ready to upload`;
}

function showRunPage(runId, goal) {
  $('run-id').textContent = `#${runId.slice(0, 6)}`;
  $('crumb').innerHTML = `/ <b>run #${runId.slice(0, 6)}</b>`;
  $('run-goal-echo').textContent = goal;
  $('page-new').classList.remove('visible');
  $('page-run').classList.add('visible');
}

async function fetchStatus() {
  try {
    const data = await getRun(state.currentRunId);
    renderRun(data);
    if (['finalized', 'failed', 'awaiting_clarification'].includes(data.status)) {
      stopPolling();
    }
  } catch (error) {
    $('run-error').textContent = error.message;
    show('run-error');
  }
}

function startPolling() {
  stopPolling();
  state.pollHandle = setInterval(fetchStatus, 900);
  fetchStatus();
}

async function startRun() {
  const file = $('file').files[0];
  const goal = $('goal-input').value.trim();

  if (!file) {
    $('new-error').textContent = 'Choose a CSV dataset first.';
    show('new-error');
    return;
  }

  if (!goal) {
    $('new-error').textContent = 'Describe what you want to predict.';
    show('new-error');
    return;
  }

  hide('new-error');
  $('start-btn').disabled = true;

  try {
    const data = await createRun(file, goal);
    state.currentRunId = data.run_id;
    state.seenActivity = 0;
    showRunPage(data.run_id, goal);
    startPolling();
  } catch (error) {
    $('new-error').textContent = error.message;
    show('new-error');
  } finally {
    $('start-btn').disabled = false;
  }
}

async function clarifyRun() {
  const answer = $('clarify-answer').value.trim();
  if (!answer) return;

  await submitClarification(state.currentRunId, answer);
  hide('clarify-card');
  startPolling();
}

function returnToNewRun() {
  stopPolling();
  resetRunState();
  $('feed').innerHTML = '';
  $('stats').innerHTML = '';
  $('history').innerHTML = '';
  $('shap-list').innerHTML = '';
  $('results').classList.remove('shown');
  hide('run-error');
  hide('clarify-card');
  hide('no-shap');
  hide('fallback-note');
  $('page-run').classList.remove('visible');
  $('page-new').classList.add('visible');
  $('crumb').innerHTML = '/ <b>new run</b>';
}

export function initRuns() {
  $('dropzone').onclick = () => $('file').click();
  $('dropzone').onkeydown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') $('file').click();
  };
  $('file').onchange = (event) => setFile(event.target.files[0]);
  $('start-btn').onclick = startRun;
  $('clarify-btn').onclick = clarifyRun;
  $('back-btn').onclick = returnToNewRun;
}

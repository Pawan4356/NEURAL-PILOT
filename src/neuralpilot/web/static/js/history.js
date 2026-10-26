import { listRuns } from './api.js';
import { $, escapeHtml, hide, show } from './dom.js';
import { timeAgo } from './format.js';

// The rail only shows this user's runs; the server enforces that, this module just draws them.
let runsById = new Map();
let openHandler = null;

const STATUS = {
  running: { dot: 'is-running', text: 'Running' },
  awaiting_clarification: { dot: 'is-wait', text: 'Needs your input' },
  finalized: { dot: '', text: 'Finished' },
  failed: { dot: 'is-fail', text: 'Failed' },
  interrupted: { dot: 'is-muted', text: 'Interrupted' },
};

export function initHistory(onOpen) {
  openHandler = onOpen;
  $('runs-list').onclick = (event) => {
    const row = event.target.closest('.run-item');
    if (row) openHandler(runsById.get(row.dataset.runId));
  };
}

export async function refreshRuns() {
  try {
    renderRuns(await listRuns());
    hide('runs-error');
  } catch (error) {
    $('runs-error-t').textContent = error.message;
    show('runs-error');
  }
}

/** Mark one run as open in the rail (or none, when null). */
export function setCurrentRun(runId) {
  document.querySelectorAll('.run-item').forEach((row) => {
    row.setAttribute('aria-current', String(row.dataset.runId === runId));
  });
}

function renderRuns(runs) {
  runsById = new Map(runs.map((run) => [run.run_id, run]));
  $('runs-list').innerHTML = runs.map(rowHtml).join('');
  $('runs-empty').hidden = runs.length > 0;
}

function rowHtml(run) {
  const status = STATUS[run.status] || { dot: '', text: run.status };
  return `
    <button class="run-item" type="button" data-run-id="${escapeHtml(run.run_id)}">
      <span class="st ${status.dot}"></span>
      <span class="t">${escapeHtml(run.dataset_name)}</span>
      <span class="sc num"></span>
      <span class="s">${escapeHtml(status.text)} · ${escapeHtml(timeAgo(run.created_at))}</span>
    </button>`;
}

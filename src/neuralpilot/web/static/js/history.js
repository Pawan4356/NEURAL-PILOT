import { listRuns } from './api.js';
import { $, escapeHtml } from './dom.js';

// The list only shows this user's runs; the server enforces that, this module just draws them.
let runsById = new Map();
let openHandler = null;

export function initHistory(onOpen) {
  openHandler = onOpen;
  $('runs-list').onclick = (event) => {
    const row = event.target.closest('.run-row');
    if (row) openHandler(runsById.get(row.dataset.runId));
  };
}

export async function refreshRuns() {
  try {
    renderRuns(await listRuns());
    $('runs-error').classList.add('hidden');
  } catch (error) {
    $('runs-error').textContent = error.message;
    $('runs-error').classList.remove('hidden');
  }
}

function renderRuns(runs) {
  runsById = new Map(runs.map((run) => [run.run_id, run]));
  $('runs-list').innerHTML = runs.map(rowHtml).join('');
  $('runs-empty').classList.toggle('hidden', runs.length > 0);
}

function rowHtml(run) {
  return `
    <button class="run-row" type="button" data-run-id="${escapeHtml(run.run_id)}">
      <span class="run-main">
        <span>${escapeHtml(run.dataset_name)}</span>
        <span class="run-goal">${escapeHtml(run.goal)}</span>
        <span class="run-meta">${escapeHtml(formatTime(run.created_at))}</span>
      </span>
      <span class="chip">${escapeHtml(run.status.replaceAll('_', ' '))}</span>
    </button>`;
}

function formatTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
}

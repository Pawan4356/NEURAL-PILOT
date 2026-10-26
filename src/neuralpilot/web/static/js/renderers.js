import { pathChart } from './chart.js';
import { $, escapeHtml } from './dom.js';
import { formatMessage } from './format.js';
import { renderReport } from './report.js';
import { state } from './state.js';

// Chip class and label for each run status.
const STATUS_CHIP = {
  running: ['is-running', 'Running'],
  awaiting_clarification: ['is-wait', 'Needs your input'],
  finalized: ['is-done', 'Finished'],
  failed: ['is-fail', 'Failed'],
  interrupted: ['is-wait', 'Interrupted'],
};

// Engine steps (deterministic code) are drawn differently from agent reasoning.
// The log's tags are either generic ("agent", "engine") or already name the step
// ("Planning agent", "Execution engine"), so classify by the text.
function isEngineStep(tag) {
  return /engine|verif/i.test(tag);
}

function stepName(tag) {
  if (tag === 'agent') return 'Agent';
  if (tag === 'engine') return 'Engine';
  return tag;
}

export function renderConfig(cfg) {
  state.config = cfg;
  $('env-model').textContent = cfg.llm_model || cfg.llm_source || 'not configured';
  $('model-chip').title = `source: ${cfg.llm_source}${cfg.llm_key_configured ? '' : ' (no API key configured)'}`;
}

/** Append activity lines the timeline has not shown yet. */
export function renderActivity(data) {
  const activity = data.activity_log || [];
  const feed = $('feed');

  activity.slice(state.seenActivity).forEach((item) => {
    const line = document.createElement('li');
    line.className = `ev ev-enter ${isEngineStep(item.tag) ? 'is-engine' : 'is-agent'}`;
    line.innerHTML =
      `<span class="ev-time num">${escapeHtml(item.time)}</span>` +
      '<span class="ev-node"></span>' +
      `<div class="ev-body"><span class="ev-name">${escapeHtml(stepName(item.tag))}</span>` +
      `<p>${formatMessage(item.message)}</p></div>`;
    feed.appendChild(line);
  });

  state.seenActivity = activity.length;
  feed.scrollTop = feed.scrollHeight;
}

export function renderRun(data) {
  renderActivity(data);

  const running = data.status === 'running';
  const [chipClass, chipText] = STATUS_CHIP[data.status] || ['', data.status];
  $('run-chip').className = `chip ${chipClass}`;
  $('run-chip-t').textContent = chipText;
  $('working').hidden = !running;

  // `iteration` counts finished iterations; while one is in flight, show it as the current one.
  const max = data.max_iterations;
  $('p-iter').textContent = running ? `${Math.min(data.iteration + 1, max)} of ${max}` : `${data.iteration} of ${max}`;
  $('p-state').textContent = chipText;

  const scores = (data.iteration_logs || []).map((log) => Number(log.composite_score));
  $('p-best').textContent = scores.length ? Math.max(...scores).toFixed(3) : '-';
  $('path-live').innerHTML = pathChart({ scores, max, compact: true, live: running });

  $('ask').hidden = data.status !== 'awaiting_clarification';
  if (data.status === 'awaiting_clarification') {
    $('ask-q').textContent = data.clarification_question || 'Which column is the target?';
  }

  const broken = data.status === 'failed' || data.status === 'interrupted';
  $('run-error-t').textContent = data.error || 'The run failed.';
  $('run-error').hidden = !broken;

  $('feed-foot').hidden = data.status !== 'finalized';
  if (data.status === 'finalized' && data.final_report) {
    renderReport(data.final_report);
  }
}

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

// The log's tags are either generic or already name the step, so only the
// known short ones are given a display name.
const STEP_NAMES = {
  agent: 'Agent',
  engine: 'Engine',
  execute: 'Execute',
  plan: 'Plan',
  requirement: 'Requirement',
};

function stepName(tag) {
  return STEP_NAMES[tag] || tag;
}

/** One activity line. A line with `progress` is rewritten in place as the work
 *  advances, so the whole line is built here rather than appended to. */
function lineHtml(item) {
  const device = item.device
    ? `<span class="ev-dev">(${escapeHtml(item.device)})</span>`
    : '';
  const bar = item.progress ? progressHtml(item.progress) : '';
  return (
    `<span class="ev-time num">${escapeHtml(item.time)}</span>` +
    '<span class="ev-node"></span>' +
    `<div class="ev-body"><span class="ev-name">${escapeHtml(stepName(item.tag))}${device}</span>` +
    `<p>${formatMessage(item.message)}</p>${bar}</div>`
  );
}

function progressHtml({ done, total }) {
  const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  return (
    `<div class="ev-bar" role="progressbar" aria-valuenow="${done}" aria-valuemin="0" aria-valuemax="${total}">` +
    `<span class="ev-bar-fill" style="width:${percent}%"></span></div>` +
    `<span class="ev-bar-n num">${done}/${total}</span>`
  );
}

export function renderConfig(cfg) {
  state.config = cfg;
  $('env-model').textContent = cfg.llm_model || cfg.llm_source || 'not configured';
  $('model-chip').title = `source: ${cfg.llm_source}${cfg.llm_key_configured ? '' : ' (no API key configured)'}`;
}

/** Append activity lines the timeline has not shown yet, and refresh the last
 *  one if it is a progress line that has advanced since the previous poll. */
export function renderActivity(data) {
  const activity = data.activity_log || [];
  const feed = $('feed');
  // Follow new lines only while the reader is already at the bottom, so
  // scrolling up to read earlier steps is not undone by the next poll.
  const followTail = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 24;

  const shown = activity[state.seenActivity - 1];
  if (shown?.progress && feed.lastElementChild) {
    feed.lastElementChild.innerHTML = lineHtml(shown);
  }

  activity.slice(state.seenActivity).forEach((item) => {
    const line = document.createElement('li');
    line.className = `ev ev-enter ${isEngineStep(item.tag) ? 'is-engine' : 'is-agent'}`;
    line.innerHTML = lineHtml(item);
    feed.appendChild(line);
  });

  state.seenActivity = activity.length;
  if (followTail) feed.scrollTop = feed.scrollHeight;
}

export function renderRun(data) {
  renderActivity(data);

  const running = data.status === 'running';
  const [chipClass, chipText] = STATUS_CHIP[data.status] || ['', data.status];
  $('run-chip').className = `chip ${chipClass}`;
  $('run-chip-t').textContent = chipText;
  $('working').hidden = !running;

  // Stopping is only possible while the run is live, and only once: the request
  // is honoured after the trial in flight, so the button waits rather than repeats.
  $('stop-btn').hidden = !running;
  $('stop-btn').disabled = state.stopRequested;
  $('stop-btn').textContent = state.stopRequested ? 'Stopping...' : 'Stop run';

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

  // The same notice reports a failed run and a failed poll, but only a failed
  // run can be retried, so the button follows the status rather than the notice.
  const broken = data.status === 'failed' || data.status === 'interrupted';
  $('run-error-t').textContent = data.error || 'The run failed.';
  $('run-error').hidden = !broken;
  $('retry-btn').hidden = !broken;

  $('feed-foot').hidden = data.status !== 'finalized';
  if (data.status === 'finalized' && data.final_report) {
    renderReport(data.final_report);
  }
}

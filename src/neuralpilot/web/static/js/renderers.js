import { $, escapeHtml, hide, show } from './dom.js';
import { state } from './state.js';

function tagClass(tag) {
  if (tag === 'plan') return 'feed-tag plan';
  if (tag === 'execute') return 'feed-tag execute';
  if (tag === 'decision') return 'feed-tag decision';
  return 'feed-tag';
}

function formatTask(task) {
  return escapeHtml(task).replace(/(score [0-9.]+)/gi, '<b>$1</b>');
}

export function renderActivity(data) {
  const activity = data.activity_log || [];
  const feed = $('feed');

  activity.slice(state.seenActivity).forEach((item) => {
    const line = document.createElement('div');
    line.className = 'feed-line';
    line.innerHTML = `<span class="feed-time">${escapeHtml(item.time)}</span><span class="${tagClass(item.tag)}">${escapeHtml(item.tag)}</span><span class="feed-msg">${formatTask(item.message)}</span>`;
    feed.appendChild(line);
  });

  state.seenActivity = activity.length;
  feed.scrollTop = feed.scrollHeight;
}

export function renderRun(data) {
  $('max-iterations').textContent = data.max_iterations;
  renderActivity(data);
  $('stat-iter').textContent = `${data.iteration} / ${data.max_iterations}`;
  $('stat-state').textContent = data.status.replaceAll('_', ' ');
  $('live-label').textContent = data.status.replaceAll('_', ' ');
  $('live-dot').classList.toggle('pulse', data.status === 'running');

  const logs = data.iteration_logs || [];
  const last = logs.at(-1);
  $('stat-score').textContent = last ? Number(last.composite_score).toFixed(3) : '-';

  if (data.status === 'awaiting_clarification') {
    $('clarify-question').textContent = data.clarification_question;
    show('clarify-card');
  }

  if (data.status === 'failed') {
    $('run-error').textContent = data.error || 'The run failed.';
    show('run-error');
  }

  if (data.status === 'finalized' && data.final_report) {
    renderReport(data.final_report);
  }
}

export function renderReport(report) {
  if (state.lastReport === report) return;
  state.lastReport = report;
  hide('no-shap');
  hide('fallback-note');

  const metrics = report.metrics || {};
  const primaryMetric = metrics.f1 !== undefined
    ? ['Weighted F1', Number(metrics.f1).toFixed(3)]
    : metrics.r2 !== undefined
      ? ['R²', Number(metrics.r2).toFixed(3)]
      : ['Metric', Object.values(metrics)[0] ?? '-'];

  $('stats').innerHTML = [
    ['Model family', report.model_family],
    ['Composite score', Number(report.composite_score).toFixed(3), true],
    primaryMetric,
    ['Iterations run', `${(report.run_history || []).length} / ${$('max-iterations').textContent}`],
  ].map((stat) => `<div class="stat"><div class="stat-label">${escapeHtml(stat[0])}</div><div class="stat-val${stat[2] ? ' accent' : ''}">${escapeHtml(stat[1])}</div></div>`).join('');

  $('history').innerHTML = (report.run_history || []).map((history) => `<tr><td>${history.iteration}</td><td>${escapeHtml(history.weakest_block || '-')}</td><td>${Number(history.composite_score).toFixed(3)}</td><td><span class="decision-tag ${history.decision === 'accept' ? 'accept' : 'refine'}">${escapeHtml(history.decision)}</span></td></tr>`).join('');

  const features = report.shap_summary?.top_features || [];
  if (features.length) {
    const max = Math.max(...features.map((feature) => Number(feature.mean_abs_shap) || 0), 1);
    $('shap-list').innerHTML = features.map((feature) => `<div class="shap-item"><div class="shap-name" title="${escapeHtml(feature.feature)}">${escapeHtml(feature.feature)}</div><div class="shap-bar-track"><div class="shap-bar-fill" style="width:${((Number(feature.mean_abs_shap) || 0) / max * 100).toFixed(0)}%"></div></div><div class="shap-val">${Number(feature.mean_abs_shap).toFixed(2)}</div></div>`).join('');
  } else {
    $('shap-list').innerHTML = '';
    show('no-shap');
  }

  const assumptions = report.assumptions || [];
  if (assumptions.length) {
    $('fallback-note').innerHTML = `<b>Run notes:</b> ${assumptions.map(escapeHtml).join(' · ')}`;
    show('fallback-note');
  }

  $('download-link').href = `/api/runs/${report.run_id}/model`;
  $('json-btn').onclick = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${report.run_id}-report.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  $('results').classList.add('shown');
  setTimeout(() => $('results').scrollIntoView({ behavior: 'smooth', block: 'start' }), 250);
}

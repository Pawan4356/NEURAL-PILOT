import { $, escapeHtml, hide, show } from './dom.js';
import { state } from './state.js';

// A holdout score at or above this is almost always the target leaking into
// the features rather than a genuinely strong model, so the report says so.
const SUSPICIOUS_SCORE = 0.99;

function tagClass(tag) {
  if (tag === 'plan') return 'feed-tag plan';
  if (tag === 'execute') return 'feed-tag execute';
  if (tag === 'decision') return 'feed-tag decision';
  return 'feed-tag';
}

function formatTask(task) {
  return escapeHtml(task).replace(/(score [0-9.]+)/gi, '<b>$1</b>');
}

/** Never round a near-perfect score up to a flat 1.000 — the extra digits are
 *  the whole signal that something is wrong. */
function metric(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n >= SUSPICIOUS_SCORE ? n.toFixed(5) : n.toFixed(digits);
}

function seconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n >= 60 ? `${Math.floor(n / 60)}m ${Math.round(n % 60)}s` : `${n.toFixed(1)}s`;
}

/** Drop the ColumnTransformer prefix so the user sees their own column name. */
function featureName(name) {
  return String(name ?? '').replace(/^(numeric|categorical|remainder|text)__/, '');
}

export function renderConfig(cfg) {
  $('env-model').textContent = cfg.llm_model || cfg.llm_source || 'not configured';
  $('env-line').title = `source: ${cfg.llm_source}${cfg.llm_key_configured ? '' : ' (no API key configured)'}`;
  $('env-line').classList.toggle('warn', !cfg.llm_key_configured);
  $('max-iterations').textContent = cfg.max_iterations;

  const holdout = Math.round((cfg.holdout_fraction ?? 0.2) * 100);
  $('env-row').innerHTML = [
    ['split', `${100 - holdout}/${holdout}`],
    ['seed', cfg.seed],
    ['cv', `${cfg.cv_folds}-fold`],
    ['budget', seconds(cfg.run_budget_s)],
    ['max iterations', cfg.max_iterations],
  ]
    .map(([k, v]) => `<div class="chip">${escapeHtml(k)} <b>${escapeHtml(v)}</b></div>`)
    .join('');
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

  // `iteration` counts finished iterations, so while work is in flight the
  // user is really watching the next one: show that instead of a stale 0.
  const finished = data.iteration;
  const running = data.status === 'running';
  $('stat-iter').textContent = running
    ? `${Math.min(finished + 1, data.max_iterations)} of ${data.max_iterations} (in progress)`
    : `${finished} of ${data.max_iterations}`;
  $('stat-state').textContent = data.status.replaceAll('_', ' ');
  $('live-label').textContent = data.status.replaceAll('_', ' ');
  $('live-dot').classList.toggle('pulse', running);

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

function renderScoreWarning(report, primary, holdoutScore) {
  if (!Number.isFinite(holdoutScore) || holdoutScore < SUSPICIOUS_SCORE) {
    hide('score-warning');
    return;
  }
  const excluded = Object.keys(report.data?.excluded_columns || {});
  const excludedNote = excluded.length
    ? `Already excluded: ${excluded.map(escapeHtml).join(', ')}.`
    : 'No columns were excluded, so every original column reached the model.';
  $('score-warning').innerHTML =
    `<b>This score is suspiciously high.</b> Holdout ${escapeHtml(primary)} is ` +
    `<b>${metric(holdoutScore)}</b>. A score this close to perfect usually means the target can be ` +
    `derived from the features rather than genuinely predicted. Single columns that give the answer away ` +
    `are dropped automatically, but a <i>combination</i> of columns can still do it. ${excludedNote} ` +
    `Check the top features below before trusting this model.`;
  show('score-warning');
}

/** A forced finish is not one thing. Running out of wall-clock is an early
 *  stop the user should worry about; reaching the iteration cap is the run
 *  ending as configured. Saying "stopped early" for both is its own lie. */
function stopKind(report) {
  if (!report.final_decision_forced) return null;
  const why = String(report.final_decision_rationale || '');
  if (/budget|time|exhaust/i.test(why)) {
    return {
      className: 'warn-box',
      short: 'out of time',
      headline: 'The run ran out of time, it did not converge.',
      advice: 'The best attempt so far was kept and finalized — more time could still improve on it. Raise NEURALPILOT_RUN_BUDGET_S to let it finish.',
    };
  }
  if (/max_iterations|iteration_count/i.test(why)) {
    return {
      className: 'warn-box info',
      short: 'hit the limit',
      headline: 'The run finished because it hit its iteration limit.',
      advice: 'It stopped as configured rather than because it stopped improving. Raise NEURALPILOT_MAX_ITERATIONS to let it keep refining.',
    };
  }
  return {
    className: 'warn-box info',
    short: 'ended by a guardrail',
    headline: 'The run was ended by a guardrail.',
    advice: 'The best attempt so far was kept and finalized.',
  };
}

function renderStopReason(report) {
  const kind = stopKind(report);
  if (!kind) {
    hide('stop-reason');
    return;
  }
  const box = $('stop-reason');
  box.className = `${kind.className} hidden`;
  box.innerHTML =
    `<b>${escapeHtml(kind.headline)}</b> ${escapeHtml(report.final_decision_rationale || '')} ` +
    `${escapeHtml(kind.advice)}`;
  show('stop-reason');
}

function renderStats(report, primary, holdoutScore) {
  const cv = (report.cv_metrics || {})[primary] || {};
  const stopped = stopKind(report);
  const historyCount = (report.run_history || []).length;
  const cards = [
    { label: 'Model family', value: report.model_family },
    { label: 'Composite score', value: Number(report.composite_score).toFixed(3), accent: true },
    {
      label: `Holdout ${primary}`,
      value: metric(holdoutScore),
      sub: Number.isFinite(Number(cv.mean))
        ? `cross-validated ${metric(cv.mean)} ± ${Number(cv.std).toFixed(5)}`
        : null,
      warn: Number.isFinite(holdoutScore) && holdoutScore >= SUSPICIOUS_SCORE,
    },
    {
      label: 'Iterations run',
      value: `${historyCount} of ${$('max-iterations').textContent}`,
      sub: stopped ? stopped.short : null,
      warn: stopped?.className === 'warn-box',
    },
  ];

  $('stats').innerHTML = cards
    .map(
      (card) =>
        `<div class="stat"><div class="stat-label">${escapeHtml(card.label)}</div>` +
        `<div class="stat-val${card.accent ? ' accent' : ''}${card.warn ? ' warn' : ''}` +
        `${String(card.value).length > 16 ? ' long' : ''}">${escapeHtml(card.value)}</div>` +
        (card.sub ? `<div class="stat-sub">${escapeHtml(card.sub)}</div>` : '') +
        '</div>'
    )
    .join('');
}

function renderHistory(report) {
  const history = report.run_history || [];
  const best = report.best_signature;

  $('history').innerHTML = history
    .map((row) => {
      const plan = row.plan_summary || {};
      const isBest = row.signature && row.signature === best;
      const decision = String(row.decision || '-');
      const decisionClass = decision === 'accept' ? 'accept' : decision === 'stop' ? 'stop' : 'refine';
      return (
        `<tr${isBest ? ' class="is-best"' : ''}>` +
        `<td>${escapeHtml(row.iteration)}${isBest ? ' <span class="pill">best</span>' : ''}</td>` +
        `<td><span class="plan-family">${escapeHtml(plan.model_family || '-')}</span>` +
        `<span class="plan-source">${escapeHtml(row.plan_source || '')}</span></td>` +
        `<td>${escapeHtml(row.weakest_block || '-')}</td>` +
        `<td>${Number(row.composite_score).toFixed(3)}</td>` +
        `<td><span class="decision-tag ${decisionClass}">${escapeHtml(decision)}` +
        `${row.decision_forced ? ' (forced)' : ''}</span></td>` +
        '</tr>'
      );
    })
    .join('');

  const measured = Number(report.attempt_count);
  $('attempt-count').innerHTML = Number.isFinite(measured)
    ? `${measured} attempt${measured === 1 ? '' : 's'} measured`
    : '';

  // The table lists one row per iteration (the planner's plan). Challengers are
  // measured too, so the winner is sometimes not a row here — say so plainly
  // rather than leaving the user with scores that look worse than the headline.
  const shownBest = history.some((row) => row.signature === best);
  const notes = [];
  if (Number.isFinite(measured) && measured > history.length) {
    notes.push(
      `${measured} plans were measured across ${history.length} iteration(s); this table lists the planner's plan for each.`
    );
  }
  if (!shownBest && best) {
    notes.push(
      `The winning attempt (${String(best).slice(0, 8)}, iteration ${report.best_iteration}) was a challenger plan and is not a row above — see Run details for what it was.`
    );
  }
  $('history-note').textContent = notes.join(' ');
}

function renderShap(report) {
  const features = report.shap_summary?.top_features || [];
  if (!features.length) {
    $('shap-list').innerHTML = '';
    show('no-shap');
    return;
  }
  hide('no-shap');
  const max = Math.max(...features.map((feature) => Number(feature.mean_abs_shap) || 0), 1e-9);
  $('shap-list').innerHTML = features
    .map((feature) => {
      const width = ((Number(feature.mean_abs_shap) || 0) / max) * 100;
      return (
        `<div class="shap-item"><div class="shap-name" title="${escapeHtml(feature.feature)}">` +
        `${escapeHtml(featureName(feature.feature))}</div>` +
        `<div class="shap-bar-track"><div class="shap-bar-fill" style="width:${width.toFixed(0)}%"></div></div>` +
        `<div class="shap-val">${Number(feature.mean_abs_shap).toFixed(3)}</div></div>`
      );
    })
    .join('');
}

function renderFacts(report) {
  const data = report.data || {};
  const method = data.methodology || {};
  const resources = report.resources || {};
  const tuning = report.tuning || {};
  const plan = report.best_plan || {};

  const steps = [
    ...(plan.preprocessing || []).map((step) => step.component),
    ...(plan.feature_engineering || []).map((step) => step.name),
  ].filter(Boolean);

  const trialsShort =
    Number.isFinite(Number(tuning.completed_trials)) &&
    Number.isFinite(Number(tuning.requested_trials)) &&
    Number(tuning.completed_trials) < Number(tuning.requested_trials);

  const facts = [
    ['Rows', `${Number(data.n_rows ?? 0).toLocaleString()} (${Number(data.n_train ?? 0).toLocaleString()} train / ${Number(data.n_holdout ?? 0).toLocaleString()} holdout)`],
    ['Features used', data.n_features_used ?? '-'],
    ['Split', `${method.strategy || '-'}, ${method.n_folds ?? '-'}-fold CV, seed ${method.seed ?? '-'}`],
    ['Winning plan', `${report.model_family}${steps.length ? ` · ${steps.join(' → ')}` : ''}`],
    [
      'Tuning',
      tuning.tuner
        ? `${tuning.tuner} — ${tuning.completed_trials ?? '?'} of ${tuning.requested_trials ?? '?'} trials` +
          (tuning.stopped_reason && tuning.stopped_reason !== 'completed' ? ` (stopped: ${tuning.stopped_reason})` : '')
        : '-',
      trialsShort,
    ],
    ['Prediction latency', `${Number(report.latency_ms_per_1k).toFixed(1)} ms / 1k rows`],
    ['Peak memory', `${Number(resources.peak_rss_mb ?? 0).toFixed(0)} MB`],
    ['Time spent', `${seconds(resources.train_s)} training · ${seconds(resources.hpo_s)} tuning`],
  ];

  $('facts').innerHTML = facts
    .map(
      ([label, value, warn]) =>
        `<div class="fact"><div class="fact-label">${escapeHtml(label)}</div>` +
        `<div class="fact-val${warn ? ' warn' : ''}">${escapeHtml(value)}</div></div>`
    )
    .join('');

  const excluded = Object.entries(report.data?.excluded_columns || {});
  if (excluded.length) {
    $('excluded-note').innerHTML =
      `<b>Columns left out of the model:</b><ul>${excluded
        .map(([col, reason]) => `<li><code>${escapeHtml(col)}</code> — ${escapeHtml(reason)}</li>`)
        .join('')}</ul>`;
    show('excluded-note');
  } else {
    hide('excluded-note');
  }
}

/** Assumptions arrive as one flat list mixing "what we decided about your data"
 *  with engine bookkeeping, often repeating the same substitution once per
 *  iteration. Split them and collapse the repeats. */
function renderNotes(report) {
  const assumptions = report.assumptions || [];
  const aboutData = /excluded from features|target|task is|converted to numbers|class|objective|metric name/i;
  const groups = { 'About your data': new Map(), 'Engine notes': new Map() };

  assumptions.forEach((raw) => {
    const text = String(raw);
    const body = text.replace(/^iteration \d+:\s*/i, '');
    const key = aboutData.test(text) && !/^iteration \d+:/i.test(text) ? 'About your data' : 'Engine notes';
    groups[key].set(body, (groups[key].get(body) || 0) + 1);
  });

  $('notes-groups').innerHTML = Object.entries(groups)
    .filter(([, items]) => items.size)
    .map(
      ([title, items]) =>
        `<div class="note-box"><b>${escapeHtml(title)}</b><ul>${[...items.entries()]
          .map(([text, count]) => `<li>${escapeHtml(text)}${count > 1 ? ` <span class="pill">×${count}</span>` : ''}</li>`)
          .join('')}</ul></div>`
    )
    .join('');
}

export function renderReport(report) {
  if (state.lastReport === report) return;
  state.lastReport = report;

  const primary = report.primary_metric || 'score';
  const holdoutScore = Number((report.holdout_metrics || report.metrics || {})[primary]);

  renderScoreWarning(report, primary, holdoutScore);
  renderStopReason(report);
  renderStats(report, primary, holdoutScore);
  renderHistory(report);
  renderShap(report);
  renderFacts(report);
  renderNotes(report);

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

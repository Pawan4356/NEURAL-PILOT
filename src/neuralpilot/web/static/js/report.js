import { pathChart } from './chart.js';
import { $, escapeHtml, hide, show } from './dom.js';
import { featureName, metric, seconds, SUSPICIOUS_SCORE } from './format.js';
import { groupAssumptions, reportText, sentence, stopKind } from './report-text.js';
import { state } from './state.js';

// ---------------------------------------------------------------------------
// Warnings: what the report must say before the user trusts the number.
// ---------------------------------------------------------------------------

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
    `<svg class="i"><use href="#i-alert"/></svg><span><b>This score is suspiciously high.</b> Holdout ${escapeHtml(primary)} is ` +
    `<b>${metric(holdoutScore)}</b>. A score this close to perfect usually means the target can be ` +
    `derived from the features rather than genuinely predicted. Single columns that give the answer away ` +
    `are dropped automatically, but a <i>combination</i> of columns can still do it. ${excludedNote} ` +
    'Check the top features below before trusting this model.</span>';
  show('score-warning');
}

function renderStopReason(report) {
  const kind = stopKind(report);
  if (!kind) {
    hide('stop-reason');
    return;
  }
  const box = $('stop-reason');
  box.className = kind.warn ? 'notice is-warn' : 'notice';
  box.innerHTML =
    `<svg class="i"><use href="#i-info"/></svg><span><b>${escapeHtml(kind.headline)}</b> ${escapeHtml(sentence(report.final_decision_rationale))} ` +
    `${escapeHtml(kind.advice)}</span>`;
  show('stop-reason');
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function renderHero(report, primary, holdoutScore) {
  const cv = (report.cv_metrics || {})[primary] || {};
  const history = report.run_history || [];
  const maxIterations = state.config?.max_iterations ?? history.length;

  $('r-family').textContent = report.model_family || '-';
  $('r-score').textContent = Number(report.composite_score).toFixed(3);
  $('r-holdout-label').textContent = `Holdout ${primary}`;
  $('r-holdout').textContent = metric(holdoutScore);
  $('r-holdout-sub').textContent = Number.isFinite(Number(cv.mean))
    ? `cross-validated ${metric(cv.mean)} ± ${Number(cv.std).toFixed(5)}`
    : '';
  $('r-latency').textContent = `${Number(report.latency_ms_per_1k).toFixed(1)} ms per 1k rows`;
  $('r-memory').textContent = `${Number(report.resources?.peak_rss_mb ?? 0).toFixed(0)} MB`;
  $('iter-used').textContent = `${history.length} of ${maxIterations} iterations used`;

  const scores = history.map((row) => Number(row.composite_score));
  $('path-report').innerHTML = pathChart({ scores, max: maxIterations, animate: true });
}

function renderHistory(report) {
  const history = report.run_history || [];
  const best = report.best_signature;

  $('history').innerHTML = history
    .map((row) => {
      const plan = row.plan_summary || {};
      const isBest = row.signature && row.signature === best;
      const decision = String(row.decision || '-');
      const verdict = decision === 'accept' ? 'accept' : decision === 'stop' ? 'stop' : 'refine';
      return (
        `<tr${isBest ? ' class="is-best"' : ''}>` +
        `<td class="num">${escapeHtml(row.iteration)}${isBest ? ' <span class="pill">best</span>' : ''}</td>` +
        `<td><span class="fam">${escapeHtml(plan.model_family || '-')}</span>` +
        `<span class="src">${escapeHtml(row.plan_source || '')}</span></td>` +
        `<td>${escapeHtml(row.weakest_block || '-')}</td>` +
        `<td class="num">${Number(row.composite_score).toFixed(3)}</td>` +
        `<td><span class="verdict ${verdict}">${escapeHtml(decision)}` +
        `${row.decision_forced ? ' (forced)' : ''}</span></td>` +
        '</tr>'
      );
    })
    .join('');

  const measured = Number(report.attempt_count);
  $('attempt-count').textContent = Number.isFinite(measured)
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
      `The winning attempt (${String(best).slice(0, 8)}, iteration ${report.best_iteration}) was a challenger plan and is not a row above — see Data and plan for what it was.`
    );
  }
  $('history-note').textContent = notes.join(' ');
  $('history-note-box').hidden = notes.length === 0;
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
    .map((feature, i) => {
      const width = ((Number(feature.mean_abs_shap) || 0) / max) * 100;
      return (
        `<div class="shap-row"><span class="nm" title="${escapeHtml(feature.feature)}">` +
        `${escapeHtml(featureName(feature.feature))}</span>` +
        `<span class="tr"><i style="width:${width.toFixed(0)}%;animation-delay:${i * 35}ms"></i></span>` +
        `<span class="v num">${Number(feature.mean_abs_shap).toFixed(3)}</span></div>`
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
    ['Time spent', `${seconds(resources.train_s)} training · ${seconds(resources.hpo_s)} tuning`],
  ];

  $('facts').innerHTML = facts
    .map(
      ([label, value, warn]) =>
        `<div class="${warn ? 'is-warn' : ''}"><dt>${escapeHtml(label)}</dt><dd class="num">${escapeHtml(value)}</dd></div>`
    )
    .join('');

  const excluded = Object.entries(report.data?.excluded_columns || {});
  if (excluded.length) {
    $('excluded-note').innerHTML =
      `<b>Columns left out of the model</b><ul>${excluded
        .map(([col, reason]) => `<li><code>${escapeHtml(col)}</code> — ${escapeHtml(reason)}</li>`)
        .join('')}</ul>`;
    show('excluded-note');
  } else {
    hide('excluded-note');
  }
}

function renderNotes(report) {
  const { about, engine } = groupAssumptions(report.assumptions);

  const items = (map) =>
    [...map.entries()]
      .map(([text, count]) => `<li>${escapeHtml(text)}${count > 1 ? ` <span class="pill">×${count}</span>` : ''}</li>`)
      .join('');

  let html = '';
  if (about.size) html += `<h3>What it decided about your data</h3><ul>${items(about)}</ul>`;
  if (engine.size) {
    html += `<details><summary>Engine bookkeeping (${engine.size})</summary><ul>${items(engine)}</ul></details>`;
  }
  $('notes-groups').innerHTML = html || '<p class="small">No notes for this run.</p>';
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

export function renderReport(report) {
  if (state.lastReport === report) return;
  state.lastReport = report;

  const primary = report.primary_metric || 'score';
  const holdoutScore = Number((report.holdout_metrics || report.metrics || {})[primary]);

  $('report-id').textContent = `#${String(report.run_id).slice(0, 6)}`;
  $('download-link').href = `/api/runs/${report.run_id}/model`;

  renderScoreWarning(report, primary, holdoutScore);
  renderStopReason(report);
  renderHero(report, primary, holdoutScore);
  renderHistory(report);
  renderShap(report);
  renderFacts(report);
  renderNotes(report);
}

/** The report as text opens in a dialog, so it can be read and copied in one
 *  click without downloading anything. */
function initTextDialog() {
  const dialog = $('text-dlg');
  $('text-btn').onclick = () => {
    // The goal and the iteration cap belong to the page, not to the report payload.
    $('text-pre').textContent = reportText(state.lastReport, {
      goal: $('report-goal').textContent,
      maxIterations: state.config?.max_iterations,
    });
    dialog.showModal();
  };
  ['text-x', 'text-done'].forEach((id) => {
    $(id).onclick = () => dialog.close();
  });
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  $('text-copy').onclick = async (event) => {
    try {
      await navigator.clipboard.writeText($('text-pre').textContent);
      event.target.textContent = 'Copied';
      setTimeout(() => (event.target.textContent = 'Copy'), 1500);
    } catch (error) {
      // Clipboard access can be blocked; the text is still on screen to copy by hand.
      event.target.textContent = 'Press Ctrl+C';
      setTimeout(() => (event.target.textContent = 'Copy'), 2500);
    }
  };
}

/** Tabs for the report sections, with arrow-key navigation. */
function initTabs() {
  const tabs = [...document.querySelectorAll('.tabs [role=tab]')];
  const activate = (tab) => {
    tabs.forEach((other) => {
      const on = other === tab;
      other.setAttribute('aria-selected', String(on));
      other.tabIndex = on ? 0 : -1;
      $(other.getAttribute('aria-controls')).hidden = !on;
    });
    tab.focus();
  };
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {
      const i = tabs.indexOf(tab);
      if (event.key === 'ArrowRight') activate(tabs[(i + 1) % tabs.length]);
      if (event.key === 'ArrowLeft') activate(tabs[(i - 1 + tabs.length) % tabs.length]);
    });
  });
}

export function initReport() {
  initTextDialog();
  initTabs();
}

/** The report as plain text, for pasting into a ticket, an email or a notebook.
 *
 *  It carries the same findings as the report screen, in reading order, so a
 *  pasted report is not a weaker account of the run than the page it came from.
 *
 *  The readings of the report that both the screen and the text need live here
 *  rather than in `report.js`: why the run stopped, how the assumptions group,
 *  how a rationale joins the advice after it. The two must say the same thing,
 *  and this is the module they can share without the text builder reaching
 *  into the DOM.
 */

import { featureName, metric, seconds, SUSPICIOUS_SCORE } from './format.js';

/** A forced finish is not one thing. Running out of wall-clock is an early
 *  stop the user should worry about; reaching the iteration cap is the run
 *  ending as configured. Saying "stopped early" for both is its own lie. */
export function stopKind(report) {
  if (!report.final_decision_forced) return null;
  const why = String(report.final_decision_rationale || '');
  if (/budget|time|exhaust/i.test(why)) {
    return {
      warn: true,
      headline: 'The run ran out of time, it did not converge.',
      advice: 'The best attempt so far was kept and finalized — more time could still improve on it. Start a new run with a longer time limit to let it finish.',
    };
  }
  if (/max_iterations|iteration_count/i.test(why)) {
    return {
      warn: false,
      headline: 'The run finished because it hit its iteration limit.',
      advice: 'It stopped as configured rather than because it stopped improving. Start a new run with more iterations to let it keep refining.',
    };
  }
  return {
    warn: false,
    headline: 'The run was ended by a guardrail.',
    advice: 'The best attempt so far was kept and finalized.',
  };
}

/** A rationale comes from the engine without final punctuation, and the advice
 *  that follows it is a sentence of its own. Join them as sentences, not as a
 *  run-on line. */
export function sentence(text) {
  const trimmed = String(text ?? '').trim();
  return !trimmed || /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

/** Assumptions arrive as one flat list mixing "what we decided about your data"
 *  with engine bookkeeping, often repeating the same substitution once per
 *  iteration. Split them and collapse the repeats into counts. */
export function groupAssumptions(assumptions) {
  const aboutData = /excluded from features|target|task is|converted to numbers|class|objective|metric name/i;
  const about = new Map();
  const engine = new Map();

  (assumptions || []).forEach((raw) => {
    const text = String(raw);
    const body = text.replace(/^iteration \d+:\s*/i, '');
    const group = aboutData.test(text) && !/^iteration \d+:/i.test(text) ? about : engine;
    group.set(body, (group.get(body) || 0) + 1);
  });
  return { about, engine };
}

// ---------------------------------------------------------------------------
// Text building blocks. The dialog shows the text in a monospace block, so
// columns are aligned with spaces rather than left to wrap where they fall.
// ---------------------------------------------------------------------------

const INDENT = '  ';

function heading(title) {
  return `${title}\n${'-'.repeat(title.length)}`;
}

/** `[label, value]` pairs as an aligned block. A pair with an empty value is
 *  dropped, so a report missing a section does not print blank labels. */
function fields(pairs) {
  const kept = pairs.filter(([, value]) => value !== '' && value !== null && value !== undefined);
  const width = Math.max(0, ...kept.map(([label]) => label.length));
  return kept.map(([label, value]) => `${INDENT}${label.padEnd(width)}  ${value}`).join('\n');
}

/** A header row plus body rows, each column as wide as its widest cell. */
function table(header, rows) {
  const widths = header.map((cell, i) =>
    Math.max(cell.length, ...rows.map((row) => String(row[i] ?? '').length)),
  );
  const line = (cells) =>
    INDENT +
    cells
      .map((cell, i) => String(cell ?? '').padEnd(widths[i]))
      .join('  ')
      .trimEnd();
  return [line(header), ...rows.map(line)].join('\n');
}

function bullets(lines, depth = 1) {
  return lines.map((line) => `${INDENT.repeat(depth)}- ${line}`).join('\n');
}

/** Wrap prose at a width that survives being pasted into a comment box. */
function wrap(text, { width = 76, indent = '' } = {}) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  words.forEach((word) => {
    const last = lines[lines.length - 1];
    if (last && `${last} ${word}`.length <= width - indent.length) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  });
  return lines.map((line) => indent + line).join('\n');
}

/** Values that arrive as objects (library versions, for one) read better as
 *  `name value` pairs than as the JSON they are stored in. */
function flatValue(value) {
  if (value === null || typeof value !== 'object') return String(value);
  if (Array.isArray(value)) return value.join(', ');
  return Object.entries(value)
    .map(([name, inner]) => `${name} ${inner}`)
    .join(', ');
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function headerSection(report, goal) {
  const target = report.target_column
    ? `${report.target_column}${report.problem_type ? ` (${report.problem_type})` : ''}`
    : '';
  return [
    'NEURALPILOT RUN REPORT',
    fields([
      ['Run', `#${String(report.run_id ?? '').slice(0, 6)}`],
      ['Dataset', report.dataset_name || ''],
      ['Goal', goal || ''],
      ['Target', target],
    ]),
  ].join('\n');
}

function resultSection(report, primary, holdoutScore, maxIterations) {
  const cv = (report.cv_metrics || {})[primary] || {};
  const used = (report.run_history || []).length;
  const decision = report.final_decision
    ? `${report.final_decision}${report.final_decision_forced ? ' (forced)' : ''} — ` +
      sentence(report.final_decision_rationale)
    : '';
  return [
    heading('RESULT'),
    fields([
      ['Model', report.model_family || '-'],
      ['Composite score', Number(report.composite_score).toFixed(3)],
      [`Holdout ${primary}`, metric(holdoutScore)],
      [
        'Cross-validated',
        Number.isFinite(Number(cv.mean)) ? `${metric(cv.mean)} +/- ${Number(cv.std).toFixed(5)}` : '',
      ],
      ['Latency', `${Number(report.latency_ms_per_1k).toFixed(1)} ms per 1k rows`],
      ['Peak memory', `${Number(report.resources?.peak_rss_mb ?? 0).toFixed(0)} MB`],
      ['Iterations used', `${used} of ${maxIterations ?? used}`],
      ['Decision', decision],
    ]),
  ].join('\n');
}

/** The same two things the screen refuses to let the number stand alone with:
 *  a score too good to be true, and a run that was cut short. */
function warningsSection(report, primary, holdoutScore) {
  const parts = [];
  if (Number.isFinite(holdoutScore) && holdoutScore >= SUSPICIOUS_SCORE) {
    const excluded = Object.keys(report.data?.excluded_columns || {});
    parts.push(
      `WARNING: this score is suspiciously high.\n${wrap(
        `Holdout ${primary} is ${metric(holdoutScore)}. A score this close to perfect usually means the ` +
          'target can be derived from the features rather than genuinely predicted. Single columns that ' +
          'give the answer away are dropped automatically, but a combination of columns can still do it. ' +
          (excluded.length
            ? `Already excluded: ${excluded.join(', ')}.`
            : 'No columns were excluded, so every original column reached the model.') +
          ' Check the top features below before trusting this model.',
      )}`,
    );
  }
  const kind = stopKind(report);
  if (kind) {
    parts.push(
      `NOTE: ${kind.headline}\n${wrap(`${sentence(report.final_decision_rationale)} ${kind.advice}`)}`,
    );
  }
  return parts.join('\n\n');
}

function metricsSection(report) {
  const holdout = report.holdout_metrics || report.metrics || {};
  const entries = Object.entries(holdout).filter(([, value]) => Number.isFinite(Number(value)));
  if (!entries.length) return '';
  return [
    heading('HOLDOUT METRICS'),
    fields(entries.map(([name, value]) => [name, metric(value)])),
  ].join('\n');
}

function dataSection(report) {
  const data = report.data || {};
  const method = data.methodology || {};
  const resources = report.resources || {};
  const tuning = report.tuning || {};
  const plan = report.best_plan || {};
  const steps = [
    ...(plan.preprocessing || []).map((step) => step.component),
    ...(plan.feature_engineering || []).map((step) => step.name),
  ].filter(Boolean);

  const block = [
    heading('DATA AND PLAN'),
    fields([
      [
        'Rows',
        `${Number(data.n_rows ?? 0).toLocaleString()} (${Number(data.n_train ?? 0).toLocaleString()} train / ` +
          `${Number(data.n_holdout ?? 0).toLocaleString()} holdout)`,
      ],
      ['Features used', data.n_features_used ?? '-'],
      ['Split', `${method.strategy || '-'}, ${method.n_folds ?? '-'}-fold CV, seed ${method.seed ?? '-'}`],
      ['Winning plan', `${report.model_family}${steps.length ? ` · ${steps.join(' -> ')}` : ''}`],
      [
        'Tuning',
        tuning.tuner
          ? `${tuning.tuner} — ${tuning.completed_trials ?? '?'} of ${tuning.requested_trials ?? '?'} trials` +
            (tuning.stopped_reason && tuning.stopped_reason !== 'completed'
              ? ` (stopped: ${tuning.stopped_reason})`
              : '')
          : '',
      ],
      ['Time spent', `${seconds(resources.train_s)} training · ${seconds(resources.hpo_s)} tuning`],
    ]),
  ];

  const hyperparameters = Object.entries(report.best_hyperparameters || {});
  if (hyperparameters.length) {
    block.push('', `${INDENT}Winning hyperparameters`);
    block.push(fields(hyperparameters.map(([name, value]) => [`${INDENT}${name}`, String(value)])));
  }

  const excluded = Object.entries(report.data?.excluded_columns || {});
  if (excluded.length) {
    block.push('', `${INDENT}Columns left out of the model`);
    block.push(bullets(excluded.map(([column, reason]) => `${column} — ${reason}`), 2));
  }
  return block.join('\n');
}

/** Challengers beside an iteration's planner plan, e.g. "LinearModel 0.812 *; ExtraTrees skipped". */
function challengerText(row, best) {
  const extra = (row.candidates || []).filter((c) => c.source !== 'planner');
  if (!extra.length) return 'none';
  return extra
    .map((c) => {
      const result =
        c.status === 'ok' ? Number(c.composite_score).toFixed(3) : c.status === 'failed' ? 'failed' : 'skipped';
      const mark = c.signature && c.signature === best ? ' *' : '';
      return `${c.model_family || '-'} ${result}${mark}`;
    })
    .join('; ');
}

function iterationsSection(report) {
  const history = report.run_history || [];
  if (!history.length) return '';

  const best = report.best_signature;
  const rows = history.map((row) => [
    `${row.iteration}${row.signature === best ? ' *' : ''}`,
    (row.plan_summary || {}).model_family || '-',
    row.plan_source || '',
    challengerText(row, best),
    row.weakest_block || '-',
    Number(row.composite_score).toFixed(3),
    `${row.decision || '-'}${row.decision_forced ? ' (forced)' : ''}`,
  ]);

  const block = [
    heading('ITERATIONS'),
    table(['#', 'Model', 'Source', 'Challengers', 'Weakest block', 'Score', 'Decision'], rows),
  ];

  // The table lists one row per iteration (the planner's plan). Challengers are
  // measured too, so the winner is sometimes not a row here — say so plainly
  // rather than leaving the reader with scores that look worse than the headline.
  const measured = Number(report.attempt_count);
  const notes = [];
  if (Number.isFinite(measured) && measured > history.length) {
    notes.push(
      `${measured} plans were measured across ${history.length} iteration(s); this table lists the planner's plan for each.`,
    );
  }
  if (best && !history.some((row) => row.signature === best)) {
    notes.push(
      `The winning attempt (${String(best).slice(0, 8)}, iteration ${report.best_iteration}) was a challenger plan and is not a row above.`,
    );
  } else if (best) {
    notes.push('* marks the winning attempt.');
  }
  if (notes.length) block.push('', wrap(notes.join(' '), { indent: INDENT }));
  return block.join('\n');
}

function featuresSection(report) {
  const features = report.shap_summary?.top_features || [];
  if (!features.length) {
    return [heading('TOP FEATURES'), `${INDENT}Not available for this run.`].join('\n');
  }
  return [
    `${heading('TOP FEATURES')}\n${INDENT}${
      report.shap_summary?.method === 'model_importance'
        ? "By the model's own feature importance (SHAP could not be computed)."
        : 'By mean |SHAP|, on training rows.'
    }`,
    fields(
      features.map((feature) => [featureName(feature.feature), Number(feature.mean_abs_shap).toFixed(3)]),
    ),
  ].join('\n');
}

function notesSection(report) {
  const { about, engine } = groupAssumptions(report.assumptions);
  if (!about.size && !engine.size) return '';

  const items = (map) =>
    bullets(
      [...map.entries()].map(([text, count]) => (count > 1 ? `${text} (x${count})` : text)),
      2,
    );

  const block = [heading('NOTES')];
  if (about.size) block.push(`${INDENT}What it decided about your data`, items(about));
  if (engine.size) {
    if (about.size) block.push('');
    block.push(`${INDENT}Engine bookkeeping (${engine.size})`, items(engine));
  }
  return block.join('\n');
}

function reproducibilitySection(report) {
  const repro = report.reproducibility || {};
  const entries = Object.entries(repro).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length && !report.mlflow_run_id) return '';
  return [
    heading('REPRODUCIBILITY'),
    fields([
      ...entries.map(([name, value]) => [name.replace(/_/g, ' '), flatValue(value)]),
      ['mlflow run', report.mlflow_run_id || ''],
    ]),
  ].join('\n');
}

/** The whole report as text. `goal` and `maxIterations` are not part of the
 *  report payload; the page knows them and passes them in. */
export function reportText(report, { goal = '', maxIterations } = {}) {
  if (!report) return 'No report yet.';
  const primary = report.primary_metric || 'score';
  const holdoutScore = Number((report.holdout_metrics || report.metrics || {})[primary]);

  return [
    headerSection(report, goal),
    warningsSection(report, primary, holdoutScore),
    resultSection(report, primary, holdoutScore, maxIterations),
    metricsSection(report),
    dataSection(report),
    iterationsSection(report),
    featuresSection(report),
    notesSection(report),
    reproducibilitySection(report),
  ]
    .filter(Boolean)
    .join('\n\n');
}

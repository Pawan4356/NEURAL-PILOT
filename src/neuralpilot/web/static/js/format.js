import { escapeHtml } from './dom.js';

// A holdout score at or above this is almost always the target leaking into
// the features rather than a genuinely strong model, so the report says so.
export const SUSPICIOUS_SCORE = 0.99;

/** Never round a near-perfect score up to a flat 1.000 — the extra digits are
 *  the whole signal that something is wrong. */
export function metric(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n >= SUSPICIOUS_SCORE ? n.toFixed(5) : n.toFixed(digits);
}

export function seconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n >= 60 ? `${Math.floor(n / 60)}m ${Math.round(n % 60)}s` : `${n.toFixed(1)}s`;
}

/** Drop the ColumnTransformer prefix so the user sees their own column name. */
export function featureName(name) {
  return String(name ?? '').replace(/^(numeric|categorical|remainder|text)__/, '');
}

/** Escape a log message, then bold any score it quotes. */
export function formatMessage(text) {
  return escapeHtml(text).replace(/(score [0-9.]+)/gi, '<b>$1</b>');
}

/** "3 minutes ago" style text for the run list. */
export function timeAgo(iso, now = Date.now()) {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const minutes = Math.max(0, Math.round((now - then) / 60000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'yesterday' : `${days} days ago`;
}

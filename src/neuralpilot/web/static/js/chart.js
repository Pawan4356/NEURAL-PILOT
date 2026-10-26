// Composite score by iteration, drawn as SVG. The vertical scale is fitted to
// the scores it shows, so small differences between iterations stay visible.

/** Round the scores' range outward to steps of 0.02, with room to breathe. */
function domain(scores) {
  if (!scores.length) return { lo: 0.7, hi: 0.95 };
  const lo = Math.max(0, Math.floor((Math.min(...scores) - 0.02) * 50) / 50);
  const hi = Math.min(1, Math.max(lo + 0.06, Math.ceil((Math.max(...scores) + 0.01) * 50) / 50));
  return { lo, hi };
}

export function pathChart({ scores, max, compact = false, animate = false, live = false }) {
  const W = compact ? 268 : 600;
  const H = compact ? 130 : 230;
  const pl = compact ? 30 : 40;
  const pr = compact ? 14 : 22;
  const pt = compact ? 18 : 28;
  const pb = 28;
  const inset = compact ? 8 : 22;
  const { lo, hi } = domain(scores);
  const X = (i) => pl + inset + (i * (W - pl - pr - inset)) / Math.max(max - 1, 1);
  const Y = (v) => pt + (1 - (v - lo) / (hi - lo)) * (H - pt - pb);
  const best = scores.length ? scores.indexOf(Math.max(...scores)) : -1;

  const label = scores.length
    ? scores.map((v, i) => `iteration ${i + 1}, ${v.toFixed(3)}`).join('; ')
    : 'no scores yet';
  let s = `<svg class="path${animate ? ' draw' : ''}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Composite score by iteration: ${label}">`;

  [0, 1, 2, 3].forEach((k) => {
    const v = lo + (k * (hi - lo)) / 3;
    s += `<line class="grid" x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}"/>` +
      `<text x="${pl - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  });

  for (let i = 0; i < max; i++) {
    s += `<line class="slot" x1="${X(i)}" x2="${X(i)}" y1="${pt - 6}" y2="${H - pb}"/>` +
      `<text x="${X(i)}" y="${H - 8}" text-anchor="middle">${i + 1}</text>`;
  }

  if (scores.length > 1) {
    const d = scores.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
    s += `<path class="line" pathLength="1" d="${d}"/>`;
  }

  // Iterations not run yet are shown as empty slots.
  for (let i = scores.length; i < max; i++) {
    s += `<circle class="ghost" cx="${X(i)}" cy="${(pt + (H - pb)) / 2 + 20}" r="4"/>`;
  }

  scores.forEach((v, i) => {
    const isBest = i === best && !live;
    if (isBest) s += `<circle class="halo" cx="${X(i)}" cy="${Y(v)}" r="13"/>`;
    s += `<circle class="pt${isBest ? ' best' : ''}" cx="${X(i)}" cy="${Y(v)}" r="${isBest ? 6.5 : 5}">` +
      `<title>Iteration ${i + 1}: ${v.toFixed(3)}</title></circle>`;
    if (!compact || i === scores.length - 1) {
      s += `<text class="val${isBest ? ' best' : ''}" x="${X(i)}" y="${Y(v) - (isBest ? 16 : 12)}" text-anchor="middle">${v.toFixed(3)}</text>`;
    }
  });

  if (live && scores.length) {
    const last = scores.length - 1;
    s += `<circle class="live" cx="${X(last)}" cy="${Y(scores[last])}" r="6"/>`;
  }
  return `${s}</svg>`;
}

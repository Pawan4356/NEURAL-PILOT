/** The rounded token that walks the four steps of the sign-in diagram: squeezed
 *  thin while it slides along the line, jiggling as it lands, then on to the
 *  next — and from the last step back to the first, which is the loop the
 *  diagram describes.
 *
 *  Purely decorative: it stays out of the accessibility tree, parks on the
 *  engine step instead of moving when the viewer prefers reduced motion, and
 *  stops whenever the sign-in page is off screen, since its timers would
 *  otherwise run for the whole session.
 */

// Kept in step with the durations in app.css (--token-travel / --token-jiggle).
const TRAVEL_MS = 520;
const JIGGLE_MS = 560;
const DWELL_MS = 620;
const ENGINE_STEP = 2; // "Verify and execute" — where the square rests when still

let timer = null;
let step = 0;
let wired = false;

function parts() {
  const flow = document.querySelector('.route .flow');
  const token = flow?.querySelector('.token');
  const markers = flow ? [...flow.querySelectorAll('ol .n')] : [];
  return token && markers.length ? { flow, token, markers } : null;
}

/** Centre the token on one marker. The rows are sized by their own text, so the
 *  markers are not evenly spaced and their offsets have to be measured. */
function centreOn({ flow, token, markers }, index) {
  const marker = markers[index];
  const rect = marker?.getBoundingClientRect();
  if (!rect?.height) return; // the panel is hidden at narrow widths
  const y = rect.top - flow.getBoundingClientRect().top + (rect.height - token.offsetHeight) / 2;
  token.style.setProperty('--token-y', `${Math.round(y)}px`);
}

/** Mark which step the token is sitting on; that outline turns square and
 *  jiggles with it. Nothing is active while it is between steps. */
function setActive(ctx, index) {
  ctx.markers.forEach((marker, i) => marker.classList.toggle('is-active', i === index));
}

/** Hand the travelling token over to the marker's own square and shake.
 *
 *  Driven by the transition actually finishing rather than by a timer: a timer
 *  fires a frame or two after the transform has already settled, and that gap
 *  reads as the square going still just before it jiggles. The timer is kept as
 *  a fallback for when transitionend never arrives. */
function land(ctx) {
  if (!ctx.flow.classList.contains('is-travelling')) return; // already landed
  clearTimeout(timer);
  ctx.flow.classList.remove('is-travelling');
  setActive(ctx, step); // square first, so the outline jiggles in that shape
  ctx.flow.classList.add('is-landing');
  timer = setTimeout(() => {
    ctx.flow.classList.remove('is-landing');
    timer = setTimeout(() => tick(ctx), DWELL_MS);
  }, JIGGLE_MS);
}

/** Travel to the next step, land, pause, repeat. Re-adding the class is what
 *  restarts each animation. */
function tick(ctx) {
  step = (step + 1) % ctx.markers.length;
  setActive(ctx, -1); // the outline it is leaving rounds off again
  ctx.flow.classList.add('is-travelling');
  centreOn(ctx, step);
  timer = setTimeout(() => land(ctx), TRAVEL_MS + 150);
}

/** One listener for the life of the page; `land` ignores anything that is not
 *  the end of a journey. */
function wire(ctx) {
  if (wired) return;
  wired = true;
  ctx.token.addEventListener('transitionend', (event) => {
    if (event.propertyName === 'transform') land(ctx);
  });
}

/** Put the token on a step with no travel animation, for the first paint and
 *  for re-measuring after a resize. */
function place(ctx, index) {
  ctx.flow.classList.add('is-placing');
  ctx.flow.classList.remove('is-travelling', 'is-landing');
  centreOn(ctx, index);
  setActive(ctx, index);
  // Let the jump land before transitions are allowed again.
  requestAnimationFrame(() => ctx.flow.classList.remove('is-placing'));
}

export function startRouteLoop() {
  stopRouteLoop();
  const ctx = parts();
  if (!ctx) return;
  wire(ctx);

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    step = ENGINE_STEP;
    place(ctx, step);
    return;
  }

  step = 0;
  place(ctx, step);
  timer = setTimeout(() => tick(ctx), DWELL_MS);
}

export function stopRouteLoop() {
  clearTimeout(timer);
  timer = null;
}

// Row heights change when the text rewraps, so the current step is re-measured
// rather than left at a stale offset. Mid-travel the tick owns the position, and
// correcting it here would cut the journey short.
window.addEventListener('resize', () => {
  const ctx = parts();
  if (ctx && !ctx.flow.classList.contains('is-travelling')) centreOn(ctx, step);
});

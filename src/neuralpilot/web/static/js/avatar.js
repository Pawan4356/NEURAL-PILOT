// Every letter owns one hue, so the same account always gets the same avatar.
// The hues step by the golden angle, which keeps neighbouring letters apart
// (a -> red, b -> green, c -> violet ...). The drifting cloud gradient itself
// lives in CSS; here we only hand it the hue.

const STEP = 137.508;
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';

/** Hue in degrees for one character; anything unknown falls back to its code. */
export function hueFor(char) {
  const lower = String(char || '').toLowerCase();
  const index = ALPHABET.indexOf(lower);
  const slot = index >= 0 ? index : lower.charCodeAt(0) || 0;
  return Math.round(slot * STEP) % 360;
}

/** Paint `text`'s first character into an avatar element with its own gradient. */
export function paintAvatar(element, text) {
  const initial = String(text || '?').trim().charAt(0) || '?';
  element.innerHTML = '';
  const letter = document.createElement('span');
  letter.textContent = initial;
  element.appendChild(letter);
  element.style.setProperty('--av-h', String(hueFor(initial)));
}

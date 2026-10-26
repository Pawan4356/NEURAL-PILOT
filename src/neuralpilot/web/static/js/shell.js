import { $, hide, show } from './dom.js';

// The app shell (rail + main column) and the sign-in page are never shown together.
// Visibility here is UI only: the server enforces access to every run.

export function showAuth() {
  hide('app');
  show('auth');
}

export function showApp(email) {
  $('who-email').textContent = email;
  $('avatar').textContent = email.charAt(0);
  hide('auth');
  show('app');
  showScreen('new');
}

/** Show one of the main screens: 'new', 'run' or 'report'. */
export function showScreen(name) {
  document.querySelectorAll('[data-screen]').forEach((screen) => {
    screen.hidden = screen.dataset.screen !== name;
  });
  closeRail();
  window.scrollTo(0, 0);
}

export function setCrumb(text) {
  $('crumb').textContent = text;
}

export function closeRail() {
  $('app').classList.remove('rail-open');
}

export function initShell() {
  $('menu-btn').onclick = () => $('app').classList.add('rail-open');
  $('scrim').onclick = closeRail;
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeRail();
  });
}

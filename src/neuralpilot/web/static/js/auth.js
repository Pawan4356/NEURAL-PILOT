import { getSession, logIn, logOut, SIGNED_OUT_EVENT, signUp } from './api.js';
import { $, hide, show } from './dom.js';
import { stopPolling } from './state.js';

// Shows the sign-in page until the server confirms a session, and hands control
// to `onSignedIn` once it does. Visibility here is only UI: the server enforces access.
let mode = 'login';
let onSignedIn = () => {};

export function initAuth(handler) {
  onSignedIn = handler;
  $('auth-btn').onclick = submitCredentials;
  $('auth-toggle').onclick = () => setMode(mode === 'login' ? 'signup' : 'login');
  $('auth-password').onkeydown = (event) => {
    if (event.key === 'Enter') submitCredentials();
  };
  $('logout-btn').onclick = logOutAndReload;
  window.addEventListener(SIGNED_OUT_EVENT, showSignedOut);

  setMode('login');
  getSession()
    .then((user) => showSignedIn(user.email))
    .catch(showSignedOut);
}

export function showSignedOut() {
  stopPolling();
  hide('session-box');
  $('page-new').classList.remove('visible');
  $('page-run').classList.remove('visible');
  $('page-auth').classList.add('visible');
}

function showSignedIn(email) {
  $('session-email').textContent = email;
  show('session-box');
  $('page-auth').classList.remove('visible');
  $('page-new').classList.add('visible');
  onSignedIn();
}

function setMode(next) {
  mode = next;
  const signingUp = mode === 'signup';
  $('auth-title').textContent = signingUp ? 'Sign up' : 'Log in';
  $('auth-btn').textContent = signingUp ? 'Create account →' : 'Log in →';
  $('auth-toggle').textContent = signingUp ? 'Have an account? Log in' : 'No account? Sign up';
  $('auth-password').setAttribute('autocomplete', signingUp ? 'new-password' : 'current-password');
  $('auth-hint').classList.toggle('hidden', !signingUp);
  hide('auth-error');
}

async function submitCredentials() {
  const email = $('auth-email').value.trim();
  const password = $('auth-password').value;
  hide('auth-error');
  $('auth-btn').disabled = true;
  try {
    const user = mode === 'signup' ? await signUp(email, password) : await logIn(email, password);
    $('auth-password').value = '';
    showSignedIn(user.email);
  } catch (error) {
    $('auth-error').textContent = error.message;
    show('auth-error');
  } finally {
    $('auth-btn').disabled = false;
  }
}

async function logOutAndReload() {
  try {
    await logOut();
  } finally {
    // A reload drops every piece of in-page state (the open run, the list, the form).
    window.location.reload();
  }
}

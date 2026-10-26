import { getSession, logIn, logOut, SIGNED_OUT_EVENT, signUp } from './api.js';
import { $, hide, show } from './dom.js';
import { showAuth, showApp } from './shell.js';
import { stopPolling } from './state.js';

const MIN_PASSWORD = 8;
let mode = 'login';
let onSignedIn = () => {};

export function initAuth(handler) {
  onSignedIn = handler;
  $('a-form').onsubmit = (event) => {
    event.preventDefault();
    submitCredentials();
  };
  $('a-toggle').onclick = () => setMode(mode === 'login' ? 'signup' : 'login');
  $('a-eye').onclick = togglePasswordVisibility;
  $('logout-btn').onclick = logOutAndReload;
  window.addEventListener(SIGNED_OUT_EVENT, showSignedOut);

  setMode('login');
  getSession()
    .then((user) => enter(user.email))
    .catch(showSignedOut);
}

export function showSignedOut() {
  stopPolling();
  showAuth();
}

function enter(email) {
  showApp(email);
  onSignedIn();
}

function setMode(next) {
  mode = next;
  const signingUp = mode === 'signup';
  // innerHTML, not textContent: the wordmark carries an <i> that textContent
  // would strip. Both strings are literals here — nothing user-supplied.
  $('a-title').innerHTML = signingUp ? 'Create your account' : 'Sign in to Neural<i>Pilot</i>';
  $('a-sub').textContent = signingUp
    ? 'Runs are private to your account, so you can come back to them any time.'
    : 'Start a run or reopen your previous ones. Each account sees only its own runs.';
  $('a-btn').textContent = signingUp ? 'Create account' : 'Log in';
  $('a-q').textContent = signingUp ? 'Already have an account?' : 'No account?';
  $('a-toggle').textContent = signingUp ? 'Log in' : 'Sign up';
  $('a-pw').setAttribute('autocomplete', signingUp ? 'new-password' : 'current-password');
  $('a-hint').hidden = !signingUp;
  hide('a-error');
}

function togglePasswordVisibility() {
  const input = $('a-pw');
  const reveal = input.type === 'password';
  input.type = reveal ? 'text' : 'password';
  $('a-eye').setAttribute('aria-label', reveal ? 'Hide password' : 'Show password');
}

function checkForm(email, password) {
  if (!/^\S+@\S+\.\S+$/.test(email)) return 'Enter a valid email address.';
  if (mode === 'signup' && password.length < MIN_PASSWORD) {
    return `Use a password with at least ${MIN_PASSWORD} characters.`;
  }
  return '';
}

async function submitCredentials() {
  const email = $('a-email').value.trim();
  const password = $('a-pw').value;
  const problem = checkForm(email, password);
  if (problem) {
    $('a-error-t').textContent = problem;
    show('a-error');
    return;
  }

  hide('a-error');
  $('a-btn').disabled = true;
  try {
    const user = mode === 'signup' ? await signUp(email, password) : await logIn(email, password);
    $('a-pw').value = '';
    enter(user.email);
  } catch (error) {
    $('a-error-t').textContent = error.message;
    show('a-error');
  } finally {
    $('a-btn').disabled = false;
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

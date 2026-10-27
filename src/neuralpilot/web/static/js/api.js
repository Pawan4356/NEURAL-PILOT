// A 401 means the session is missing or expired. Announce it once so the page
// can show the sign-in form, and still throw so the caller can show its own message.
export const SIGNED_OUT_EVENT = 'np:signed-out';

async function parseJsonResponse(response, fallbackMessage) {
  const data = await response.json();
  if (response.status === 401) {
    window.dispatchEvent(new Event(SIGNED_OUT_EVENT));
  }
  if (!response.ok) {
    throw Error(data.detail || fallbackMessage);
  }
  return data;
}

function formOf(fields) {
  const body = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    body.append(key, value);
  }
  return body;
}

export async function getSession() {
  const response = await fetch('/api/auth/me');
  return parseJsonResponse(response, 'Not signed in');
}

export async function logIn(email, password) {
  const response = await fetch('/api/auth/login', { method: 'POST', body: formOf({ email, password }) });
  return parseJsonResponse(response, 'Could not log in');
}

export async function signUp(email, password) {
  const response = await fetch('/api/auth/signup', { method: 'POST', body: formOf({ email, password }) });
  return parseJsonResponse(response, 'Could not create account');
}

export async function logOut() {
  const response = await fetch('/api/auth/logout', { method: 'POST' });
  return parseJsonResponse(response, 'Could not log out');
}

export async function listRuns() {
  const response = await fetch('/api/runs');
  return parseJsonResponse(response, 'Could not load your runs');
}

export async function createRun(file, goal, budgetMin) {
  const body = new FormData();
  body.append('file', file);
  body.append('goal', goal);
  body.append('budget_min', String(budgetMin));

  const response = await fetch('/api/runs', { method: 'POST', body });
  return parseJsonResponse(response, 'Failed to start run');
}

export async function getRun(runId) {
  const response = await fetch(`/api/runs/${runId}`);
  return parseJsonResponse(response, 'Could not load run');
}

export async function submitClarification(runId, answer) {
  const body = new FormData();
  body.append('answer', answer);

  const response = await fetch(`/api/runs/${runId}/clarify`, { method: 'POST', body });
  return parseJsonResponse(response, 'Could not submit clarification');
}

export async function cancelRun(runId) {
  const response = await fetch(`/api/runs/${runId}/cancel`, { method: 'POST' });
  return parseJsonResponse(response, 'Could not stop the run');
}

/** Start a new run from the same dataset and goal. Returns the new run. */
export async function retryRun(runId) {
  const response = await fetch(`/api/runs/${runId}/retry`, { method: 'POST' });
  return parseJsonResponse(response, 'Could not retry the run');
}

export async function getConfig() {
  const response = await fetch('/api/config');
  return parseJsonResponse(response, 'Could not load configuration');
}

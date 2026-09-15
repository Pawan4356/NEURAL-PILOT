async function parseJsonResponse(response, fallbackMessage) {
  const data = await response.json();
  if (!response.ok) {
    throw Error(data.detail || fallbackMessage);
  }
  return data;
}

export async function createRun(file, goal) {
  const body = new FormData();
  body.append('file', file);
  body.append('goal', goal);

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

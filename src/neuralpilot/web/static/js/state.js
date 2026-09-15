export const state = {
  currentRunId: null,
  pollHandle: null,
  seenActivity: 0,
  lastReport: null,
};

export function resetRunState() {
  state.currentRunId = null;
  state.lastReport = null;
  state.seenActivity = 0;
}

export function stopPolling() {
  if (state.pollHandle) {
    clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

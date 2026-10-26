export const state = {
  config: null, // server settings from /api/config (budget, iterations, split)
  currentRunId: null,
  pollHandle: null,
  seenActivity: 0,
  lastReport: null,
  showReportWhenReady: false, // set when a finished run is opened from the list
};

export function resetRunState() {
  state.currentRunId = null;
  state.lastReport = null;
  state.seenActivity = 0;
  state.showReportWhenReady = false;
}

export function stopPolling() {
  if (state.pollHandle) {
    clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

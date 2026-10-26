export const $ = (id) => document.getElementById(id);

// `hidden` (not a class) so the [hidden] rule in the stylesheet always wins.
export function show(elOrId) {
  const el = typeof elOrId === 'string' ? $(elOrId) : elOrId;
  el.hidden = false;
}

export function hide(elOrId) {
  const el = typeof elOrId === 'string' ? $(elOrId) : elOrId;
  el.hidden = true;
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

/** Every lookup here is a contract with index.html. When one breaks, name the
 *  element: "Cannot read properties of null" names nothing, and the usual
 *  cause — a script left over from an older version of the page — is invisible
 *  from that message alone. */
export const $ = (id) => {
  const element = document.getElementById(id);
  if (!element) {
    throw Error(`The page is missing #${id}. Reload with Ctrl+Shift+R to pick up the current scripts.`);
  }
  return element;
};

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

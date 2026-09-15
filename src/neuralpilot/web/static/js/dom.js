export const $ = (id) => document.getElementById(id);

export function show(id) {
  $(id).classList.remove('hidden');
}

export function hide(id) {
  $(id).classList.add('hidden');
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

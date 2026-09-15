import { $ } from './dom.js';

const STORAGE_KEY = 'neuralpilot-theme';

export function setTheme(mode) {
  document.documentElement.dataset.theme = mode;
  $('btn-dark').classList.toggle('active', mode === 'dark');
  $('btn-light').classList.toggle('active', mode === 'light');
  localStorage.setItem(STORAGE_KEY, mode);
  
  const logo = document.querySelector('.brand img');
  if (logo) {
    logo.src = `/logo.svg?theme=${mode}`;
  }
}

export function initTheme() {
  $('btn-dark').onclick = () => setTheme('dark');
  $('btn-light').onclick = () => setTheme('light');

  const savedTheme = localStorage.getItem(STORAGE_KEY);
  if (savedTheme) {
    setTheme(savedTheme);
  }
}

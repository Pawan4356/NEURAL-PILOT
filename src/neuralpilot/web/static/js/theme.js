const STORAGE_KEY = 'neuralpilot-theme';
const REVEAL_MS = 550;

function iconFor(mode) {
  // The button shows the mode it switches *to*.
  return mode === 'dark' ? 'i-sun' : 'i-moon';
}

function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  const next = mode === 'dark' ? 'light' : 'dark';
  document.querySelectorAll('.theme-btn').forEach((button) => {
    button.innerHTML = `<svg class="i"><use href="#${iconFor(mode)}"/></svg>`;
    button.setAttribute('aria-label', `Switch to ${next} theme`);
  });
  // The logo is recoloured by the server for each theme.
  document.querySelectorAll('[data-logo]').forEach((logo) => {
    logo.src = `/logo.svg?theme=${mode}`;
  });
}

function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

function revealFromButton(event, mode) {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!document.startViewTransition || reduceMotion) {
    applyTheme(mode);
    return;
  }

  const rect = event.currentTarget.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const radius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y),
  );

  const transition = document.startViewTransition(() => applyTheme(mode));
  transition.ready.then(() => {
    document.documentElement.animate(
      {
        clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`],
      },
      {
        duration: REVEAL_MS,
        easing: 'ease-in-out',
        pseudoElement: '::view-transition-new(root)',
      },
    );
  });
}

export function initTheme() {
  const savedTheme = localStorage.getItem(STORAGE_KEY);
  applyTheme(savedTheme === 'light' ? 'light' : 'dark');

  document.querySelectorAll('.theme-btn').forEach((button) => {
    button.onclick = (event) => {
      const mode = currentTheme() === 'dark' ? 'light' : 'dark';
      localStorage.setItem(STORAGE_KEY, mode);
      revealFromButton(event, mode);
    };
  });
}

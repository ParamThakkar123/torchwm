// Remove duplicate search buttons injected by the theme to keep the header tidy.
document.addEventListener('DOMContentLoaded', () => {
  try {
    const buttons = Array.from(document.querySelectorAll('.search-button-field'));
    if (buttons.length <= 1) return;
    // Keep the first visible search button, remove others.
    let kept = null;
    for (const b of buttons) {
      if (!kept && b.offsetParent !== null) {
        kept = b;
        continue;
      }
      b.remove();
    }
  } catch (e) {
    // swallow errors
    console.warn('fix_navbar.js error', e);
  }
});

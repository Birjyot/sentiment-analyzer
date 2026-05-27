# ui/dotfield_bg.py
"""Animated DotField canvas background.

Embeds via st.components.v1.html(). It self-positions its iframe as a 
fixed background layer and listens for mouse events on the parent window.
"""

import streamlit as st
import streamlit.components.v1 as components

# ── SELF-CONTAINED HTML + JS ──────────────────────────────────────────────────
_DOTFIELD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: transparent; overflow: hidden; }
  canvas { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
(function(){
  /* ── Self-position the Streamlit iframe to the background ── */
  try {
    const frame = window.frameElement;
    if (frame) {
      frame.style.position = 'fixed';
      frame.style.top = '0';
      frame.style.left = '0';
      frame.style.width = '100vw';
      frame.style.height = '100vh';
      frame.style.zIndex = '0';
      frame.style.border = 'none';
      frame.style.pointerEvents = 'none';
      
      const wrapper = frame.parentElement;
      if (wrapper) {
        wrapper.style.position = 'fixed';
        wrapper.style.top = '0';
        wrapper.style.left = '0';
        wrapper.style.width = '100vw';
        wrapper.style.height = '100vh';
        wrapper.style.zIndex = '0';
        wrapper.style.pointerEvents = 'none';
      }
    }
  } catch(e) {}

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  const TWO_PI = Math.PI * 2;

  // React Bits config
  const DOT_RADIUS   = 1.5;
  const DOT_SPACING  = 14;
  const CURSOR_R     = 500;
  const BULGE_STR    = 67;
  const GRAD_FROM    = 'rgba(168, 85, 247, 0.45)';
  const GRAD_TO      = 'rgba(180, 151, 207, 0.35)';

  let dots = [];
  let W = 0, H = 0;
  let mouseX = -9999, mouseY = -9999;
  let prevX = -9999, prevY = -9999;
  let speed = 0, engagement = 0;

  function buildDots(w, h) {
    const step = DOT_RADIUS + DOT_SPACING;
    const cols = Math.floor(w / step);
    const rows = Math.floor(h / step);
    const padX = (w % step) / 2;
    const padY = (h % step) / 2;
    dots = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const ax = padX + c * step + step / 2;
        const ay = padY + r * step + step / 2;
        dots.push({ ax, ay, sx: ax, sy: ay });
      }
    }
  }

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    buildDots(W, H);
  }

  function updateSpeed() {
    const dx = prevX - mouseX;
    const dy = prevY - mouseY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    speed += (dist - speed) * 0.5;
    if (speed < 0.001) speed = 0;
    prevX = mouseX;
    prevY = mouseY;
  }
  setInterval(updateSpeed, 20);

  // Listen on parent window because iframe pointerEvents = 'none'
  try {
    if (window.parent && window.parent !== window) {
      window.parent.addEventListener('mousemove', e => {
        mouseX = e.clientX;
        mouseY = e.clientY;
      }, { passive: true });
    }
  } catch(e) {}

  // Local fallback
  window.addEventListener('mousemove', e => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  }, { passive: true });

  function tick() {
    const crSq = CURSOR_R * CURSOR_R;
    const rad  = DOT_RADIUS / 2;

    const targetEng = Math.min(speed / 5, 1);
    engagement += (targetEng - engagement) * 0.06;
    if (engagement < 0.001) engagement = 0;

    ctx.clearRect(0, 0, W, H);

    const grad = ctx.createLinearGradient(0, 0, W, H);
    grad.addColorStop(0, GRAD_FROM);
    grad.addColorStop(1, GRAD_TO);
    ctx.fillStyle = grad;

    ctx.beginPath();
    for (let i = 0; i < dots.length; i++) {
      const d = dots[i];
      const dx = mouseX - d.ax;
      const dy = mouseY - d.ay;
      const dSq = dx * dx + dy * dy;

      if (dSq < crSq && engagement > 0.01) {
        const dist  = Math.sqrt(dSq);
        const t     = 1 - dist / CURSOR_R;
        const push  = t * t * BULGE_STR * engagement;
        const angle = Math.atan2(dy, dx);
        d.sx += (d.ax - Math.cos(angle) * push - d.sx) * 0.15;
        d.sy += (d.ay - Math.sin(angle) * push - d.sy) * 0.15;
      } else {
        d.sx += (d.ax - d.sx) * 0.1;
        d.sy += (d.ay - d.sy) * 0.1;
      }

      ctx.moveTo(d.sx + rad, d.sy);
      ctx.arc(d.sx, d.sy, rad, 0, TWO_PI);
    }
    ctx.fill();
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', resize);
  resize();
  tick();
})();
</script>
</body>
</html>"""

# ── CSS TO MAKE STREAMLIT TRANSPARENT ─────────────────────────────────────────
_BG_CSS = """
<style>
/* Make Streamlit background transparent so dots show through */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background: transparent !important;
}
/* Ensure the UI elements sit above the dot background */
section.main {
    position: relative !important;
    z-index: 1 !important;
}
</style>
"""

def render_dotfield_background():
    st.markdown(_BG_CSS, unsafe_allow_html=True)
    components.html(_DOTFIELD_HTML, height=0, scrolling=False)

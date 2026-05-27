# ui/shapegrid_bg.py
"""Animated ShapeGrid canvas background.

Embeds via st.components.v1.html(). It self-positions its iframe as a 
fixed background layer and listens for mouse events on the parent window.
"""

import streamlit as st
import streamlit.components.v1 as components

# ── SELF-CONTAINED HTML + JS ──────────────────────────────────────────────────
_SHAPEGRID_HTML = """<!DOCTYPE html>
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

  // React Bits config
  const direction = 'diagonal';
  const speed = 0.5;
  const borderColor = 'rgba(255, 255, 255, 0.05)'; // Subtle premium white borders
  const squareSize = 40;
  const hoverFillColor = 'rgba(99, 102, 241, 0.2)'; // Premium hover fill (indigo matching the app)
  const hoverTrailAmount = 5;

  let gridOffset = { x: 0, y: 0 };
  let hoveredSquare = null;
  let trailCells = [];
  let cellOpacities = new Map();

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const offsetX = ((gridOffset.x % squareSize) + squareSize) % squareSize;
    const offsetY = ((gridOffset.y % squareSize) + squareSize) % squareSize;
    const cols = Math.ceil(canvas.width / squareSize) + 3;
    const rows = Math.ceil(canvas.height / squareSize) + 3;

    for (let col = -2; col < cols; col++) {
      for (let row = -2; row < rows; row++) {
        const sx = col * squareSize + offsetX;
        const sy = row * squareSize + offsetY;

        const cellKey = `${col},${row}`;
        const alpha = cellOpacities.get(cellKey);
        if (alpha) {
          ctx.globalAlpha = alpha;
          ctx.fillStyle = hoverFillColor;
          ctx.fillRect(sx, sy, squareSize, squareSize);
          ctx.globalAlpha = 1;
        }

        ctx.strokeStyle = borderColor;
        ctx.strokeRect(sx, sy, squareSize, squareSize);
      }
    }

    const gradient = ctx.createRadialGradient(
      canvas.width / 2, canvas.height / 2, 0,
      canvas.width / 2, canvas.height / 2,
      Math.sqrt(canvas.width ** 2 + canvas.height ** 2) / 2
    );
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function updateAnimation() {
    const effectiveSpeed = Math.max(speed, 0.1);
    const wrapX = squareSize;
    const wrapY = squareSize;

    if (direction === 'diagonal') {
      gridOffset.x = (gridOffset.x - effectiveSpeed + wrapX) % wrapX;
      gridOffset.y = (gridOffset.y - effectiveSpeed + wrapY) % wrapY;
    }

    updateCellOpacities();
    drawGrid();
    requestAnimationFrame(updateAnimation);
  }

  function updateCellOpacities() {
    const targets = new Map();

    if (hoveredSquare) {
      targets.set(`${hoveredSquare.x},${hoveredSquare.y}`, 1);
    }

    if (hoverTrailAmount > 0) {
      for (let i = 0; i < trailCells.length; i++) {
        const t = trailCells[i];
        const key = `${t.x},${t.y}`;
        if (!targets.has(key)) {
          targets.set(key, (trailCells.length - i) / (trailCells.length + 1));
        }
      }
    }

    for (const [key] of targets) {
      if (!cellOpacities.has(key)) {
        cellOpacities.set(key, 0);
      }
    }

    for (const [key, opacity] of cellOpacities) {
      const target = targets.get(key) || 0;
      const next = opacity + (target - opacity) * 0.15;
      if (next < 0.005) {
        cellOpacities.delete(key);
      } else {
        cellOpacities.set(key, next);
      }
    }
  }

  function handleMouseMove(mouseX, mouseY) {
    const offsetX = ((gridOffset.x % squareSize) + squareSize) % squareSize;
    const offsetY = ((gridOffset.y % squareSize) + squareSize) % squareSize;

    const adjustedX = mouseX - offsetX;
    const adjustedY = mouseY - offsetY;

    const col = Math.floor(adjustedX / squareSize);
    const row = Math.floor(adjustedY / squareSize);

    if (!hoveredSquare || hoveredSquare.x !== col || hoveredSquare.y !== row) {
      if (hoveredSquare && hoverTrailAmount > 0) {
        trailCells.unshift({ ...hoveredSquare });
        if (trailCells.length > hoverTrailAmount) trailCells.length = hoverTrailAmount;
      }
      hoveredSquare = { x: col, y: row };
    }
  }

  function handleMouseLeave() {
    if (hoveredSquare && hoverTrailAmount > 0) {
      trailCells.unshift({ ...hoveredSquare });
      if (trailCells.length > hoverTrailAmount) trailCells.length = hoverTrailAmount;
    }
    hoveredSquare = null;
  }

  // Listen on parent window because iframe pointerEvents = 'none'
  try {
    if (window.parent && window.parent !== window) {
      window.parent.addEventListener('mousemove', e => {
        handleMouseMove(e.clientX, e.clientY);
      }, { passive: true });
    }
  } catch(e) {}

  window.addEventListener('mousemove', e => {
    handleMouseMove(e.clientX, e.clientY);
  }, { passive: true });

  window.addEventListener('mouseleave', handleMouseLeave);

  updateAnimation();
})();
</script>
</body>
</html>"""

# ── CSS TO MAKE STREAMLIT TRANSPARENT ─────────────────────────────────────────
_BG_CSS = """
<style>
/* Make Streamlit background transparent so grid shows through */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background: transparent !important;
}
/* Ensure the UI elements sit above the background */
section.main {
    position: relative !important;
    z-index: 1 !important;
}
</style>
"""

def render_shapegrid_background():
    st.markdown(_BG_CSS, unsafe_allow_html=True)
    components.html(_SHAPEGRID_HTML, height=0, scrolling=False)

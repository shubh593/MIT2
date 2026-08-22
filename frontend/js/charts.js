// HUD Graphics & Telemetry Charts

export class HudCharts {
  static drawRadialMeter(canvas, score, color = "#00f2fe", label = "SAFETY INDEX") {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.38;

    ctx.clearRect(0, 0, w, h);

    // Background track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0.75 * Math.PI, 2.25 * Math.PI);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.stroke();

    // Foreground progress arc
    const progressAngle = 0.75 * Math.PI + (score / 100) * (1.5 * Math.PI);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0.75 * Math.PI, progressAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0; // Reset shadow

    // Center text score
    ctx.fillStyle = "#ffffff";
    ctx.font = `bold ${Math.round(w * 0.18)}px 'Orbitron', monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(score)}%`, cx, cy - 4);

    // Label below
    ctx.fillStyle = "#94a3b8";
    ctx.font = `700 ${Math.round(w * 0.065)}px 'Orbitron', sans-serif`;
    ctx.fillText(label, cx, cy + radius * 0.65);
  }

  static drawHazardBreakdown(canvas, classBreakdown) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const keys = Object.keys(classBreakdown || {});
    if (keys.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("NO HAZARDS DETECTED", w / 2, h / 2);
      return;
    }

    const total = Object.values(classBreakdown).reduce((a, b) => a + b, 0);
    const colors = {
      crater: "#ef4444",
      rock: "#f59e0b",
      boulder: "#f97316",
      hazard: "#ec4899",
      rough_terrain: "#a855f7"
    };

    let startX = 10;
    const barWidth = w - 20;
    const barHeight = 16;
    const barY = 24;

    // Draw segmented bar
    let currentX = startX;
    keys.forEach(k => {
      const count = classBreakdown[k];
      const segmentWidth = (count / total) * barWidth;
      ctx.fillStyle = colors[k.toLowerCase()] || "#00f2fe";
      ctx.fillRect(currentX, barY, segmentWidth, barHeight);
      currentX += segmentWidth;
    });

    // Legend
    let legendY = 64;
    keys.forEach((k, idx) => {
      const count = classBreakdown[k];
      const col = colors[k.toLowerCase()] || "#00f2fe";

      ctx.fillStyle = col;
      ctx.fillRect(10 + (idx % 2) * 140, legendY + Math.floor(idx / 2) * 22, 10, 10);

      ctx.fillStyle = "#cbd5e1";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(`${k.toUpperCase()}: ${count}`, 26 + (idx % 2) * 140, legendY + 9 + Math.floor(idx / 2) * 22);
    });
  }
}

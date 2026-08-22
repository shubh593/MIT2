// Descent Trajectory Flight Simulation Controller

export class DescentSimulator {
  constructor(modalElement, radarCanvasId, telemetryContainerId) {
    this.modal = modalElement;
    this.radarCanvas = document.getElementById(radarCanvasId);
    this.telemetryContainer = document.getElementById(telemetryContainerId);
    this.animationTimer = null;
    this.currentFrameIdx = 0;
    this.frames = [];
  }

  start(simData) {
    if (!simData || !simData.frames || simData.frames.length === 0) return;
    this.frames = simData.frames;
    this.targetZone = simData.target_zone;
    this.targetCoords = simData.target_coordinates;
    this.currentFrameIdx = 0;

    this.modal.classList.add("open");

    if (this.animationTimer) clearInterval(this.animationTimer);
    this.animationTimer = setInterval(() => this.step(), 220);
  }

  stop() {
    if (this.animationTimer) {
      clearInterval(this.animationTimer);
      this.animationTimer = null;
    }
    this.modal.classList.remove("open");
  }

  step() {
    if (this.currentFrameIdx >= this.frames.length) {
      clearInterval(this.animationTimer);
      this.animationTimer = null;
      return;
    }

    const frame = this.frames[this.currentFrameIdx];
    this.renderFrame(frame);
    this.currentFrameIdx++;
  }

  renderFrame(frame) {
    // 1. Render Telemetry numbers
    if (this.telemetryContainer) {
      this.telemetryContainer.innerHTML = `
        <div class="sim-row">
          <span>MISSION TIME:</span>
          <span style="color: var(--color-cyan); font-weight: bold;">T+ ${frame.time_s}s</span>
        </div>
        <div class="sim-row">
          <span>ALTITUDE:</span>
          <span style="color: #fff; font-weight: bold;">${frame.altitude_m} m</span>
        </div>
        <div class="sim-row">
          <span>DESCENT VELOCITY:</span>
          <span style="color: ${frame.velocity_z_mps > 15 ? 'var(--color-amber)' : 'var(--color-emerald)'}; font-weight: bold;">
            -${frame.velocity_z_mps} m/s
          </span>
        </div>
        <div class="sim-row">
          <span>HORIZONTAL DRIFT (X, Y):</span>
          <span>${frame.drift_x_mps} / ${frame.drift_y_mps} m/s</span>
        </div>
        <div class="sim-row">
          <span>PROPELLANT REMAINING:</span>
          <span style="color: var(--color-cyan);">${frame.fuel_pct}%</span>
        </div>
        <div class="sim-row">
          <span>GIMBAL ATTITUDE:</span>
          <span>${frame.pitch_deg}&deg; PITCH</span>
        </div>
        <div class="sim-row" style="border: 1px solid ${frame.alert_level === 'GREEN_OPTIMAL' ? 'var(--color-emerald)' : 'var(--color-cyan)'}">
          <span>GUIDANCE VERDICT:</span>
          <span style="color: ${frame.alert_level === 'GREEN_OPTIMAL' ? 'var(--color-emerald)' : 'var(--color-cyan)'}; font-weight: bold;">
            ${frame.guidance_msg}
          </span>
        </div>
      `;
    }

    // 2. Render Radar Vector HUD on canvas
    if (this.radarCanvas) {
      const ctx = this.radarCanvas.getContext("2d");
      const w = this.radarCanvas.width;
      const h = this.radarCanvas.height;
      const cx = w / 2;
      const cy = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Radar Concentric Circles
      for (let r = 30; r <= 110; r += 25) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 242, 254, 0.15)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Radar Crosshairs
      ctx.beginPath();
      ctx.moveTo(cx - 120, cy);
      ctx.lineTo(cx + 120, cy);
      ctx.moveTo(cx, cy - 120);
      ctx.lineTo(cx, cy + 120);
      ctx.strokeStyle = "rgba(0, 242, 254, 0.25)";
      ctx.stroke();

      // Radar Rotating Sweep
      const sweepAngle = (this.currentFrameIdx * 0.4) % (Math.PI * 2);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, 110, sweepAngle, sweepAngle + 0.35);
      ctx.fillStyle = "rgba(0, 242, 254, 0.12)";
      ctx.fill();

      // Target LZ Bullseye (Center)
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(16, 185, 129, 0.4)";
      ctx.fill();
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Lander Current Position Vector Offset
      const dx = (frame.position.x - this.targetCoords.x) * 0.4;
      const dy = (frame.position.y - this.targetCoords.y) * 0.4;
      const landerX = cx + dx;
      const landerY = cy + dy;

      // Draw trajectory trace
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(landerX, landerY);
      ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Lander Diamond Icon
      ctx.beginPath();
      ctx.moveTo(landerX, landerY - 8);
      ctx.lineTo(landerX + 8, landerY);
      ctx.lineTo(landerX, landerY + 8);
      ctx.lineTo(landerX - 8, landerY);
      ctx.closePath();
      ctx.fillStyle = "#00f2fe";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();

      // Label
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillText(`ALT: ${frame.altitude_m}m`, landerX + 12, landerY - 6);
    }
  }
}

import { ApiService } from './api.js';
import { HudCharts } from './charts.js';
import { DescentSimulator } from './simulation.js';

// Main Application State
const State = {
  activePresetId: null,
  uploadedFile: null,
  analysisResult: null,
  activeLayers: {
    annotated: true,
    heatmap: true,
    safeZones: true,
    grid: true
  },
  images: {
    raw: null,
    annotated: null,
    heatmap: null
  },
  hoverCoord: null,
  selectedLZ: null
};

// DOM References
let canvas, ctx;
let descentSimulator;

document.addEventListener("DOMContentLoaded", async () => {
  initCanvas();
  initClock();
  initSimulator();
  setupEventListeners();
  await loadPresets();
  await checkHealthAndModelInfo();
  
  // Auto-run analysis on the first preset
  if (State.activePresetId) {
    runAnalysis();
  }
});

function initCanvas() {
  canvas = document.getElementById("terrainCanvas");
  ctx = canvas.getContext("2d");

  canvas.addEventListener("mousemove", onCanvasMouseMove);
  canvas.addEventListener("mouseleave", onCanvasMouseLeave);
  canvas.addEventListener("click", onCanvasClick);
}

function initClock() {
  const clockEl = document.getElementById("missionClock");
  setInterval(() => {
    const now = new Date();
    if (clockEl) {
      clockEl.textContent = `UTC ${now.toISOString().substring(11, 19)}`;
    }
  }, 1000);
}

function initSimulator() {
  const modal = document.getElementById("simModal");
  descentSimulator = new DescentSimulator(modal, "simRadarCanvas", "simTelemetryContainer");
  
  document.getElementById("closeSimBtn").addEventListener("click", () => descentSimulator.stop());
}

async function checkHealthAndModelInfo() {
  try {
    const info = await ApiService.getModelInfo();
    const engineBadge = document.getElementById("engineBadge");
    if (engineBadge && info.onnx_model && info.onnx_model.exists) {
      engineBadge.textContent = "AI ENGINE: ONNX & PYTORCH ONLINE";
    }
  } catch (e) {
    console.warn("System check note:", e);
  }
}

async function loadPresets() {
  try {
    const presets = await ApiService.getPresets();
    const grid = document.getElementById("presetGrid");
    grid.innerHTML = "";

    presets.forEach((p, idx) => {
      const card = document.createElement("div");
      card.className = `preset-card ${idx === 0 ? "active" : ""}`;
      card.dataset.id = p.id;
      if (idx === 0) State.activePresetId = p.id;

      card.innerHTML = `
        <img class="preset-thumb" src="${p.thumbnail || ''}" alt="${p.name}"/>
        <div class="preset-name">${p.name}</div>
        <div class="preset-body">${p.body} // ${p.id}</div>
      `;

      card.addEventListener("click", () => {
        document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
        card.classList.add("active");
        State.activePresetId = p.id;
        State.uploadedFile = null;
        document.getElementById("fileInput").value = "";
        runAnalysis();
      });

      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function setupEventListeners() {
  // Analyze Trigger
  document.getElementById("runScanBtn").addEventListener("click", () => runAnalysis());

  // Layer Toggles
  document.querySelectorAll(".layer-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const layer = btn.dataset.layer;
      State.activeLayers[layer] = !State.activeLayers[layer];
      btn.classList.toggle("active", State.activeLayers[layer]);
      renderScene();
    });
  });

  // Sliders
  setupSliderSync("landerRadiusSlider", "landerRadiusVal", " px");
  setupSliderSync("safetyMarginSlider", "safetyMarginVal", " px");
  setupSliderSync("confSlider", "confVal", "");

  // File Upload
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Export Report Button
  document.getElementById("exportReportBtn").addEventListener("click", handleExportReport);
  document.getElementById("closeReportBtn").addEventListener("click", () => {
    document.getElementById("reportModal").classList.remove("open");
  });
}

function setupSliderSync(sliderId, valueId, suffix) {
  const slider = document.getElementById(sliderId);
  const valDisplay = document.getElementById(valueId);
  slider.addEventListener("input", () => {
    valDisplay.textContent = `${slider.value}${suffix}`;
  });
}

function handleFileSelected(file) {
  State.uploadedFile = file;
  State.activePresetId = null;
  document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
  
  const dropText = document.querySelector(".dropzone-text");
  if (dropText) dropText.textContent = `Selected: ${file.name}`;
  
  runAnalysis();
}

async function runAnalysis() {
  const scanBar = document.getElementById("scanningBar");
  scanBar.classList.add("active");
  const runBtn = document.getElementById("runScanBtn");
  runBtn.disabled = true;
  runBtn.textContent = "AI SCANNING IN PROGRESS...";

  try {
    const engine = document.getElementById("engineSelect").value;
    const confThreshold = parseFloat(document.getElementById("confSlider").value);
    const landerRadius = parseInt(document.getElementById("landerRadiusSlider").value);
    const safetyMargin = parseInt(document.getElementById("safetyMarginSlider").value);

    const result = await ApiService.analyzeTerrain({
      file: State.uploadedFile,
      presetId: State.activePresetId,
      engine,
      confThreshold,
      landerRadius,
      safetyMargin,
      topKSites: 5
    });

    State.analysisResult = result;
    await loadResultImages(result.visualizations);
    updateDashboardUI(result);
    renderScene();

  } catch (err) {
    alert(`Analysis error: ${err.message}`);
    console.error(err);
  } finally {
    scanBar.classList.remove("active");
    runBtn.disabled = false;
    runBtn.textContent = "⚡ INITIATE AI TERRAIN SCAN";
  }
}

async function loadResultImages(vis) {
  const loadImage = (src) => new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });

  const [raw, annotated, heatmap] = await Promise.all([
    loadImage(vis.raw_image),
    loadImage(vis.annotated_image),
    loadImage(vis.risk_heatmap_overlay)
  ]);

  State.images.raw = raw;
  State.images.annotated = annotated;
  State.images.heatmap = heatmap;

  canvas.width = raw.width;
  canvas.height = raw.height;
}

function renderScene() {
  if (!State.images.raw) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Base Image (Annotated or Raw)
  if (State.activeLayers.annotated && State.images.annotated) {
    ctx.drawImage(State.images.annotated, 0, 0);
  } else if (State.images.raw) {
    ctx.drawImage(State.images.raw, 0, 0);
  }

  // 2. Draw Risk Heatmap Layer with alpha blending
  if (State.activeLayers.heatmap && State.images.heatmap) {
    ctx.globalAlpha = 0.55;
    ctx.drawImage(State.images.heatmap, 0, 0);
    ctx.globalAlpha = 1.0;
  }

  // 3. Draw Grid Lines
  if (State.activeLayers.grid) {
    ctx.strokeStyle = "rgba(0, 242, 254, 0.12)";
    ctx.lineWidth = 1;
    const step = 64;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
  }

  // 4. Draw Safe Landing Zones Highlights if enabled
  if (State.activeLayers.safeZones && State.analysisResult) {
    const lzs = State.analysisResult.recommended_landing_zones || [];
    lzs.forEach(lz => {
      const [cx, cy] = lz.center;
      const isSelected = State.selectedLZ && State.selectedLZ.id === lz.id;
      
      if (isSelected) {
        // Glowing target ring around selected LZ
        ctx.beginPath();
        ctx.arc(cx, cy, lz.radius + 14, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.shadowColor = "#00f2fe";
        ctx.shadowBlur = 15;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });
  }

  // 5. Draw Interactive Cursor Reticle
  if (State.hoverCoord) {
    const { x, y } = State.hoverCoord;
    ctx.strokeStyle = "rgba(0, 242, 254, 0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Lander probe ring preview
    const r = parseInt(document.getElementById("landerRadiusSlider").value);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0, 242, 254, 0.8)";
    ctx.stroke();
  }
}

function updateDashboardUI(res) {
  const v = res.mission_verdict;
  const t = res.telemetry;

  // Verdict Banner
  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner ${v.verdict_color}`;
  document.getElementById("verdictBadge").innerHTML = `
    <span>${v.overall_verdict === 'GO' ? '🟢' : v.overall_verdict === 'CAUTION' ? '🟡' : '🔴'}</span>
    MISSION CLEARANCE: ${v.overall_verdict}
  `;
  document.getElementById("verdictText").textContent = v.verdict_detail;

  // Telemetry Numbers
  document.getElementById("metricLatency").textContent = `${t.inference_time_ms} ms`;
  document.getElementById("metricEngine").textContent = t.engine_used.split(' ')[0];
  document.getElementById("metricCoverage").textContent = `${v.hazard_coverage_pct}%`;
  document.getElementById("metricHazards").textContent = `${v.total_obstacles}`;

  // Draw Charts
  const radialCanvas = document.getElementById("safetyRadialCanvas");
  const primaryScore = res.recommended_landing_zones[0]?.safety_score || 0;
  const radialColor = v.overall_verdict === 'GO' ? '#10b981' : v.overall_verdict === 'CAUTION' ? '#f59e0b' : '#ef4444';
  HudCharts.drawRadialMeter(radialCanvas, primaryScore, radialColor, "PRIMARY LZ SAFETY");

  const hazardCanvas = document.getElementById("hazardBreakdownCanvas");
  HudCharts.drawHazardBreakdown(hazardCanvas, v.class_breakdown);

  // Render Landing Zone Cards
  renderLandingZoneCards(res.recommended_landing_zones);
}

function renderLandingZoneCards(lzList) {
  const container = document.getElementById("lzList");
  container.innerHTML = "";

  if (!lzList || lzList.length === 0) {
    container.innerHTML = `<div style="color: var(--text-dim); font-size: 0.8rem;">No landing zones meet safety clearance criteria.</div>`;
    return;
  }

  lzList.forEach((lz, idx) => {
    const card = document.createElement("div");
    card.className = `lz-card ${idx === 0 ? "selected" : ""}`;
    if (idx === 0) State.selectedLZ = lz;

    const scoreCol = lz.safety_score >= 80 ? 'var(--color-emerald)' : lz.safety_score >= 60 ? 'var(--color-amber)' : 'var(--color-ruby)';

    card.innerHTML = `
      <div class="lz-header-row">
        <span class="lz-designation" style="color: ${scoreCol}">[${lz.id}] ${lz.designation}</span>
        <span class="lz-score" style="color: ${scoreCol}">${lz.safety_score}% SAFE</span>
      </div>
      <div class="lz-details-row">
        <span>COORDS: (${Math.round(lz.center[0])}, ${Math.round(lz.center[1])})</span>
        <span>CLEARANCE: ${lz.safety_margin_px}px MARGIN</span>
      </div>
      <div class="lz-details-row">
        <span>FLATNESS: ${lz.flatness_index}%</span>
        <button class="btn btn-secondary btn-sim-target" style="padding: 3px 8px; font-size: 0.68rem;">SIMULATE DESCENT &rarr;</button>
      </div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".lz-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      State.selectedLZ = lz;
      renderScene();
    });

    card.querySelector(".btn-sim-target").addEventListener("click", (e) => {
      e.stopPropagation();
      launchDescentSimulation(lz);
    });

    container.appendChild(card);
  });
}

async function launchDescentSimulation(targetLZ) {
  try {
    const simData = await ApiService.simulateDescent(targetLZ);
    descentSimulator.start(simData);
  } catch (e) {
    alert(`Simulation failed: ${e.message}`);
  }
}

// Canvas Mouse Hover & Probe Events
function onCanvasMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  State.hoverCoord = { x, y };
  document.getElementById("probeCoord").textContent = `X: ${Math.round(x)} | Y: ${Math.round(y)}`;
  renderScene();
}

function onCanvasMouseLeave() {
  State.hoverCoord = null;
  renderScene();
}

async function onCanvasClick(e) {
  if (!State.hoverCoord || !State.analysisResult) return;
  const { x, y } = State.hoverCoord;
  const landerRadius = parseInt(document.getElementById("landerRadiusSlider").value);

  const probe = await ApiService.probePoint(x, y, landerRadius);
  if (probe) {
    const statusEl = document.getElementById("probeStatus");
    const scoreEl = document.getElementById("probeSafetyScore");
    
    statusEl.textContent = probe.status;
    statusEl.style.color = probe.status_color === 'emerald' ? 'var(--color-emerald)' : probe.status_color === 'amber' ? 'var(--color-amber)' : 'var(--color-ruby)';
    scoreEl.textContent = `${probe.landing_safety_pct}%`;
  }
}

async function handleExportReport() {
  if (!State.analysisResult) {
    alert("Please run an analysis first.");
    return;
  }

  try {
    const reportData = await ApiService.exportReport(State.analysisResult);
    const content = document.getElementById("reportContent");
    content.textContent = reportData.formatted_markdown;
    document.getElementById("reportModal").classList.add("open");

    // JSON Download button
    document.getElementById("downloadJsonBtn").onclick = () => {
      const blob = new Blob([JSON.stringify(reportData.report_json, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `AEROSAFE_MISSION_REPORT_${Date.now()}.json`;
      a.click();
    };
  } catch (e) {
    alert(`Failed to export report: ${e.message}`);
  }
}

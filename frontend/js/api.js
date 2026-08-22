// API Communication Client for AEROSAFE-LUNAR

const API_BASE = window.location.origin;

export const ApiService = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getModelInfo() {
    const res = await fetch(`${API_BASE}/api/model-info`);
    if (!res.ok) throw new Error('Model info query failed');
    return res.json();
  },

  async getPresets() {
    const res = await fetch(`${API_BASE}/api/presets`);
    if (!res.ok) throw new Error('Failed to load presets');
    return res.json();
  },

  async analyzeTerrain({ file, presetId, engine, confThreshold, iouThreshold, landerRadius, safetyMargin, topKSites }) {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (presetId) {
      formData.append('preset_id', presetId);
    }
    if (engine) {
      formData.append('engine', engine);
    }
    formData.append('conf_threshold', confThreshold ?? 0.25);
    formData.append('iou_threshold', iouThreshold ?? 0.45);
    formData.append('lander_radius_px', landerRadius ?? 45);
    formData.append('safety_margin_px', safetyMargin ?? 25);
    formData.append('top_k_sites', topKSites ?? 5);

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail || 'Analysis failed');
    }
    return res.json();
  },

  async probePoint(x, y, landerRadius) {
    const res = await fetch(`${API_BASE}/api/probe-point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        x: Math.round(x),
        y: Math.round(y),
        lander_radius_px: landerRadius ?? 45
      })
    });
    if (!res.ok) return null;
    return res.json();
  },

  async simulateDescent(landingZone, altitudeStart = 1200) {
    const res = await fetch(`${API_BASE}/api/simulate-descent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        landing_zone: landingZone,
        altitude_start_m: altitudeStart,
        steps: 30
      })
    });
    if (!res.ok) throw new Error('Simulation failed');
    return res.json();
  },

  async exportReport(analysisData, missionName = "LUNAR-ARTEMIS-SITE-VAL") {
    const res = await fetch(`${API_BASE}/api/export-report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis_data: analysisData,
        mission_name: missionName,
        operator_id: "AUTONOMOUS-AI-DIRECTOR"
      })
    });
    if (!res.ok) throw new Error('Report export failed');
    return res.json();
  }
};

# AEROSAFE-LUNAR

**AI-Based Planetary Landing Risk Assessment** — Hackathon Challenge SW08

An autonomous hazard-avoidance and landing-site optimization system. It detects terrain hazards from planetary surface imagery, builds a continuous spatial risk map, identifies and ranks safe touchdown zones, simulates a descent trajectory to the chosen site, and exports a mission-ready clearance report — all through a live Mission Control style dashboard.

---

## Why this matters

Planetary landings fail most often due to undetected terrain hazards discovered too late during descent — this was a key factor behind Chandrayaan-2's setback, and precision hazard avoidance was central to Chandrayaan-3's success. AEROSAFE-LUNAR automates that decision: instead of a human or a simple threshold rule judging terrain safety, an AI pipeline detects hazards, quantifies risk continuously across the whole surface, and recommends ranked, physically-validated landing zones in real time.

## System overview

```
Image input (upload or preset)
        │
        ▼
Dual-engine hazard detection (ONNX → PyTorch → classical CV fallback)
        │
        ▼
Spatial risk surface (hazard heatmap + terrain roughness)
        │
        ▼
Landing zone optimization (distance transform + multi-factor scoring)
        │
        ▼
Descent trajectory simulation ──► Mission report export
        │
        ▼
Mission Control dashboard (live HUD, heatmap overlay, telemetry)
```

## Core engineering highlights

- **Three-tier detection resilience**: the system tries ONNX Runtime first (fast, optimized inference), falls back to native PyTorch YOLO, and — if both return zero detections — falls back again to a classical computer-vision detector (Hough Circle Transform for craters, adaptive-threshold contour analysis for rocks/boulders). The system never fails silently; it degrades gracefully.
- **Continuous risk surface, not just bounding boxes**: hazards produce a Gaussian-falloff heatmap (risk decreases smoothly with distance from each hazard), combined with an independent surface-roughness signal derived from Sobel and Laplacian image gradients — so risk assessment isn't solely dependent on what the AI model detects.
- **Physically-grounded landing zone search**: candidate sites are found using a Euclidean distance transform (maximizing clearance from every hazard), scored on clearance distance (50%), local flatness (35%), and approach trajectory favorability (15%), then filtered with non-maximum suppression so ranked zones (Primary / Secondary / Contingency) don't overlap.
- **Interactive point probing**: any pixel on the analyzed terrain can be queried for its individual landing safety score and nearest-hazard distance — supports "what if we land here instead" exploration.
- **Synthetic descent telemetry**: a full time-series simulation (altitude, descent velocity, horizontal drift, fuel remaining, thruster pitch, radar lock, alert level) is generated for the chosen landing zone, enabling a live "watch the descent" demo.
- **Mission-grade reporting**: a structured GO / CAUTION / NO-GO verdict with a formatted markdown mission briefing, exportable on demand.

## Tech stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI (served via Uvicorn), CORS-enabled |
| AI / detection engine | Ultralytics YOLO — dual inference via PyTorch (`best.pt`) and ONNX Runtime (`best.onnx`), with classical OpenCV fallback |
| Risk analysis | NumPy, OpenCV (Sobel/Laplacian gradients, Gaussian heatmaps) |
| Frontend | Vanilla JS (modular: `api.js`, `hud.js`, `charts.js`, `simulation.js`), custom CSS "Mission Control HUD" styling |
| Package management | [uv](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`) |
| Language / runtime | Python 3.12 |

## Model details

Extracted directly from the shipped `best.onnx` metadata:

| Property | Value |
|---|---|
| Framework | Ultralytics YOLO v8.4.126 |
| Input size | 1024 × 1024, 3-channel |
| Trained on | Custom dataset: *Moon Crater & Boulder Detection* |
| Task | Object detection (hazard localization) |
| NMS | Applied in backend post-processing (not baked into export) |
| License | AGPL-3.0 (Ultralytics) |

Detected hazard classes are risk-weighted individually (see `backend/config.py`):

| Class | Risk weight |
|---|---|
| Crater | 0.95 |
| Hazard (generic) | 0.90 |
| Boulder | 0.85 |
| Rock | 0.80 |
| Slope | 0.75 |
| Rough terrain | 0.70 |
| Shadow | 0.50 |

## Project structure

```
.
├── main.py                          # Entry point — launches FastAPI app via Uvicorn
├── pyproject.toml / uv.lock          # Dependencies, managed via uv
├── best.pt / best.onnx               # Trained YOLO model — PyTorch & ONNX exports
├── backend/
│   ├── app.py                        # FastAPI app setup, CORS, static mount, startup hooks
│   ├── config.py                     # Model paths, risk weights, default detection parameters
│   ├── engine/
│   │   ├── detector.py               # PlanetaryHazardDetector — dual-engine + CV fallback detection
│   │   └── model_inspector.py        # Inspects ONNX/PyTorch model metadata for the API
│   ├── analytics/
│   │   ├── risk_analyzer.py          # Hazard heatmap, terrain roughness, combined risk surface, verdicts
│   │   ├── landing_zone_finder.py    # Distance-transform based safe zone search & ranking
│   │   └── trajectory_sim.py         # Synthetic descent telemetry generator
│   ├── routes/
│   │   ├── analysis_routes.py        # /api/analyze, /api/probe-point
│   │   ├── simulation_routes.py      # /api/simulate-descent
│   │   └── system_routes.py          # /api/health, /api/model-info, /api/presets, /api/export-report
│   └── utils/
│       ├── image_ops.py              # Base64 encoding, heatmap colorization, HUD overlay rendering
│       └── preset_generator.py       # Generates 4 synthetic planetary terrain presets on startup
├── frontend/
│   ├── index.html                    # Mission Control HUD dashboard
│   ├── css/hud.css                   # Sci-fi telemetry styling
│   └── js/                           # api.js, hud.js, charts.js, simulation.js
├── data/
│   ├── presets/                      # Synthetic terrain images (Lunar south pole, Mare Tranquillitatis, Mars Jezero, Europa/Enceladus)
│   └── exports/                      # Generated mission reports
└── tests/
    └── test_clearance.py             # Landing clearance logic tests
```

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | System status, model availability |
| GET | `/api/model-info` | Inspects loaded ONNX/PyTorch model metadata |
| GET | `/api/presets` | Lists the 4 built-in planetary terrain scenarios with thumbnails |
| POST | `/api/analyze` | Main pipeline — image in, full risk analysis + ranked landing zones out |
| POST | `/api/probe-point` | Query landing safety at an arbitrary clicked coordinate |
| POST | `/api/simulate-descent` | Generates descent trajectory telemetry to a chosen landing zone |
| POST | `/api/export-report` | Produces a formatted mission clearance briefing (JSON + markdown) |

Full interactive documentation is auto-generated at `/docs`.

## Getting started

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py
```

- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

Four built-in planetary terrain presets are generated automatically on first startup, so the system is demo-ready with no external dataset required.

## Roadmap / future work

- Persist analysis history to a database for auditability across missions
- Live/continuous camera-stream ingestion (currently image-in, image-out per request)
- Slope estimation from real digital elevation model (DEM) data, layered onto the existing roughness signal
- Multi-class hazard taxonomy refinement using real annotated planetary imagery at scale
- Production hardening: authentication, TLS, rate limiting on public endpoints

## Team

| Role | Name | Responsibility |
|---|---|---|
| ML Lead | | Model training & dual-engine inference pipeline |
| Data Engineer | | Dataset preparation & terrain preset curation |
| Backend/Logic Engineer | | Risk scoring, landing zone optimization, API |
| Frontend/Visualization Engineer | | Mission Control HUD dashboard |
| Integration & Pitch Lead | | System integration, demo & presentation |

## License

Built on Ultralytics YOLO, licensed under **AGPL-3.0**. See [Ultralytics License](https://ultralytics.com/license).
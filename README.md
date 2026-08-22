# AI_Based Landing risk assessment system 

**AI-Based Planetary Landing Risk Assessment** — Hackathon 

An AI-powered landing assistance system that detects terrain hazards (craters, boulders) from descent imagery, calculates a landing-risk score per zone, and recommends the safest landing location — built to support the kind of hazard-avoidance decisions that were critical to missions like Chandrayaan-3.

---

## Problem statement

Planetary landings fail most often due to undetected terrain hazards — craters, boulders, and uneven slopes that aren't visible until a lander is already descending. AeroSafe-Lunar automates hazard detection and risk scoring so that mission teams (or an autonomous lander) get a fast, explainable recommendation on where to land safely.

## How it works

1. **Image input** — a descent image (from camera feed or file) is passed to the system.
2. **Hazard detection** — a YOLO-based computer vision model scans the image and detects hazards (craters/boulders).
3. **Risk scoring** — detected hazards are mapped onto a grid; each zone gets a computed risk score based on hazard density and severity.
4. **Recommendation** — the safest zones are ranked and surfaced on a live dashboard, with the underlying data available via a documented API.

## Tech stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI (served via Uvicorn) |
| AI / detection model | YOLO (Ultralytics), dual inference: PyTorch (`best.pt`) + ONNX Runtime (`best.onnx`) |
| Image processing | OpenCV (headless), Pillow |
| Numerical / scoring | NumPy, SciPy |
| Package management | [uv](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`) |
| Language / runtime | Python 3.12 |

## Why a dual inference engine (PyTorch + ONNX)

- **`best.pt`** — the native PyTorch weights, used during development, validation, and further fine-tuning.
- **`best.onnx`** — an ONNX Runtime–optimized export of the same model, used for fast, lightweight, dependency-light inference — the same pattern used in real-world edge and robotics deployments where inference speed matters more than training flexibility.

## Model details

Extracted directly from the shipped `best.onnx` model metadata:

| Property | Value |
|---|---|
| Task | Object detection |
| Framework | Ultralytics YOLO v8.4.126 |
| Input size | 1024 × 1024, 3-channel |
| Trained on | Custom dataset: *Moon Crater & Boulder Detection* |
| Classes | 1 (unified hazard class — crater/boulder detection combined) |
| NMS | Applied post-inference in backend (not baked into the exported model) |
| License | AGPL-3.0 (Ultralytics) |

> **Note:** the current model detects a single unified "hazard" class rather than separately labeled craters vs. boulders. This is documented here as a known limitation / roadmap item — see **Future Work**.

## Project structure

```
.
├── main.py              # Entry point — launches FastAPI app via Uvicorn
├── pyproject.toml        # Project dependencies (managed via uv)
├── uv.lock                # Locked dependency versions for reproducible installs
├── best.pt                # Trained YOLO model weights (PyTorch)
├── best.onnx              # Exported YOLO model (ONNX Runtime, optimized inference)
├── backend/                # FastAPI application, API routes, inference & scoring logic
└── .python-version         # Pinned Python version (3.12)
```

> `backend/` contains the FastAPI app (`app.py`) referenced by `main.py` — routes, risk-scoring logic, and dashboard templates live there. Keep this folder documented as it grows; it's the core of the system.

## Getting started

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv) installed

```bash
# Install dependencies (reads pyproject.toml + uv.lock)
uv sync

# Run the application
uv run python main.py
```

Once running:
- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Roadmap / future work

- Split single hazard class into separate **crater** and **boulder** classes for more granular risk scoring
- Add persistent storage (database) to log analyzed images, detections, and chosen landing zones for auditability
- Real-time video/camera-stream ingestion (currently designed around static/sequential image input)
- Slope and terrain-roughness data fusion (DEM-based) alongside visual detection for more robust scoring
- Security hardening for production deployment (auth, TLS, rate limiting) — out of scope for the hackathon prototype

## Team

*(Fill in team member names and roles here)*

| Role | Name | Responsibility |
|---|---|---|
| ML Lead | | Model training & inference pipeline |
| Data Engineer | | Dataset preparation & curation |
| Backend/Logic Engineer | | Risk scoring & API |
| Frontend/Visualization Engineer | | Dashboard & UI |
| Integration & Pitch Lead | | System integration & presentation |

## License

Model built on Ultralytics YOLO, licensed under **AGPL-3.0**. See [Ultralytics License](https://ultralytics.com/license) for details.
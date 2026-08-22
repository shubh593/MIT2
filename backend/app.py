import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes.analysis_routes import router as analysis_router
from backend.routes.simulation_routes import router as sim_router
from backend.routes.system_routes import router as system_router
from backend.utils.preset_generator import generate_sample_planetary_terrains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AeroSafeApp")

app = FastAPI(
    title="AEROSAFE-LUNAR: AI Planetary Landing Risk Assessment System",
    description="Autonomous hazard avoidance, spatial terrain risk scoring, and landing zone optimization (SW08).",
    version="1.0.0"
)

# Enable CORS for local development and embedded browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(system_router)
app.include_router(analysis_router)
app.include_router(sim_router)

# Mount static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    """Ensure presets and models are preloaded on startup."""
    logger.info("Initializing AEROSAFE-LUNAR AI Mission Control System...")
    try:
        generate_sample_planetary_terrains()
        logger.info("Planetary presets initialized successfully.")
    except Exception as e:
        logger.warning(f"Preset initialization warning: {e}")

@app.get("/")
async def serve_index():
    """Serves the main Space HUD Mission Control dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "AEROSAFE-LUNAR API online. Frontend index.html not yet created."}

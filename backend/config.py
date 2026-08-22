import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PT_PATH = BASE_DIR / "best.pt"
MODEL_ONNX_PATH = BASE_DIR / "best.onnx"
PRESETS_DIR = BASE_DIR / "data" / "presets"
EXPORTS_DIR = BASE_DIR / "data" / "exports"

# Ensure runtime directories exist
PRESETS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Hazard Risk Weights
CLASS_RISK_WEIGHTS = {
    "crater": 0.95,
    "boulder": 0.85,
    "rock": 0.80,
    "hazard": 0.90,
    "rough_terrain": 0.70,
    "slope": 0.75,
    "shadow": 0.50,
}

# Default Analysis Parameters
DEFAULT_PARAMS = {
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "lander_radius_px": 45,  # Radius of lander footprint in pixels
    "safety_margin_px": 25,  # Buffer margin required around lander
    "engine": "auto",        # 'onnx', 'pt', or 'auto'
    "top_k_sites": 5,        # Top recommended landing zones
}

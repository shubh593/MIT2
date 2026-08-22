import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel
import cv2

from backend.config import (
    MODEL_PT_PATH,
    MODEL_ONNX_PATH,
    PRESETS_DIR,
    DEFAULT_PARAMS
)
from backend.engine.model_inspector import inspect_onnx_model, inspect_pt_model
from backend.utils.preset_generator import generate_sample_planetary_terrains
from backend.utils.image_ops import image_to_base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["System & Presets"])

class ExportReportRequest(BaseModel):
    analysis_data: Dict[str, Any]
    mission_name: Optional[str] = "LUNAR-ARTEMIS-SITE-VAL"
    operator_id: Optional[str] = "MISSION-DIRECTOR-AI"

@router.get("/health")
def health_check():
    """Returns system status and model readiness."""
    return {
        "status": "online",
        "service": "Planetary Landing Risk Assessment (SW08)",
        "models": {
            "onnx_available": MODEL_ONNX_PATH.exists(),
            "pt_available": MODEL_PT_PATH.exists(),
        }
    }

@router.get("/model-info")
def get_model_info():
    """Inspects ONNX and PyTorch weights, classes, and tensor attributes."""
    onnx_info = inspect_onnx_model(MODEL_ONNX_PATH)
    pt_info = inspect_pt_model(MODEL_PT_PATH)
    
    return {
        "onnx_model": onnx_info,
        "pt_model": pt_info,
        "default_parameters": DEFAULT_PARAMS,
    }

@router.get("/presets")
def get_presets():
    """Lists pre-configured planetary surface test scenarios with preview thumbnails."""
    presets = generate_sample_planetary_terrains()
    
    # Attach thumbnail base64
    for p in presets:
        img_path = Path(p["file_path"])
        if img_path.exists():
            img = cv2.imread(str(img_path))
            thumb = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            p["thumbnail"] = image_to_base64(thumb, quality=75)
            
    return presets

@router.post("/export-report")
def export_mission_report(req: ExportReportRequest):
    """
    Generates structured mission clearance report formatted for aerospace telemetry briefings.
    """
    data = req.analysis_data
    telemetry = data.get("telemetry", {})
    verdict = data.get("mission_verdict", {})
    landing_zones = data.get("recommended_landing_zones", [])
    hazards = data.get("hazards", {})
    
    report = {
        "report_header": {
            "mission_id": req.mission_name,
            "security_clearance": "AEROSPACE AUTONOMOUS TOUCHDOWN ASSURANCE",
            "problem_code": "SW08 - AI-Based Landing Risk Assessment",
            "evaluator": req.operator_id,
            "engine": telemetry.get("engine_used", "AI Dual Engine"),
            "resolution": telemetry.get("image_resolution", "N/A"),
        },
        "executive_summary": {
            "overall_status": verdict.get("overall_verdict", "UNKNOWN"),
            "operational_recommendation": verdict.get("verdict_detail", "N/A"),
            "mean_risk_index_pct": verdict.get("mean_risk_index", 0),
            "hazard_coverage_pct": verdict.get("hazard_coverage_pct", 0),
            "total_obstacles_identified": hazards.get("total_count", 0),
        },
        "recommended_landing_corridors": [
            {
                "site_id": lz.get("id"),
                "rank": lz.get("rank"),
                "designation": lz.get("designation"),
                "target_pixel_coords": lz.get("center"),
                "safety_score_pct": lz.get("safety_score"),
                "clearance_distance_px": lz.get("clearance_distance_px"),
                "safety_margin_px": lz.get("safety_margin_px"),
                "flatness_index_pct": lz.get("flatness_index"),
            }
            for lz in landing_zones
        ],
        "hazard_breakdown_by_category": verdict.get("class_breakdown", {}),
    }
    
    return {
        "status": "success",
        "report_json": report,
        "formatted_markdown": _format_markdown_report(report)
    }

def _format_markdown_report(rep: Dict[str, Any]) -> str:
    header = rep["report_header"]
    summary = rep["executive_summary"]
    lzs = rep["recommended_landing_corridors"]
    
    md = f"""# 🚀 Planetary Landing Risk & Hazard Assessment Briefing
**Mission Code:** {header['mission_id']} | **Problem ID:** {header['problem_code']}
**AI Engine:** {header['engine']} | **Sensor Resolution:** {header['resolution']}

---

## 🎯 Executive Verdict: **{summary['overall_status']}**
> {summary['operational_recommendation']}

- **Mean Terrain Risk Index:** {summary['mean_risk_index_pct']}%
- **Hazard Surface Density:** {summary['hazard_coverage_pct']}%
- **Total Detected Obstacles (Craters / Rocks):** {summary['total_obstacles_identified']}

---

## 📍 Ranked Safe Touchdown Zones
| Designation | Coordinates (X, Y) | Safety Score | Clearance Margin | Flatness Index |
| :--- | :--- | :--- | :--- | :--- |
"""
    for lz in lzs:
        coords = f"({lz['target_pixel_coords'][0]}, {lz['target_pixel_coords'][1]})"
        md += f"| **{lz['site_id']} - {lz['designation']}** | `{coords}` | **{lz['safety_score_pct']}%** | {lz['safety_margin_px']} px | {lz['flatness_index_pct']}% |\n"

    md += "\n---\n*Generated by AEROSAFE-LUNAR Autonomous Decision Support Engine (SW08)*"
    return md

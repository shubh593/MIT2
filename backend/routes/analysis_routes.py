import io
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

from backend.config import DEFAULT_PARAMS, PRESETS_DIR
from backend.engine.detector import PlanetaryHazardDetector
from backend.analytics.risk_analyzer import TerrainRiskAnalyzer
from backend.analytics.landing_zone_finder import LandingZoneOptimizer
from backend.utils.image_ops import (
    image_to_base64,
    base64_to_image,
    colorize_risk_heatmap,
    render_hud_tactical_overlay
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Terrain Risk Analysis"])

# Global singleton detector
detector_instance = PlanetaryHazardDetector()

# Cached last analysis for interactive coordinate probing
last_analysis_cache: Dict[str, Any] = {}

class ProbeRequest(BaseModel):
    x: int
    y: int
    lander_radius_px: Optional[int] = DEFAULT_PARAMS["lander_radius_px"]

@router.post("/analyze")
async def analyze_terrain(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
    engine: Optional[str] = Form("auto"),
    conf_threshold: Optional[float] = Form(DEFAULT_PARAMS["conf_threshold"]),
    iou_threshold: Optional[float] = Form(DEFAULT_PARAMS["iou_threshold"]),
    lander_radius_px: Optional[int] = Form(DEFAULT_PARAMS["lander_radius_px"]),
    safety_margin_px: Optional[int] = Form(DEFAULT_PARAMS["safety_margin_px"]),
    top_k_sites: Optional[int] = Form(DEFAULT_PARAMS["top_k_sites"]),
):
    """
    Main Analysis Endpoint:
    Processes terrain image, detects hazards with YOLO (.pt / .onnx),
    generates spatial risk heatmap, scores and identifies optimal landing zones.
    """
    start_total = time.perf_counter()
    image_bgr = None
    
    # 1. Ingest image from file or preset
    if file and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        source_name = file.filename
    elif preset_id:
        preset_file = PRESETS_DIR / f"{preset_id}.jpg"
        if preset_file.exists():
            image_bgr = cv2.imread(str(preset_file))
            source_name = f"Preset: {preset_id}"
        else:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")
    else:
        # Default fallback to first preset
        presets = list(PRESETS_DIR.glob("*.jpg"))
        if presets:
            image_bgr = cv2.imread(str(presets[0]))
            source_name = presets[0].stem
        else:
            raise HTTPException(status_code=400, detail="No image provided and no presets available.")
            
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode input image.")

    h, w = image_bgr.shape[:2]

    # 2. AI Hazard Detection
    det_res = detector_instance.detect(
        image_bgr=image_bgr,
        engine=engine or "auto",
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold)
    )
    detections = det_res["detections"]

    # 3. Spatial Risk Analysis & Surface Mapping
    hazard_heat = TerrainRiskAnalyzer.compute_hazard_heatmap((h, w), detections)
    roughness_map = TerrainRiskAnalyzer.estimate_terrain_roughness(image_bgr)
    combined_risk_surface = TerrainRiskAnalyzer.generate_combined_risk_surface(hazard_heat, roughness_map)
    risk_summary = TerrainRiskAnalyzer.summarize_risk(combined_risk_surface, detections)

    # 4. Landing Zone Optimization (Safe Touchdown Areas)
    optimizer = LandingZoneOptimizer(
        lander_radius_px=int(lander_radius_px),
        safety_margin_px=int(safety_margin_px)
    )
    safe_zones = optimizer.find_safe_zones(
        image_shape=(h, w),
        risk_surface=combined_risk_surface,
        detections=detections,
        top_k=int(top_k_sites)
    )

    # 5. Render HUD Tactical Visuals
    annotated_bgr = render_hud_tactical_overlay(
        image_bgr=image_bgr,
        detections=detections,
        landing_zones=safe_zones,
        lander_radius_px=int(lander_radius_px),
        safety_margin_px=int(safety_margin_px)
    )

    # Encode images
    raw_b64 = image_to_base64(image_bgr)
    annotated_b64 = image_to_base64(annotated_bgr)
    heatmap_b64 = colorize_risk_heatmap(combined_risk_surface)

    total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

    # Cache for interactive point prober
    last_analysis_cache["risk_surface"] = combined_risk_surface
    last_analysis_cache["detections"] = detections
    last_analysis_cache["shape"] = (h, w)

    return {
        "status": "success",
        "source": source_name,
        "telemetry": {
            "total_latency_ms": total_time_ms,
            "inference_time_ms": det_res["inference_time_ms"],
            "engine_used": det_res["engine"],
            "image_resolution": f"{w}x{h}",
        },
        "mission_verdict": risk_summary,
        "landing_parameters": {
            "lander_radius_px": lander_radius_px,
            "safety_margin_px": safety_margin_px,
            "total_clearance_required_px": lander_radius_px + safety_margin_px,
        },
        "hazards": {
            "total_count": len(detections),
            "items": detections,
        },
        "recommended_landing_zones": safe_zones,
        "visualizations": {
            "raw_image": raw_b64,
            "annotated_image": annotated_b64,
            "risk_heatmap_overlay": heatmap_b64,
        }
    }

@router.post("/probe-point")
async def probe_point_risk(req: ProbeRequest):
    """
    Interactive Coordinate Inspector:
    Evaluates real-time risk, nearest hazard proximity, and landing viability for clicked (x, y).
    """
    if "risk_surface" not in last_analysis_cache:
        raise HTTPException(status_code=400, detail="No active terrain analysis in cache. Please run analysis first.")
        
    res = LandingZoneOptimizer.evaluate_point_risk(
        x=req.x,
        y=req.y,
        lander_radius=req.lander_radius_px or DEFAULT_PARAMS["lander_radius_px"],
        risk_surface=last_analysis_cache["risk_surface"],
        detections=last_analysis_cache["detections"]
    )
    return res

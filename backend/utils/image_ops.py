import base64
import io
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple

def image_to_base64(image_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode BGR numpy image to base64 JPEG/PNG string."""
    is_success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not is_success:
        raise ValueError("Failed to encode image to base64")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def base64_to_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to BGR numpy image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

def colorize_risk_heatmap(heatmap_0_to_1: np.ndarray) -> str:
    """
    Colorize continuous [0, 1] risk surface using a high-visibility Sci-Fi palette:
    Low Risk (Safe) = Deep Transparent Blue / Emerald
    Medium Risk = Amber / Orange
    High Risk (Hazard) = Vibrant Laser Red / Crimson
    Returns base64 PNG with alpha transparency for direct web canvas overlay.
    """
    h, w = heatmap_0_to_1.shape
    u8_heatmap = np.clip(heatmap_0_to_1 * 255.0, 0, 255).astype(np.uint8)
    
    # Custom 4-channel BGRA colormap
    color_bgr = cv2.applyColorMap(u8_heatmap, cv2.COLORMAP_JET)
    
    # Calculate alpha channel: transparent where risk is low, opaque where risk is high
    alpha = np.clip((heatmap_0_to_1 - 0.08) / 0.85 * 210, 0, 210).astype(np.uint8)
    
    bgra = cv2.merge([color_bgr[:, :, 0], color_bgr[:, :, 1], color_bgr[:, :, 2], alpha])
    
    is_success, buffer = cv2.imencode(".png", bgra)
    if not is_success:
        raise ValueError("Failed to encode heatmap overlay")
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def render_hud_tactical_overlay(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    landing_zones: List[Dict[str, Any]],
    lander_radius_px: int,
    safety_margin_px: int
) -> np.ndarray:
    """
    Renders military/aerospace tactical HUD annotations directly on the terrain image.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    
    # 1. Draw detected hazard bounding boxes with glow & corner brackets
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cname = d["class_name"].upper()
        conf = int(d["confidence"] * 100)
        
        # Hazard color: Crimson / Red for craters & boulders
        color = (0, 0, 255) if "crater" in cname.lower() else (0, 140, 255)
        
        # Draw bounding rect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw corner accents
        bracket_len = min(12, max(4, (x2 - x1) // 4))
        cv2.line(annotated, (x1, y1), (x1 + bracket_len, y1), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y1), (x1, y1 + bracket_len), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y2), (x2 - bracket_len, y2), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y2), (x2, y2 - bracket_len), (255, 255, 255), 2)
        
        # Tag label
        label = f"{cname} {conf}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(0, y1)), (15, 15, 25), -1)
        cv2.putText(annotated, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Draw Recommended Landing Zones (Bullseye Reticles)
    for lz in landing_zones:
        cx, cy = int(lz["center"][0]), int(lz["center"][1])
        rad = lander_radius_px
        total_rad = rad + safety_margin_px
        rank = lz["rank"]
        score = lz["safety_score"]
        
        if rank == 1:
            color = (0, 255, 100) # Bright Emerald
            tag = f"[LZ-01 PRIMARY] {score}%"
        elif rank == 2:
            color = (255, 220, 0) # Cyan
            tag = f"[LZ-02 SECONDARY] {score}%"
        else:
            color = (0, 180, 255) # Amber
            tag = f"[LZ-{rank:02d}] {score}%"
            
        # Draw outer safety buffer circle (dashed-like)
        cv2.circle(annotated, (cx, cy), total_rad, (120, 120, 120), 1, cv2.LINE_AA)
        
        # Draw lander touchdown footprint
        cv2.circle(annotated, (cx, cy), rad, color, 2, cv2.LINE_AA)
        
        # Crosshair reticle
        cv2.line(annotated, (cx - rad - 8, cy), (cx - rad + 8, cy), color, 2)
        cv2.line(annotated, (cx + rad - 8, cy), (cx + rad + 8, cy), color, 2)
        cv2.line(annotated, (cx, cy - rad - 8), (cx, cy - rad + 8), color, 2)
        cv2.line(annotated, (cx, cy + rad - 8), (cx, cy + rad + 8), color, 2)
        cv2.circle(annotated, (cx, cy), 3, color, -1)
        
        # Label badge
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (cx - tw // 2 - 4, cy + rad + 4), (cx + tw // 2 + 4, cy + rad + th + 10), (10, 20, 30), -1)
        cv2.rectangle(annotated, (cx - tw // 2 - 4, cy + rad + 4), (cx + tw // 2 + 4, cy + rad + th + 10), color, 1)
        cv2.putText(annotated, tag, (cx - tw // 2, cy + rad + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated

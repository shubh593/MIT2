import os
import cv2
import numpy as np
from pathlib import Path
from backend.config import PRESETS_DIR

def generate_sample_planetary_terrains():
    """
    Generates realistic synthetic planetary surface terrain maps
    (Lunar Highlands, South Pole Craters, Martian Regolith, Mare Plains)
    if not already present in PRESETS_DIR.
    """
    scenarios = [
        {
            "id": "lunar_south_pole",
            "name": "Lunar South Pole (Shackleton Rim)",
            "body": "Moon",
            "description": "High-contrast polar crater field with deep shadows and ejecta boulders.",
            "craters": [(180, 180, 75), (420, 360, 95), (510, 140, 45), (120, 450, 60), (320, 520, 40)],
            "boulder_count": 35,
            "base_tone": 110,
            "contrast": 1.4,
            "tint": [1.0, 1.0, 1.0] # Lunar gray
        },
        {
            "id": "mare_tranquillitatis",
            "name": "Mare Tranquillitatis (Smooth Plains)",
            "body": "Moon",
            "description": "Relatively flat lunar basalt plain with sparse micro-craters and isolated regolith rocks.",
            "craters": [(120, 140, 40), (520, 480, 35)],
            "boulder_count": 12,
            "base_tone": 135,
            "contrast": 0.9,
            "tint": [0.95, 0.98, 1.0]
        },
        {
            "id": "mars_jezero_delta",
            "name": "Mars Jezero Crater Delta",
            "body": "Mars",
            "description": "Martian ancient delta basin with scattered rocky ridges and boulder clusters.",
            "craters": [(220, 310, 80), (450, 190, 65), (140, 480, 50)],
            "boulder_count": 48,
            "base_tone": 125,
            "contrast": 1.2,
            "tint": [0.75, 0.88, 1.25] # Rusty Martian orange/red tint
        },
        {
            "id": "europa_ridge_plains",
            "name": "Icy Satellite Complex (High Hazard)",
            "body": "Europa / Enceladus",
            "description": "Chaotic fractured ice surface with numerous impact sites and steep elevation gradients.",
            "craters": [(160, 200, 85), (320, 220, 70), (480, 380, 90), (220, 460, 75), (390, 120, 55)],
            "boulder_count": 60,
            "base_tone": 150,
            "contrast": 1.6,
            "tint": [1.2, 1.15, 1.0] # Cyan-ice tint
        }
    ]
    
    presets_meta = []
    
    for sc in scenarios:
        file_path = PRESETS_DIR / f"{sc['id']}.jpg"
        if not file_path.exists():
            img = _synthesize_planetary_surface(
                width=640,
                height=640,
                craters=sc["craters"],
                boulder_count=sc["boulder_count"],
                base_tone=sc["base_tone"],
                contrast=sc["contrast"],
                tint=sc["tint"]
            )
            cv2.imwrite(str(file_path), img)
            
        presets_meta.append({
            "id": sc["id"],
            "name": sc["name"],
            "body": sc["body"],
            "description": sc["description"],
            "file_name": f"{sc['id']}.jpg",
            "file_path": str(file_path)
        })
        
    return presets_meta

def _synthesize_planetary_surface(
    width: int,
    height: int,
    craters: list,
    boulder_count: int,
    base_tone: int,
    contrast: float,
    tint: list
) -> np.ndarray:
    """Generates procedural planetary surface texture using Perlin-like noise, craters, and boulders."""
    # 1. Base fractal noise
    np.random.seed(42 + len(craters))
    noise1 = cv2.resize(np.random.randn(40, 40), (width, height), interpolation=cv2.INTER_CUBIC)
    noise2 = cv2.resize(np.random.randn(120, 120), (width, height), interpolation=cv2.INTER_CUBIC)
    noise3 = cv2.resize(np.random.randn(320, 320), (width, height), interpolation=cv2.INTER_CUBIC)
    
    surface = noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2
    surface = (surface - surface.min()) / (surface.max() - surface.min() + 1e-5)
    img_gray = (surface * 70 + base_tone).astype(np.float32)
    
    # 2. Carve craters with rims and shadowed interiors
    y_coords, x_coords = np.ogrid[:height, :width]
    
    for cx, cy, radius in craters:
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        norm_dist = dist / (radius + 1e-5)
        
        # Depression inside crater
        inside_mask = norm_dist < 1.0
        depth_profile = (1.0 - norm_dist ** 2) * 55.0
        
        # Raised rim
        rim_mask = (norm_dist >= 0.85) & (norm_dist <= 1.25)
        rim_height = np.sin((norm_dist - 0.85) / 0.4 * np.pi) * 35.0
        
        # Directional sunlight shadow (illuminated top-left, shadowed bottom-right)
        angle = np.arctan2(y_coords - cy, x_coords - cx)
        sun_shading = np.cos(angle - 0.75 * np.pi) * 28.0
        
        img_gray[inside_mask] -= depth_profile[inside_mask]
        img_gray[inside_mask] += sun_shading[inside_mask]
        img_gray[rim_mask] += rim_height[rim_mask]
        img_gray[rim_mask] += (sun_shading[rim_mask] * 0.7)

    # 3. Add scattered rocks/boulders with high specular highlights & cast shadows
    for _ in range(boulder_count):
        bx = np.random.randint(20, width - 20)
        by = np.random.randint(20, height - 20)
        brad = np.random.randint(3, 14)
        
        # Shadow
        cv2.circle(img_gray, (bx + brad // 2, by + brad // 2), brad, (20.0,), -1)
        # Rock body
        cv2.circle(img_gray, (bx, by), brad, (210.0,), -1)
        # Highlight on top-left
        cv2.circle(img_gray, (bx - brad // 3, by - brad // 3), max(1, brad // 3), (250.0,), -1)

    # Contrast & Tint
    img_gray = np.clip((img_gray - 128) * contrast + 128, 0, 255).astype(np.uint8)
    
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0] * tint[0], 0, 255)
    bgr[:, :, 1] = np.clip(bgr[:, :, 1] * tint[1], 0, 255)
    bgr[:, :, 2] = np.clip(bgr[:, :, 2] * tint[2], 0, 255)
    
    return bgr.astype(np.uint8)

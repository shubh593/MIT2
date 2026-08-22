import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.engine.detector import PlanetaryHazardDetector
from backend.analytics.risk_analyzer import TerrainRiskAnalyzer
from backend.analytics.landing_zone_finder import LandingZoneOptimizer
from backend.config import PRESETS_DIR

def run_tests():
    detector = PlanetaryHazardDetector()
    preset_files = list(PRESETS_DIR.glob("*.jpg"))
    print(f"Testing {len(preset_files)} terrain maps...")

    for p in preset_files:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        det_res = detector.detect(img)
        detections = det_res["detections"]
        
        heat = TerrainRiskAnalyzer.compute_hazard_heatmap((h, w), detections)
        rough = TerrainRiskAnalyzer.estimate_terrain_roughness(img)
        comb = TerrainRiskAnalyzer.generate_combined_risk_surface(heat, rough)
        
        lander_rad = 45
        safety_marg = 25
        opt = LandingZoneOptimizer(lander_radius_px=lander_rad, safety_margin_px=safety_marg)
        lzs = opt.find_safe_zones((h, w), comb, detections, top_k=4)
        
        print(f"\n==========================================")
        print(f"TERRAIN: {p.stem.upper()}")
        print(f"Resolution: {w}x{h} | Hazards Detected: {len(detections)}")
        print(f"==========================================")
        
        for lz in lzs:
            cx, cy = lz["center"]
            # Verify distance to each hazard
            min_dist_to_hazard = float("inf")
            for d in detections:
                dist = np.hypot(cx - d["center"][0], cy - d["center"][1]) - d["radius"]
                if dist < min_dist_to_hazard:
                    min_dist_to_hazard = dist
                    
            is_in_blank_safe_area = min_dist_to_hazard >= lander_rad
            status_text = "PASSED (BLANK CLEAR AREA)" if is_in_blank_safe_area else "WARNING (NEAR HAZARD)"
            
            print(f"  * {lz['id']} [{lz['designation']}]: ({cx}, {cy})")
            print(f"    - Safety Score: {lz['safety_score']}%")
            print(f"    - Min Hazard Distance: {min_dist_to_hazard:.1f} px (Required: >={lander_rad} px)")
            print(f"    - Safety Buffer Margin: {lz['safety_margin_px']:.1f} px")
            print(f"    - Blank Area Verification: {status_text}")
            assert is_in_blank_safe_area, f"Landing spot {lz['id']} violated obstacle clearance buffer!"

    print("\nAll landing spot blank area clearance tests PASSED successfully!")

if __name__ == "__main__":
    run_tests()

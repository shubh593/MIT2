import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from backend.config import DEFAULT_PARAMS

class LandingZoneOptimizer:
    """
    Identifies, scores, and ranks optimal planetary landing sites based on
    lander physical footprint, obstacle clearance, terrain flatness, and approach paths.
    """

    def __init__(
        self,
        lander_radius_px: int = DEFAULT_PARAMS["lander_radius_px"],
        safety_margin_px: int = DEFAULT_PARAMS["safety_margin_px"]
    ):
        self.lander_radius_px = lander_radius_px
        self.safety_margin_px = safety_margin_px
        self.total_required_radius = lander_radius_px + safety_margin_px

    def find_safe_zones(
        self,
        image_shape: Tuple[int, int],
        risk_surface: np.ndarray,
        detections: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Calculates safe candidate landing sites using distance transform and local aperture scoring.
        """
        h, w = image_shape[:2]
        
        # 1. Create binary obstacle mask (1 for hazard, 0 for clear)
        hazard_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill bounding boxes and circles of detected hazards
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = int(d["center"][0]), int(d["center"][1])
            rad = int(d["radius"])
            # Fill rectangular obstacle box
            cv2.rectangle(hazard_mask, (max(0, x1 - 4), max(0, y1 - 4)), (min(w, x2 + 4), min(h, y2 + 4)), 255, -1)
            # Also fill circular footprint
            cv2.circle(hazard_mask, (cx, cy), rad + 6, 255, -1)
            
        # Also mark regions with high surface roughness/risk > 0.50
        hazard_mask[risk_surface > 0.50] = 255
        
        # Border margins (prevent landing too close to camera/sensor edge)
        edge_margin = self.total_required_radius
        hazard_mask[:edge_margin, :] = 255
        hazard_mask[-edge_margin:, :] = 255
        hazard_mask[:, :edge_margin] = 255
        hazard_mask[:, -edge_margin:] = 255
        
        # 2. Compute Euclidean Distance Transform from hazards
        clear_space = cv2.bitwise_not(hazard_mask)
        dist_map = cv2.distanceTransform(clear_space, cv2.DIST_L2, 5)
        
        # 3. Find candidate landing centers using sliding grid and local maxima
        step = max(10, int(self.lander_radius_px * 0.5))
        candidates = []
        
        img_center_x, img_center_y = w / 2.0, h / 2.0
        max_center_dist = np.hypot(img_center_x, img_center_y)
        
        # Target minimum clearance (lander radius + safety buffer)
        min_clearance_required = self.total_required_radius
        
        # Check if any pixels satisfy the full clearance
        if np.max(dist_map) < min_clearance_required:
            min_clearance_required = self.lander_radius_px # Fallback if terrain is very dense
        
        for y in range(edge_margin, h - edge_margin, step):
            for x in range(edge_margin, w - edge_margin, step):
                dist_to_hazard = float(dist_map[y, x])
                
                # Must satisfy clear blank terrain requirement
                if dist_to_hazard >= min_clearance_required:
                    # Extract local footprint patch
                    x1, y1 = max(0, x - self.lander_radius_px), max(0, y - self.lander_radius_px)
                    x2, y2 = min(w, x + self.lander_radius_px), min(h, y + self.lander_radius_px)
                    local_patch = risk_surface[y1:y2, x1:x2]
                    
                    mean_local_risk = float(np.mean(local_patch)) if local_patch.size > 0 else 1.0
                    roughness_std = float(np.std(local_patch)) if local_patch.size > 0 else 1.0
                    
                    # Clearance score (0 to 1) - higher distance margin is better
                    clearance_score = min(1.0, dist_to_hazard / (self.total_required_radius * 2.0))
                    
                    # Flatness score (0 to 1) - low risk and low variance
                    flatness_score = max(0.0, 1.0 - (mean_local_risk * 0.7 + roughness_std * 0.3))
                    
                    # Approach trajectory score (slight preference for accessible sectors)
                    dist_to_center = np.hypot(x - img_center_x, y - img_center_y)
                    trajectory_score = max(0.4, 1.0 - (dist_to_center / max_center_dist) * 0.5)
                    
                    # Composite Safety Score (0 - 100%)
                    raw_score = (clearance_score * 0.50) + (flatness_score * 0.35) + (trajectory_score * 0.15)
                    safety_score = round(float(raw_score * 100), 1)
                    
                    candidates.append({
                        "center": [x, y],
                        "radius": self.lander_radius_px,
                        "clearance_distance_px": round(dist_to_hazard, 1),
                        "safety_margin_px": round(dist_to_hazard - self.lander_radius_px, 1),
                        "safety_score": safety_score,
                        "flatness_index": round(flatness_score * 100, 1),
                        "local_risk_index": round(mean_local_risk * 100, 1),
                        "trajectory_score": round(trajectory_score * 100, 1),
                    })
                    
        # Sort candidates by safety score descending
        candidates.sort(key=lambda c: c["safety_score"], reverse=True)
        
        # Non-maximum suppression for candidate landing sites (avoid clustered overlapping zones)
        selected_zones = []
        min_site_separation = self.total_required_radius * 1.5
        
        for cand in candidates:
            cx, cy = cand["center"]
            # Check overlap with already selected
            too_close = False
            for sel in selected_zones:
                sx, sy = sel["center"]
                if np.hypot(cx - sx, cy - sy) < min_site_separation:
                    too_close = True
                    break
                    
            if not too_close:
                rank = len(selected_zones) + 1
                cand["rank"] = rank
                cand["id"] = f"LZ-{rank:02d}"
                
                if rank == 1:
                    cand["designation"] = "PRIMARY TARGET"
                    cand["category"] = "Optimal"
                    cand["theme_color"] = "emerald"
                elif rank == 2:
                    cand["designation"] = "SECONDARY SITE"
                    cand["category"] = "Viable"
                    cand["theme_color"] = "cyan"
                elif rank == 3:
                    cand["designation"] = "CONTINGENCY SITE"
                    cand["category"] = "Backup"
                    cand["theme_color"] = "amber"
                else:
                    cand["designation"] = f"ALTERNATE #{rank}"
                    cand["category"] = "Alternate"
                    cand["theme_color"] = "slate"
                    
                selected_zones.append(cand)
                if len(selected_zones) >= top_k:
                    break
                    
        return selected_zones

    @staticmethod
    def evaluate_point_risk(
        x: int,
        y: int,
        lander_radius: int,
        risk_surface: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Interactive probe: Evaluates risk telemetry at any arbitrary coordinate clicked by user.
        """
        h, w = risk_surface.shape[:2]
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        
        point_risk = float(risk_surface[y, x])
        
        # Find distance to nearest detected obstacle
        nearest_hazard = None
        min_dist = float("inf")
        
        for d in detections:
            cx, cy = d["center"]
            dist = np.hypot(x - cx, y - cy) - d["radius"]
            if dist < min_dist:
                min_dist = dist
                nearest_hazard = d
                
        min_dist_clamped = max(0.0, float(min_dist))
        
        # Local patch risk around point
        x1, y1 = max(0, x - lander_radius), max(0, y - lander_radius)
        x2, y2 = min(w, x + lander_radius), min(h, y + lander_radius)
        patch = risk_surface[y1:y2, x1:x2]
        footprint_risk = float(np.mean(patch)) if patch.size > 0 else point_risk
        
        landing_safety_pct = max(0.0, min(100.0, (1.0 - footprint_risk) * 100))
        
        if landing_safety_pct >= 75:
            status = "TOUCHDOWN NOMINAL (SAFE)"
            color = "emerald"
        elif landing_safety_pct >= 50:
            status = "TOUCHDOWN MARGINAL (CAUTION)"
            color = "amber"
        else:
            status = "TOUCHDOWN CRITICAL (HAZARDOUS)"
            color = "ruby"
            
        return {
            "coordinates": {"x": x, "y": y},
            "point_risk_pct": round(point_risk * 100, 1),
            "footprint_risk_pct": round(footprint_risk * 100, 1),
            "landing_safety_pct": round(landing_safety_pct, 1),
            "nearest_hazard_dist_px": round(min_dist_clamped, 1),
            "nearest_hazard_class": nearest_hazard["class_name"] if nearest_hazard else "None",
            "nearest_hazard_confidence": nearest_hazard["confidence"] if nearest_hazard else 0.0,
            "status": status,
            "status_color": color,
        }

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from backend.config import CLASS_RISK_WEIGHTS

class TerrainRiskAnalyzer:
    """
    Computes spatial risk density, slope/roughness approximations,
    and landing safety surfaces from detected hazards and raw topography.
    """

    @staticmethod
    def compute_hazard_heatmap(
        image_shape: Tuple[int, int],
        detections: List[Dict[str, Any]],
        sigma_scale: float = 0.6
    ) -> np.ndarray:
        """
        Creates a 2D spatial continuous risk heatmap in range [0.0, 1.0].
        Uses Gaussian falloff centered around detected obstacles.
        """
        h, w = image_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Grid coordinate matrices
        y_grid, x_grid = np.ogrid[:h, :w]
        
        for det in detections:
            cx, cy = det["center"]
            radius = max(det["radius"], 10.0)
            risk = det.get("risk_score", 0.8)
            sigma = radius * sigma_scale
            
            # Distance squared from obstacle center
            dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            
            # Gaussian hazard influence
            influence = risk * np.exp(-dist_sq / (2 * (sigma ** 2) + 1e-6))
            
            # Accumulate with saturation clamp
            heatmap = np.maximum(heatmap, influence)
            
        return np.clip(heatmap, 0.0, 1.0)

    @staticmethod
    def estimate_terrain_roughness(image_bgr: np.ndarray) -> np.ndarray:
        """
        Estimates surface roughness & slope variation using Laplacian & Sobel gradients.
        High texture variations indicate jagged rocks, boulders, or crater rims.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Gradient magnitude (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Surface local variance / roughness
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        roughness = np.abs(laplacian) * 0.5 + magnitude * 0.5
        
        # Normalize to [0.0, 1.0]
        norm_roughness = cv2.normalize(roughness, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return norm_roughness

    @staticmethod
    def generate_combined_risk_surface(
        hazard_heatmap: np.ndarray,
        roughness_map: np.ndarray,
        hazard_weight: float = 0.75,
        roughness_weight: float = 0.25
    ) -> np.ndarray:
        """
        Combines AI hazard detections with photometric surface roughness.
        """
        combined = (hazard_heatmap * hazard_weight) + (roughness_map * roughness_weight)
        return np.clip(combined, 0.0, 1.0)

    @staticmethod
    def summarize_risk(
        risk_surface: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates overall terrain risk metrics and telemetry statistics.
        """
        mean_risk = float(np.mean(risk_surface))
        max_risk = float(np.max(risk_surface))
        
        # Percentage of terrain with risk > 0.45
        hazard_coverage_pct = float(np.sum(risk_surface > 0.45) / risk_surface.size * 100)
        
        # Class distribution
        class_counts = {}
        for d in detections:
            cname = d.get("class_name", "hazard")
            class_counts[cname] = class_counts.get(cname, 0) + 1
            
        # Global Mission Feasibility Rating
        if hazard_coverage_pct < 20 and mean_risk < 0.25:
            overall_verdict = "GO"
            verdict_detail = "Optimal Planetary Surface: Abundant safe landing corridors detected."
            verdict_color = "emerald"
        elif hazard_coverage_pct < 50 and mean_risk < 0.50:
            overall_verdict = "CAUTION"
            verdict_detail = "Moderate Hazard Density: Precision autonomous guidance required."
            verdict_color = "amber"
        else:
            overall_verdict = "NO-GO"
            verdict_detail = "Severe Hazard Density: Terrain exceeds safe touchdown clearance limits."
            verdict_color = "ruby"

        return {
            "overall_verdict": overall_verdict,
            "verdict_detail": verdict_detail,
            "verdict_color": verdict_color,
            "mean_risk_index": round(mean_risk * 100, 1),
            "max_risk_index": round(max_risk * 100, 1),
            "hazard_coverage_pct": round(hazard_coverage_pct, 1),
            "total_obstacles": len(detections),
            "class_breakdown": class_counts,
        }

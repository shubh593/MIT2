import math
import numpy as np
from typing import Dict, Any, List

class DescentTrajectorySimulator:
    """
    Simulates real-time terminal descent trajectory of planetary lander
    with altitude step-down, sensor field-of-view contraction, and hazard collision avoidance telemetry.
    """

    @staticmethod
    def simulate_descent_profile(
        target_landing_zone: Dict[str, Any],
        initial_position: Dict[str, float] = None,
        altitude_start_m: float = 1200.0,
        steps: int = 25,
        terrain_width_m: float = 2000.0
    ) -> Dict[str, Any]:
        """
        Generates simulated time-series telemetry steps for lander approach.
        """
        tx, ty = target_landing_zone["center"]
        target_score = target_landing_zone["safety_score"]
        
        # If initial position not provided, start offset with slight deviation
        if initial_position is None:
            init_x = float(tx + np.random.uniform(-180, 180))
            init_y = float(ty + np.random.uniform(-180, 180))
        else:
            init_x = float(initial_position.get("x", tx + 100))
            init_y = float(initial_position.get("y", ty + 100))
            
        telemetry_frames = []
        time_step = 2.0  # seconds per frame
        
        # Flight physics approximations
        for i in range(steps):
            t = i * time_step
            progress = i / (steps - 1)  # 0.0 to 1.0
            
            # Smooth descent curve (Hermite spline / cosine blend)
            blend = 0.5 * (1.0 - math.cos(math.pi * progress))
            
            # Current altitude (m)
            alt = round(altitude_start_m * (1.0 - blend), 1)
            
            # Current coordinates (interpolating towards target)
            cur_x = round(init_x + (tx - init_x) * blend, 1)
            cur_y = round(init_y + (ty - init_y) * blend, 1)
            
            # Descent velocity (m/s)
            vel_z = round(35.0 * (1.0 - progress * 0.85) + np.random.uniform(-0.5, 0.5), 2)
            if alt < 10:
                vel_z = round(max(0.4, vel_z * 0.1), 2)
                
            # Horizontal drift velocity (m/s)
            drift_x = round((tx - cur_x) * 0.08, 2)
            drift_y = round((ty - cur_y) * 0.08, 2)
            
            # Fuel remaining percentage
            fuel_pct = round(100.0 - progress * 42.0, 1)
            
            # Thruster gimbal pitch angle (degrees)
            pitch_deg = round(math.sin(progress * math.pi * 3) * (1.0 - progress) * 4.5, 1)
            
            # Radar altimeter lock status
            radar_lock = True if alt < 1000 else (i % 2 == 0)
            
            # Hazard proximity alert level
            dist_to_target = math.hypot(cur_x - tx, cur_y - ty)
            if alt < 80 and dist_to_target < 20:
                alert_level = "GREEN_OPTIMAL"
                guidance_msg = "TERMINAL TOUCHDOWN VELOCITY CONFIRMED"
            elif alt < 250:
                alert_level = "YELLOW_CORRIDOR"
                guidance_msg = "FINE ATTITUDE ALIGNMENT ACTIVE"
            else:
                alert_level = "NOMINAL"
                guidance_msg = "APPROACH TRAJECTORY TRACKED"
                
            telemetry_frames.append({
                "step": i + 1,
                "time_s": round(t, 1),
                "altitude_m": alt,
                "position": {"x": cur_x, "y": cur_y},
                "velocity_z_mps": vel_z,
                "drift_x_mps": drift_x,
                "drift_y_mps": drift_y,
                "pitch_deg": pitch_deg,
                "fuel_pct": fuel_pct,
                "radar_lock": radar_lock,
                "alert_level": alert_level,
                "guidance_msg": guidance_msg,
            })
            
        return {
            "simulation_id": f"SIM_TRAJ_{int(alt)}",
            "target_zone": target_landing_zone["id"],
            "target_coordinates": {"x": tx, "y": ty},
            "initial_altitude_m": altitude_start_m,
            "total_duration_s": round(steps * time_step, 1),
            "final_safety_score": target_score,
            "trajectory_status": "SUCCESSFUL_TOUCHDOWN_SIMULATED",
            "frames": telemetry_frames,
        }

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.analytics.trajectory_sim import DescentTrajectorySimulator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Autonomous Descent Simulation"])

class SimulateRequest(BaseModel):
    landing_zone: Dict[str, Any]
    initial_position: Optional[Dict[str, float]] = None
    altitude_start_m: Optional[float] = 1200.0
    steps: Optional[int] = 30

@router.post("/simulate-descent")
async def simulate_descent_trajectory(req: SimulateRequest):
    """
    Simulates real-time autonomous lander descent telemetry to a target Landing Zone.
    """
    if not req.landing_zone or "center" not in req.landing_zone:
        raise HTTPException(status_code=400, detail="Invalid target landing zone provided.")
        
    sim_data = DescentTrajectorySimulator.simulate_descent_profile(
        target_landing_zone=req.landing_zone,
        initial_position=req.initial_position,
        altitude_start_m=req.altitude_start_m or 1200.0,
        steps=req.steps or 30
    )
    
    return sim_data

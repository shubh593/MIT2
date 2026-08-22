import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def inspect_onnx_model(onnx_path: Path) -> Dict[str, Any]:
    """Inspect ONNX model inputs, outputs, and metadata."""
    info = {
        "format": "ONNX",
        "path": str(onnx_path),
        "exists": onnx_path.exists(),
        "size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2) if onnx_path.exists() else 0,
        "input_shape": None,
        "output_shape": None,
        "classes": [],
        "names": {},
    }
    
    if not onnx_path.exists():
        return info

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        info["input_name"] = inputs[0].name if inputs else "input"
        info["input_shape"] = inputs[0].shape if inputs else []
        info["output_names"] = [o.name for o in outputs]
        info["output_shapes"] = [o.shape for o in outputs]
        
        # Check custom metadata
        meta = session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            try:
                # Often stored as json string or python dict string
                names_str = meta["names"].replace("'", '"')
                info["names"] = json.loads(names_str)
                info["classes"] = list(info["names"].values())
            except Exception:
                info["classes_raw"] = meta["names"]
                
    except Exception as e:
        logger.warning(f"Error inspecting ONNX model: {e}")
        info["error"] = str(e)
        
    return info

def inspect_pt_model(pt_path: Path) -> Dict[str, Any]:
    """Inspect PyTorch / Ultralytics YOLO model."""
    info = {
        "format": "PyTorch (.pt)",
        "path": str(pt_path),
        "exists": pt_path.exists(),
        "size_mb": round(pt_path.stat().st_size / (1024 * 1024), 2) if pt_path.exists() else 0,
        "classes": [],
        "names": {},
    }
    
    if not pt_path.exists():
        return info
        
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        if hasattr(model, "names") and model.names:
            info["names"] = model.names
            info["classes"] = list(model.names.values())
        if hasattr(model, "task"):
            info["task"] = model.task
    except Exception as e:
        logger.warning(f"Error inspecting PyTorch model with Ultralytics: {e}")
        # Fallback torch inspect
        try:
            import torch
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                m = ckpt["model"]
                if hasattr(m, "names"):
                    info["names"] = m.names
                    info["classes"] = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
        except Exception as e2:
            info["error"] = f"{e} | Fallback: {e2}"
            
    return info

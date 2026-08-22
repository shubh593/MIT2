import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from backend.config import MODEL_PT_PATH, MODEL_ONNX_PATH, DEFAULT_PARAMS, CLASS_RISK_WEIGHTS

logger = logging.getLogger(__name__)

class PlanetaryHazardDetector:
    """
    Dual-engine planetary terrain obstacle detector.
    Supports ONNX Runtime (fast edge inference) and PyTorch (Ultralytics).
    """

    def __init__(
        self,
        onnx_path: Path = MODEL_ONNX_PATH,
        pt_path: Path = MODEL_PT_PATH,
        prefer_engine: str = "auto"
    ):
        self.onnx_path = Path(onnx_path)
        self.pt_path = Path(pt_path)
        self.prefer_engine = prefer_engine
        
        self.ort_session = None
        self.pt_model = None
        self.active_engine = "none"
        self.class_names: Dict[int, str] = {}
        self.input_size = (640, 640)
        
        self._init_models()

    def _init_models(self):
        """Initialize ONNX and/or PyTorch model engines."""
        # Try ONNX first if preferred or auto
        if self.prefer_engine in ("onnx", "auto") and self.onnx_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                
                # Extract input shape
                inp = self.ort_session.get_inputs()[0]
                if len(inp.shape) >= 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                    self.input_size = (inp.shape[2], inp.shape[3])
                
                # Extract metadata classes
                meta = self.ort_session.get_modelmeta().custom_metadata_map
                if "names" in meta:
                    try:
                        import json
                        names_raw = meta["names"].replace("'", '"')
                        names_dict = json.loads(names_raw)
                        self.class_names = {int(k): str(v) for k, v in names_dict.items()}
                    except Exception:
                        pass
                
                self.active_engine = "onnx"
                logger.info(f"Initialized ONNX engine successfully with input size {self.input_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX runtime: {e}")

        # Try PyTorch model if preferred or load both for immediate switching
        if self.pt_path.exists():
            try:
                from ultralytics import YOLO
                self.pt_model = YOLO(str(self.pt_path))
                if hasattr(self.pt_model, "names") and self.pt_model.names:
                    for k, v in self.pt_model.names.items():
                        if str(v) != "0":
                            self.class_names[int(k)] = str(v)
                if self.active_engine == "none":
                    self.active_engine = "pt"
                logger.info(f"Initialized PyTorch Ultralytics engine successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize PyTorch YOLO model: {e}")

        # If class names are empty or default to digits like '0', map to planetary taxonomy
        if not self.class_names or list(self.class_names.values()) == ["0"] or "0" in self.class_names.values():
            self.class_names = {
                0: "crater_hazard",
                1: "boulder",
                2: "rock",
                3: "rough_surface"
            }

    def _preprocess_onnx(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and pad image (letterbox) to model input dimensions."""
        h, w = image_bgr.shape[:2]
        target_h, target_w = self.input_size
        
        # Calculate scale and padding
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Normalize and change channel order to NCHW RGB
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, (pad_x, pad_y)

    def _postprocess_onnx(
        self,
        output: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Convert raw YOLO ONNX output [1, C, N] or [1, N, C] to bounding box detections."""
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad
        
        preds = output[0]  # Shape could be [84, 8400] or [8400, 84] or [N, 6]
        if preds.shape[0] < preds.shape[1]:
            preds = np.transpose(preds, (1, 0))  # Ensure [N, features]
            
        num_features = preds.shape[1]
        num_classes = num_features - 4
        
        boxes = []
        confidences = []
        class_ids = []
        
        for row in preds:
            # Check format: [cx, cy, w, h, cls0, cls1, ...] or [cx, cy, w, h, obj_conf, cls0, ...]
            if num_classes > 0:
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                score = float(scores[cls_id])
            else:
                score = float(row[4]) if len(row) > 4 else 0.5
                cls_id = 0
                
            if score >= conf_threshold:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Unpad and scale back to original image
                x1 = (cx - w / 2 - pad_x) / scale
                y1 = (cy - h / 2 - pad_y) / scale
                x2 = (cx + w / 2 - pad_x) / scale
                y2 = (cy + h / 2 - pad_y) / scale
                
                # Clip to image boundaries
                x1 = max(0, min(orig_w - 1, x1))
                y1 = max(0, min(orig_h - 1, y1))
                x2 = max(0, min(orig_w - 1, x2))
                y2 = max(0, min(orig_h - 1, y2))
                
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(score))
                class_ids.append(cls_id)
                
        # Non-maximum suppression (NMS)
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, bw, bh = boxes[idx]
                    cls_id = class_ids[idx]
                    cls_name = self.class_names.get(cls_id, f"hazard_{cls_id}")
                    score = confidences[idx]
                    
                    # Risk multiplier based on class
                    risk_weight = CLASS_RISK_WEIGHTS.get(cls_name.lower(), 0.8)
                    
                    detections.append({
                        "id": f"hz_{len(detections) + 1}",
                        "bbox": [x, y, x + bw, y + bh],  # [x1, y1, x2, y2]
                        "center": [x + bw / 2, y + bh / 2],
                        "width": bw,
                        "height": bh,
                        "radius": max(bw, bh) / 2.0,
                        "confidence": round(score, 3),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "risk_score": round(score * risk_weight, 3),
                    })
                    
        return detections

    def _detect_pytorch(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Inference with Ultralytics YOLO PyTorch model."""
        orig_h, orig_w = image_bgr.shape[:2]
        results = self.pt_model(image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"hazard_{cls_id}")
                
                bw = x2 - x1
                bh = y2 - y1
                risk_weight = CLASS_RISK_WEIGHTS.get(cls_name.lower(), 0.8)
                
                detections.append({
                    "id": f"hz_{i + 1}",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [float(x1 + bw / 2), float(y1 + bh / 2)],
                    "width": int(bw),
                    "height": int(bh),
                    "radius": float(max(bw, bh) / 2.0),
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "risk_score": round(conf * risk_weight, 3),
                })
                
        return detections

    def _fallback_cv_detector(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computer Vision fallback for detecting circular craters and high-contrast rocks
        if deep learning engine yields zero detections (e.g. low-texture synthetic moon maps).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape
        
        detections = []
        
        # 1. Crater detection via Hough Circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=int(min(h, w) * 0.45)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (cx, cy, r) in enumerate(circles[0, :8]):  # Take top 8 prominent circles
                x1 = max(0, int(cx - r))
                y1 = max(0, int(cy - r))
                x2 = min(w - 1, int(cx + r))
                y2 = min(h - 1, int(cy + r))
                
                detections.append({
                    "id": f"cv_crater_{i+1}",
                    "bbox": [x1, y1, x2, y2],
                    "center": [float(cx), float(cy)],
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "radius": float(r),
                    "confidence": 0.82,
                    "class_id": 0,
                    "class_name": "crater",
                    "risk_score": 0.88,
                })
                
        # 2. Rock / Boulder detection via Adaptive Threshold & Contours
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if 80 < area < 4000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / (bh + 1e-5)
                if 0.5 < aspect < 2.0:
                    detections.append({
                        "id": f"cv_rock_{len(detections)+1}",
                        "bbox": [bx, by, bx + bw, by + bh],
                        "center": [float(bx + bw / 2), float(by + bh / 2)],
                        "width": int(bw),
                        "height": int(bh),
                        "radius": float(max(bw, bh) / 2.0),
                        "confidence": 0.75,
                        "class_id": 1,
                        "class_name": "rock",
                        "risk_score": 0.72,
                    })
                    if len(detections) >= 25:
                        break

        return detections

    def detect(
        self,
        image_bgr: np.ndarray,
        engine: str = "auto",
        conf_threshold: float = DEFAULT_PARAMS["conf_threshold"],
        iou_threshold: float = DEFAULT_PARAMS["iou_threshold"],
        enable_cv_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Run obstacle detection on the input image.
        Returns detection list, execution time, and model telemetry.
        """
        start_time = time.perf_counter()
        orig_shape = image_bgr.shape[:2]
        engine_used = "none"
        detections = []
        
        target_engine = engine if engine in ("onnx", "pt") else self.active_engine
        
        # Run ONNX
        if target_engine == "onnx" and self.ort_session is not None:
            try:
                tensor, scale, pad = self._preprocess_onnx(image_bgr)
                input_name = self.ort_session.get_inputs()[0].name
                raw_outputs = self.ort_session.run(None, {input_name: tensor})
                detections = self._postprocess_onnx(
                    raw_outputs[0], scale, pad, orig_shape, conf_threshold, iou_threshold
                )
                engine_used = "ONNX Runtime (best.onnx)"
            except Exception as e:
                logger.error(f"ONNX inference failed: {e}")
                
        # Run PyTorch if ONNX was not used or failed
        if not detections and (target_engine == "pt" or self.pt_model is not None):
            try:
                detections = self._detect_pytorch(image_bgr, conf_threshold, iou_threshold)
                engine_used = "PyTorch YOLO (best.pt)"
            except Exception as e:
                logger.error(f"PyTorch inference failed: {e}")

        # If zero detections and fallback allowed, invoke CV fallback detector
        if not detections and enable_cv_fallback:
            detections = self._fallback_cv_detector(image_bgr)
            engine_used = f"{engine_used} + CV Terrain Heuristics" if engine_used != "none" else "CV Terrain Fallback"
            
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "engine": engine_used,
            "inference_time_ms": latency_ms,
            "hazards_detected": len(detections),
            "detections": detections,
            "image_dimensions": {"width": orig_shape[1], "height": orig_shape[0]},
        }

from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import os
import sys

# Ensure /tmp is writable for model cache
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"
os.environ["ULTRALYTICS_DIR"] = "/tmp/ultralytics"

import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# ── Model (loaded once per cold start) ───────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
        _model.to("cpu")
    return _model

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_image(pil_img: Image.Image, max_size: int = 640) -> np.ndarray:
    w, h = pil_img.size
    scale = min(max_size / max(w, h), 1.0)
    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def run_detection(model, bgr_img: np.ndarray, conf: float = 0.35):
    return model(bgr_img, conf=conf, device="cpu", verbose=False)


def draw_boxes(bgr_img: np.ndarray, results) -> np.ndarray:
    img = bgr_img.copy()
    accent = (0, 229, 160)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = r.names[int(box.cls[0])]
            text = f"{label} {conf:.0%}"
            cv2.rectangle(img, (x1, y1), (x2, y2), accent, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), accent, -1)
            cv2.putText(img, text, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (13, 13, 13), 1, cv2.LINE_AA)
    return img


def build_detections(results):
    from collections import defaultdict
    class_data = defaultdict(lambda: {"count": 0, "max_conf": 0.0})
    total = 0
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            class_data[label]["count"] += 1
            class_data[label]["max_conf"] = max(class_data[label]["max_conf"], conf)
            total += 1
    return {
        "total": total,
        "unique_classes": len(class_data),
        "objects": [
            {"label": lbl, "count": d["count"], "max_conf": round(d["max_conf"], 3)}
            for lbl, d in sorted(class_data.items(), key=lambda x: -x[1]["count"])
        ]
    }


# ── Vercel handler ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            image_b64 = body.get("image", "")
            conf = float(body.get("conf", 0.35))

            # Decode image
            image_data = base64.b64decode(image_b64)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")

            model = get_model()
            bgr = preprocess_image(pil_img)
            results = run_detection(model, bgr, conf)
            annotated = draw_boxes(bgr, results)
            detections = build_detections(results)

            # Encode annotated image as base64 PNG
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            pil_out.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            response = {"image": img_b64, **detections}
            self._respond(200, response)

        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

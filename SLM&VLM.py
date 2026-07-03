import cv2
import torch
import numpy as np
import math
import requests
import json
import time
import threading
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from deep_sort_realtime.deepsort_tracker import DeepSort

# Try import Ultralytics YOLO (yolov8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics YOLO not available.")

# --------------------------
# Config / params
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = "yolov8s.pt"
FOCAL_LENGTH = 640
STOP_THRESHOLD_M = 1.0
CONFIDENCE_THRESHOLD = 0.35

# -------------------------------------------------------
# HuggingFace API Config
# -------------------------------------------------------
# -------------------------------------------------------
# HuggingFace Inference Router Config (NEW as of 2025)
# -------------------------------------------------------
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type":  "application/json",
}

# Average real-world heights (cm) for distance estimation
OBJECT_REAL_SIZES = {
    "door":       200,
    "chair":       90,
    "person":     170,
    "table":       75,
    "car":        150,
    "refrigerator":170,
    "laptop":      30,
    "tv":         100,
    "bed":         60,
    "bottle":      25,
    "cup":         12,
    "backpack":    50,
    "suitcase":    70,
    "default":    100,
}

HAZARD_LABELS = {"person", "car", "truck", "wall", "fence"}

# --------------------------
# Initialize models
# --------------------------
print(f"[INFO] Using device: {DEVICE}")

processor  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(DEVICE)
blip_model.eval()

tracker  = DeepSort(max_age=30)
detector = None
if YOLO_AVAILABLE:
    detector = YOLO(YOLO_WEIGHTS)

# --------------------------
# SLM Decision State
# (runs in background thread so it doesn't block 30fps loop)
# --------------------------
slm_lock          = threading.Lock()
slm_decision      = "Move Forward"
slm_reason        = "Initializing SLM..."
slm_color         = (0, 255, 0)
slm_last_update   = 0
SLM_UPDATE_INTERVAL = 2.0     # seconds between SLM calls — tune as needed


def build_prompt(annotations, frame_width, target_obj=None):
    """
    Convert detection list into a short natural-language scene description
    and ask Phi-2 what the robot should do.
    """
    if not annotations:
        scene = "No objects detected in the scene."
    else:
        parts = []
        for ann in annotations:
            cls   = ann.get("cls", "object")
            dist  = ann.get("distance_m", float("inf"))
            state = ann.get("state", {})
            cx    = ann.get("center", (frame_width // 2, 0))[0]

            # Position
            third = frame_width // 3
            if cx < third:
                pos = "left"
            elif cx < 2 * third:
                pos = "center"
            else:
                pos = "right"

            dist_str = f"{dist:.2f}m" if dist != float("inf") else "unknown distance"

            # State details
            state_parts = []
            if "door_state"     in state: state_parts.append(state["door_state"])
            if "person_motion"  in state: state_parts.append(state["person_motion"])
            if "occupancy"      in state: state_parts.append(state["occupancy"])
            if "orientation"    in state: state_parts.append(state["orientation"])
            if "screen_state"   in state: state_parts.append(state["screen_state"])
            if "vehicle_motion" in state: state_parts.append(state["vehicle_motion"])
            if "fridge_state"   in state: state_parts.append(state["fridge_state"])
            if "attendance"     in state: state_parts.append(state["attendance"])

            state_str = f"({', '.join(state_parts)})" if state_parts else ""
            parts.append(f"{cls}{state_str} at {dist_str} on the {pos}")

        scene = "; ".join(parts) + "."

    target_line = f"The robot's goal is to find: {target_obj}." if target_obj else \
                  "The robot has no specific target — navigate safely indoors."

    prompt = (
        "You are an indoor robot navigation assistant.\n"
        f"Scene: {scene}\n"
        f"{target_line}\n"
        "Based on the scene, give a short navigation decision in this exact format:\n"
        "Decision: <action>\n"
        "Reason: <one sentence>\n"
        "Valid actions: Move Forward, Stop, Slow Down, Go Left, Go Right, "
        "Searching, Detected, Wait.\n"
        "Decision:"
    )
    return prompt


def parse_slm_response(raw_text):
    """
    Extract Decision + Reason from Phi-2 raw output.
    Falls back gracefully if format is unexpected.
    """
    text = raw_text.strip()

    decision = "Move Forward"
    reason   = text[:120]   # fallback: use raw output trimmed

    # Try to parse structured output
    lines = text.splitlines()
    for line in lines:
        l = line.strip()
        if l.lower().startswith("decision:"):
            decision = l.split(":", 1)[1].strip()
        elif l.lower().startswith("reason:"):
            reason = l.split(":", 1)[1].strip()

    # Map decision text to color
    d_low = decision.lower()
    if "stop" in d_low:
        color = (0, 0, 255)
    elif "slow" in d_low or "wait" in d_low:
        color = (0, 165, 255)
    elif "detected" in d_low:
        color = (0, 165, 255)
    elif "search" in d_low:
        color = (0, 200, 255)
    else:
        color = (0, 255, 0)

    return decision, reason, color


def call_phi2_api(prompt):
    """
    Call HuggingFace Inference Router (new API, OpenAI-compatible format).
    """
    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 80,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        elif resp.status_code == 503:
            return "Decision: Move Forward\nReason: Model loading, please wait."
        else:
            print(f"[HF API] Error {resp.status_code}: {resp.text[:200]}")
            return "Decision: Move Forward\nReason: API error, using safe default."
    except requests.exceptions.Timeout:
        print("[HF API] Timeout — using fallback decision")
        return "Decision: Slow Down\nReason: SLM timeout, proceeding cautiously."
    except Exception as e:
        print(f"[HF API] Exception: {e}")
        return "Decision: Move Forward\nReason: API unavailable."


def slm_worker(annotations_snapshot, frame_width, target_obj):
    """
    Background thread: call Phi-2 and update global decision state.
    """
    global slm_decision, slm_reason, slm_color

    prompt   = build_prompt(annotations_snapshot, frame_width, target_obj)
    raw_text = call_phi2_api(prompt)
    decision, reason, color = parse_slm_response(raw_text)

    with slm_lock:
        slm_decision = decision
        slm_reason   = reason
        slm_color    = color


# --------------------------
# Utility functions (unchanged from your original)
# --------------------------
def estimate_distance(bbox_height_px, real_height_cm):
    if bbox_height_px <= 0:
        return None
    return (real_height_cm * FOCAL_LENGTH) / float(bbox_height_px)


def get_position_description(center_x, frame_width):
    third = frame_width // 3
    if center_x < third:
        return "Left"
    elif center_x < 2 * third:
        return "Center"
    else:
        return "Right"


def screen_origin(frame_w, frame_h, y_offset=10):
    return (frame_w // 2, frame_h - y_offset)


def draw_translucent_path(frame, origin, target_center, bbox_w_px,
                           color=(0, 255, 255), alpha=0.25):
    overlay  = frame.copy()
    cx, cy   = target_center
    half_w   = max(6, int(bbox_w_px // 2))
    pts = np.array([
        [origin[0], origin[1]],
        [cx - half_w, cy],
        [cx + half_w, cy]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_arrow_and_angle(frame, origin, target_center,
                          color=(0, 255, 255), thickness=2):
    ox, oy = origin
    tx, ty = target_center
    cv2.arrowedLine(frame, (ox, oy), (tx, ty), color, thickness, tipLength=0.15)

    dx = tx - ox
    dy = oy - ty
    if dx == 0 and dy == 0:
        angle_deg = 0.0
    else:
        angle_deg = math.degrees(math.atan2(dx, dy))

    abs_angle = abs(angle_deg)
    if abs_angle < 7:
        steer_text = "Straight"
    elif angle_deg > 0:
        steer_text = f"Steer Right {abs_angle:.0f}deg"
    else:
        steer_text = f"Steer Left {abs_angle:.0f}deg"

    cv2.putText(frame, steer_text,
                (origin[0] - 160, origin[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return angle_deg


# --------------------------
# Indoor object state helpers
# --------------------------
prev_centers = {}

def infer_motion(obj_id, cx, cy):
    prev = prev_centers.get(obj_id)
    prev_centers[obj_id] = (cx, cy)
    if prev is None:
        return "unknown"
    dy = cy - prev[1]
    return "moving" if abs(dy) >= 1 else "stationary"


def simple_door_state(bbox_height, frame_height):
    ratio = bbox_height / frame_height
    if ratio < 0.35:
        return "open"
    elif ratio > 0.5:
        return "closed"
    return "unknown"


def chair_occupancy(chair_ann, all_annotations):
    """Check if a person bbox significantly overlaps the chair bbox."""
    cx1, cy1, cx2, cy2 = chair_ann["xyxy"]
    for ann in all_annotations:
        if ann["cls"] != "person":
            continue
        px1, py1, px2, py2 = ann["xyxy"]
        # Intersection
        ix1, iy1 = max(cx1, px1), max(cy1, py1)
        ix2, iy2 = min(cx2, px2), min(cy2, py2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            chair_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if inter_area / chair_area > 0.25:
                return "occupied"
    return "empty"


def table_clutter(table_ann, all_annotations):
    """Check if any object center falls inside the table bbox."""
    tx1, ty1, tx2, ty2 = table_ann["xyxy"]
    for ann in all_annotations:
        if ann["cls"] in ("table", "person"):
            continue
        cx, cy = ann.get("center", (0, 0))
        if tx1 < cx < tx2 and ty1 < cy < ty2:
            return "cluttered"
    return "clear"


def screen_state(crop_bgr):
    """Laptop / TV screen on or off via brightness."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return "on" if mean_brightness > 60 else "off"


def bottle_orientation(w, h):
    return "knocked_over" if w > h else "upright"


def bag_attendance(bag_ann, all_annotations, proximity_px=150):
    """Is the bag near a person?"""
    bx = (bag_ann["xyxy"][0] + bag_ann["xyxy"][2]) // 2
    by = (bag_ann["xyxy"][1] + bag_ann["xyxy"][3]) // 2
    for ann in all_annotations:
        if ann["cls"] != "person":
            continue
        px, py = ann.get("center", (0, 0))
        if math.hypot(bx - px, by - py) < proximity_px:
            return "attended"
    return "unattended"


def fridge_state(bbox_width, frame_width):
    """Wide bounding box often means door is swung open."""
    ratio = bbox_width / frame_width
    return "open" if ratio > 0.35 else "closed"


def get_indoor_state(cls_name, ann, all_annotations, crop, frame_h, frame_w):
    """
    Master dispatcher — returns state dict for any indoor object.
    """
    state = {}
    x1, y1, x2, y2 = ann["xyxy"]
    cx, cy = ann.get("center", ((x1+x2)//2, (y1+y2)//2))
    obj_id = f"{cls_name}_{id(ann)}"

    if cls_name == "door":
        state["door_state"] = simple_door_state(y2 - y1, frame_h)

    elif cls_name == "person":
        state["person_motion"] = infer_motion(obj_id, cx, cy)

    elif cls_name in ("chair", "sofa", "couch", "bench"):
        state["occupancy"] = chair_occupancy(ann, all_annotations)

    elif cls_name == "table":
        state["occupancy"] = table_clutter(ann, all_annotations)

    elif cls_name == "bed":
        state["occupancy"] = chair_occupancy(ann, all_annotations)   # reuse overlap logic

    elif cls_name in ("laptop", "tv", "monitor"):
        if crop is not None and crop.size > 0:
            state["screen_state"] = screen_state(crop)

    elif cls_name in ("bottle", "cup", "bowl"):
        w = x2 - x1
        h = y2 - y1
        state["orientation"] = bottle_orientation(w, h)

    elif cls_name in ("backpack", "suitcase", "handbag"):
        state["attendance"] = bag_attendance(ann, all_annotations)

    elif cls_name in ("car", "truck", "motorcycle", "bicycle"):
        state["vehicle_motion"] = infer_motion(obj_id, cx, cy)

    elif cls_name == "refrigerator":
        state["fridge_state"] = fridge_state(x2 - x1, frame_w)

    return state


# --------------------------
# Hard safety fallback
# (runs every frame — overrides SLM if immediate danger)
# --------------------------
def hard_safety_check(annotations, frame_width, stop_threshold_m=STOP_THRESHOLD_M):
    """
    Returns (decision, reason, color) if an immediate hazard exists,
    else returns None so SLM decision is used.
    """
    for ann in annotations:
        cls  = ann.get("cls", "")
        dist = ann.get("distance_m", float("inf"))
        if any(h in cls for h in HAZARD_LABELS) and dist <= stop_threshold_m:
            cx  = ann.get("center", (frame_width // 2, 0))[0]
            pos = get_position_description(cx, frame_width)
            return (
                "STOP!",
                f"Immediate hazard: {cls} within {dist:.2f}m ({pos})",
                (0, 0, 255)
            )
    return None


# --------------------------
# Main loop
# --------------------------
def main():
    global slm_decision, slm_reason, slm_color, slm_last_update

    url = "http://192.0.0.4:8080/video"
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    user_command = input("Enter navigation command (e.g., 'find door', blank for all): ").strip().lower()
    target_obj   = None
    if user_command:
        for obj in ["door", "chair", "person", "table", "car",
                    "laptop", "bottle", "refrigerator", "bed", "backpack"]:
            if obj in user_command:
                target_obj = obj
                break

    print(f"[INFO] Target object : {target_obj or 'None (general navigation)'}")
    print(f"[INFO] SLM model     : {HF_MODEL_ID}")
    print(f"[INFO] SLM interval  : every {SLM_UPDATE_INTERVAL}s")
    print("[INFO] Hard safety override active every frame.")

    slm_thread = None     # background thread handle

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        frame_h, frame_w = frame.shape[:2]
        annotations      = []

        # -------- YOLO Detection --------
        if YOLO_AVAILABLE:
            try:
                results = detector(frame, device=0, imgsz=640)[0]
            except Exception:
                results = detector(frame)[0]

            raw_anns = []
            for box in results.boxes:
                conf = float(box.conf.cpu().numpy())
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                cls_id   = int(box.cls.cpu().numpy())
                xyxy     = box.xyxy[0].cpu().numpy().astype(int)
                x1,y1,x2,y2 = xyxy
                cls_name = detector.model.names[cls_id].lower()
                center   = ((x1+x2)//2, (y1+y2)//2)
                h_box    = max(1, y2 - y1)
                real_h   = OBJECT_REAL_SIZES.get(cls_name, OBJECT_REAL_SIZES["default"])
                dist_cm  = estimate_distance(h_box, real_h)
                dist_m   = dist_cm / 100.0 if dist_cm else float("inf")
                crop     = frame[y1:y2, x1:x2]

                raw_anns.append({
                    "cls":        cls_name,
                    "xyxy":       (x1, y1, x2, y2),
                    "center":     center,
                    "distance_m": dist_m,
                    "state":      {},
                    "crop":       crop,
                })

            # Second pass: compute context states (needs full list for relationships)
            for ann in raw_anns:
                crop = ann.pop("crop", None)
                ann["state"] = get_indoor_state(
                    ann["cls"], ann, raw_anns,
                    crop, frame_h, frame_w
                )
                annotations.append(ann)

        else:
            annotations.append({
                "cls":        "unknown",
                "xyxy":       (0, 0, frame_w, frame_h),
                "center":     (frame_w // 2, frame_h // 2),
                "distance_m": float("inf"),
                "state":      {},
            })

        # -------- Hard Safety Check (every frame) --------
        safety_override = hard_safety_check(annotations, frame_w)

        # -------- SLM Update (background thread, every N seconds) --------
        now = time.time()
        if (now - slm_last_update) >= SLM_UPDATE_INTERVAL:
            if slm_thread is None or not slm_thread.is_alive():
                slm_last_update = now
                # Deep copy annotations for thread safety
                ann_snapshot = [
                    {k: v for k, v in a.items() if k != "crop"}
                    for a in annotations
                ]
                slm_thread = threading.Thread(
                    target=slm_worker,
                    args=(ann_snapshot, frame_w, target_obj),
                    daemon=True
                )
                slm_thread.start()

        # -------- Pick Final Decision --------
        if safety_override:
            decision, reason, text_color = safety_override
            source_tag = "[SAFETY]"
        else:
            with slm_lock:
                decision   = slm_decision
                reason     = slm_reason
                text_color = slm_color
            source_tag = "[SLM]"

        # -------- Drawing --------
        origin = screen_origin(frame_w, frame_h, y_offset=8)
        cv2.circle(frame, origin, 6, (255, 255, 255), -1)
        cv2.putText(frame, "Vehicle",
                    (origin[0] - 40, origin[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Path arrow to target object
        path_target = None
        if annotations and target_obj:
            candidates = [a for a in annotations if target_obj in a["cls"]]
            if candidates:
                path_target = min(candidates, key=lambda a: a["distance_m"])

        if path_target:
            cx, cy   = path_target["center"]
            x1,y1,x2,y2 = path_target["xyxy"]
            bbox_w   = x2 - x1
            draw_translucent_path(frame, origin, (cx, cy), bbox_w,
                                   color=(0, 255, 255), alpha=0.25)
            draw_arrow_and_angle(frame, origin, (cx, cy))

        # Draw all detections
        for ann in annotations:
            x1, y1, x2, y2 = ann["xyxy"]
            state = ann["state"]
            dist  = ann["distance_m"]
            label = ann["cls"]

            state_parts = []
            for key in ("door_state","person_motion","occupancy",
                        "orientation","screen_state","vehicle_motion",
                        "fridge_state","attendance"):
                val = state.get(key)
                if val and val not in ("unknown", ""):
                    state_parts.append(val)

            if state_parts:
                label += f" ({', '.join(state_parts)})"

            color     = (0, 0, 255)  if (target_obj and target_obj in ann["cls"]) else (0, 255, 0)
            thickness = 3            if (target_obj and target_obj in ann["cls"]) else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if dist != float("inf"):
                cv2.putText(frame, f"{dist:.2f}m", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Navigation HUD
        cv2.putText(frame, f"{source_tag} {decision}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
        cv2.putText(frame, reason,
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # SLM update indicator (small dot blinks when SLM thread is running)
        indicator_color = (0, 255, 255) if (slm_thread and slm_thread.is_alive()) else (100, 100, 100)
        cv2.circle(frame, (frame_w - 20, 20), 7, indicator_color, -1)
        cv2.putText(frame, "SLM", (frame_w - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Context-Aware Indoor Navigation (Phi-2)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
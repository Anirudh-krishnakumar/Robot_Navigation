# rule_based_navigation.py
# Pure rule-based indoor navigation — NO SLM, NO API calls
# Drop-in fallback or standalone system
# Expanded object context states for all common indoor objects

import cv2
import torch
import numpy as np
import math
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not available.")

# --------------------------
# Config
# --------------------------
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS         = "yolov8s.pt"
FOCAL_LENGTH         = 640
STOP_THRESHOLD_M     = 1.0
SLOW_THRESHOLD_M     = 2.0
CONFIDENCE_THRESHOLD = 0.35

OBJECT_REAL_SIZES = {
    "door":         200,
    "chair":         90,
    "person":       170,
    "table":         75,
    "car":          150,
    "refrigerator": 170,
    "laptop":        30,
    "tv":           100,
    "monitor":      100,
    "bed":           60,
    "bottle":        25,
    "cup":           12,
    "backpack":      50,
    "suitcase":      70,
    "handbag":       35,
    "sofa":          90,
    "couch":         90,
    "bench":         50,
    "microwave":     35,
    "oven":          60,
    "sink":          50,
    "toilet":        75,
    "clock":         30,
    "vase":          30,
    "fire hydrant":  60,
    "stop sign":     75,
    "bicycle":      100,
    "motorcycle":   110,
    "truck":        250,
    "default":      100,
}

HAZARD_LABELS = {"person", "car", "truck", "motorcycle", "bicycle",
                 "wall", "fence", "fire hydrant", "stop sign"}

# --------------------------
# Models
# --------------------------
print(f"[INFO] Device: {DEVICE}")
processor  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base").to(DEVICE)
blip_model.eval()

tracker  = DeepSort(max_age=30)
detector = YOLO(YOLO_WEIGHTS) if YOLO_AVAILABLE else None

# --------------------------
# State tracking memory
# --------------------------
prev_centers = {}   # obj_id -> (cx, cy)

# --------------------------
# Utility
# --------------------------
def estimate_distance(bbox_h_px, real_h_cm):
    if bbox_h_px <= 0:
        return None
    return (real_h_cm * FOCAL_LENGTH) / float(bbox_h_px)

def get_position_description(cx, frame_w):
    t = frame_w // 3
    if cx < t:       return "Left"
    elif cx < 2 * t: return "Center"
    else:            return "Right"

def screen_origin(fw, fh, y_offset=10):
    return (fw // 2, fh - y_offset)

def draw_translucent_path(frame, origin, target_center, bbox_w_px,
                           color=(0, 255, 255), alpha=0.25):
    overlay = frame.copy()
    cx, cy  = target_center
    hw      = max(6, int(bbox_w_px // 2))
    pts = np.array([[origin[0], origin[1]],
                    [cx - hw, cy],
                    [cx + hw, cy]], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_arrow_and_angle(frame, origin, target_center,
                          color=(0, 255, 255), thickness=2):
    ox, oy = origin
    tx, ty = target_center
    cv2.arrowedLine(frame, (ox, oy), (tx, ty), color, thickness, tipLength=0.15)
    dx, dy = tx - ox, oy - ty
    angle_deg = math.degrees(math.atan2(dx, dy)) if (dx or dy) else 0.0
    abs_a = abs(angle_deg)
    if abs_a < 7:       steer = "Straight"
    elif angle_deg > 0: steer = f"Steer Right {abs_a:.0f}deg"
    else:               steer = f"Steer Left  {abs_a:.0f}deg"
    cv2.putText(frame, steer, (origin[0] - 160, origin[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return angle_deg

# --------------------------
# Per-object state helpers
# --------------------------
def infer_motion(obj_id, cx, cy):
    prev = prev_centers.get(obj_id)
    prev_centers[obj_id] = (cx, cy)
    if prev is None:
        return "unknown"
    return "moving" if abs(cy - prev[1]) >= 1 else "stationary"

def door_state(bbox_h, frame_h):
    r = bbox_h / frame_h
    if r < 0.35:  return "open"
    if r > 0.50:  return "closed"
    return "ajar"

def chair_occupancy(chair_ann, all_anns):
    cx1, cy1, cx2, cy2 = chair_ann["xyxy"]
    for a in all_anns:
        if a["cls"] != "person": continue
        px1, py1, px2, py2 = a["xyxy"]
        ix1, iy1 = max(cx1, px1), max(cy1, py1)
        ix2, iy2 = min(cx2, px2), min(cy2, py2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            area  = max(1, (cx2 - cx1) * (cy2 - cy1))
            if inter / area > 0.25:
                return "occupied"
    return "empty"

def table_clutter(table_ann, all_anns):
    tx1, ty1, tx2, ty2 = table_ann["xyxy"]
    for a in all_anns:
        if a["cls"] in ("table", "person"): continue
        cx, cy = a.get("center", (0, 0))
        if tx1 < cx < tx2 and ty1 < cy < ty2:
            return "cluttered"
    return "clear"

def screen_on_off(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return "on" if np.mean(gray) > 60 else "off"

def bottle_orientation(w, h):
    return "knocked_over" if w > h else "upright"

def bag_attendance(bag_ann, all_anns, prox_px=150):
    bx = (bag_ann["xyxy"][0] + bag_ann["xyxy"][2]) // 2
    by = (bag_ann["xyxy"][1] + bag_ann["xyxy"][3]) // 2
    for a in all_anns:
        if a["cls"] != "person": continue
        px, py = a.get("center", (0, 0))
        if math.hypot(bx - px, by - py) < prox_px:
            return "attended"
    return "unattended"

def fridge_state(bbox_w, frame_w):
    return "open" if (bbox_w / frame_w) > 0.35 else "closed"

def bed_occupancy(bed_ann, all_anns):
    """Reuse chair overlap logic for bed."""
    return chair_occupancy(bed_ann, all_anns)

def sink_usage(sink_ann, all_anns, prox_px=200):
    """Is a person standing near the sink?"""
    sx = (sink_ann["xyxy"][0] + sink_ann["xyxy"][2]) // 2
    sy = (sink_ann["xyxy"][1] + sink_ann["xyxy"][3]) // 2
    for a in all_anns:
        if a["cls"] != "person": continue
        px, py = a.get("center", (0, 0))
        if math.hypot(sx - px, sy - py) < prox_px:
            return "in_use"
    return "idle"

def microwave_state(crop_bgr):
    """Brightness heuristic — lit interior = running."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return "running" if np.mean(gray) > 80 else "idle"

def vehicle_motion(obj_id, cx, cy):
    return infer_motion(obj_id, cx, cy)   # reuse motion tracker

def crowd_density(all_anns, frame_w, frame_h):
    """Count persons and classify density."""
    count = sum(1 for a in all_anns if a["cls"] == "person")
    area  = frame_w * frame_h
    density = count / (area / 100000)   # persons per 100k px
    if count == 0:   return "empty"
    if density < 1:  return "sparse"
    if density < 3:  return "moderate"
    return "crowded"

# --------------------------
# Master state dispatcher
# --------------------------
def get_object_state(cls, ann, all_anns, crop, frame_h, frame_w):
    """
    Returns state dict for any detected object.
    Covers 25+ object classes with meaningful context.
    """
    state = {}
    x1, y1, x2, y2 = ann["xyxy"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    obj_id = f"{cls}_{id(ann)}"

    # ── Structural ──────────────────────────────────────────────
    if cls == "door":
        state["door_state"] = door_state(y2 - y1, frame_h)

    # ── Seating ─────────────────────────────────────────────────
    elif cls in ("chair", "sofa", "couch", "bench"):
        state["occupancy"] = chair_occupancy(ann, all_anns)

    # ── Surfaces ────────────────────────────────────────────────
    elif cls == "table":
        state["surface"] = table_clutter(ann, all_anns)

    # ── Sleep ───────────────────────────────────────────────────
    elif cls == "bed":
        state["occupancy"] = bed_occupancy(ann, all_anns)

    # ── People ──────────────────────────────────────────────────
    elif cls == "person":
        state["motion"] = infer_motion(obj_id, cx, cy)
        state["crowd"]  = crowd_density(all_anns, frame_w, frame_h)

    # ── Screens ─────────────────────────────────────────────────
    elif cls in ("laptop", "tv", "monitor"):
        state["screen"] = screen_on_off(crop)

    # ── Containers / Bottles ────────────────────────────────────
    elif cls in ("bottle", "cup", "bowl", "vase"):
        state["orientation"] = bottle_orientation(x2 - x1, y2 - y1)

    # ── Bags ────────────────────────────────────────────────────
    elif cls in ("backpack", "suitcase", "handbag"):
        state["attendance"] = bag_attendance(ann, all_anns)

    # ── Appliances ──────────────────────────────────────────────
    elif cls == "refrigerator":
        state["fridge"] = fridge_state(x2 - x1, frame_w)

    elif cls == "microwave":
        state["appliance"] = microwave_state(crop)

    elif cls in ("oven", "toaster"):
        state["appliance"] = microwave_state(crop)   # same brightness heuristic

    elif cls == "sink":
        state["usage"] = sink_usage(ann, all_anns)

    elif cls == "toilet":
        # Lid open = bbox aspect ratio is squarish; closed = wider
        ratio = (x2 - x1) / max(1, (y2 - y1))
        state["lid"] = "open" if ratio < 1.1 else "closed"

    # ── Vehicles ────────────────────────────────────────────────
    elif cls in ("car", "truck", "motorcycle", "bicycle"):
        state["motion"] = vehicle_motion(obj_id, cx, cy)

    # ── Clock ───────────────────────────────────────────────────
    elif cls == "clock":
        state["visibility"] = "visible"   # placeholder; extend with OCR if needed

    # ── Safety objects ──────────────────────────────────────────
    elif cls == "fire hydrant":
        state["clearance"] = "blocked" if any(
            math.hypot(cx - a.get("center", (0,0))[0],
                       cy - a.get("center", (0,0))[1]) < 80
            for a in all_anns if a["cls"] != "fire hydrant"
        ) else "clear"

    elif cls == "stop sign":
        state["relevance"] = "active"

    return state


# --------------------------
# Rule-based decision engine
# --------------------------
def decide_action(annotations, frame_w,
                  target_obj=None,
                  stop_threshold_m=STOP_THRESHOLD_M,
                  slow_threshold_m=SLOW_THRESHOLD_M):
    """
    Pure rule-based navigation decision.
    Priority order:
      1. Target found & within range → DETECTED / Go <pos>
      2. Immediate hazard            → STOP
      3. Near hazard                 → SLOW DOWN
      4. General obstacle avoidance  → Go Left / Right
      5. Clear path                  → Move Forward
    """
    if not annotations:
        return "Move Forward", "No objects detected", (0, 255, 0)

    # Normalise
    for ann in annotations:
        x1, y1, x2, y2 = ann["xyxy"]
        ann["center"]     = ((x1+x2)//2, (y1+y2)//2)
        ann["distance_m"] = float(ann.get("distance_m") or float("inf"))
        ann["cls_low"]    = ann.get("cls", "").lower()

    # ── 1. Target object ────────────────────────────────────────
    if target_obj:
        matches = [a for a in annotations if target_obj in a["cls_low"]]
        if matches:
            best = min(matches, key=lambda a: a["distance_m"])
            pos  = get_position_description(best["center"][0], frame_w)
            d    = best["distance_m"]
            st   = best.get("state", {})

            # Object-specific detected logic
            if target_obj == "door":
                ds = st.get("door_state", "unknown")
                if d <= stop_threshold_m:
                    if ds == "open":
                        return "DETECTED!", f"Door is OPEN at {d:.2f}m ({pos}) — proceed", (0, 200, 100)
                    else:
                        return "STOP!", f"Door CLOSED at {d:.2f}m ({pos}) — wait", (0, 0, 255)
                return f"Go {pos}", f"Door ({ds}) at {d:.2f}m", (0, 255, 0)

            elif target_obj == "person":
                motion = st.get("motion", "unknown")
                crowd  = st.get("crowd", "")
                if d <= stop_threshold_m:
                    return "STOP!", f"Person ({motion}) within {d:.2f}m ({pos})", (0, 0, 255)
                tag = f"{motion}, {crowd}" if crowd else motion
                return f"Go {pos}", f"Person ({tag}) at {d:.2f}m", (0, 255, 0)

            elif target_obj in ("chair", "sofa", "bench"):
                occ = st.get("occupancy", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"{target_obj} ({occ}) at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"{target_obj} ({occ}) at {d:.2f}m", (0, 255, 0)

            elif target_obj == "table":
                surf = st.get("surface", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"Table ({surf}) at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"Table ({surf}) at {d:.2f}m", (0, 255, 0)

            elif target_obj in ("laptop", "tv", "monitor"):
                scr = st.get("screen", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"{target_obj} screen {scr} at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"{target_obj} ({scr}) at {d:.2f}m", (0, 255, 0)

            elif target_obj == "refrigerator":
                fst = st.get("fridge", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"Fridge ({fst}) at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"Fridge ({fst}) at {d:.2f}m", (0, 255, 0)

            elif target_obj in ("backpack", "suitcase", "handbag"):
                att = st.get("attendance", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"{target_obj} ({att}) at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"{target_obj} ({att}) at {d:.2f}m", (0, 255, 0)

            elif target_obj == "bed":
                occ = st.get("occupancy", "unknown")
                if d <= stop_threshold_m:
                    return "DETECTED!", f"Bed ({occ}) at {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"Bed ({occ}) at {d:.2f}m", (0, 255, 0)

            else:
                # Generic target
                if d <= stop_threshold_m:
                    return "DETECTED!", f"{target_obj} within {d:.2f}m ({pos})", (0, 200, 100)
                return f"Go {pos}", f"{target_obj} at {d:.2f}m", (0, 255, 0)

        else:
            return "Searching", f"Looking for {target_obj}...", (0, 165, 255)

    # ── 2. Immediate hazard STOP ─────────────────────────────────
    hazards_near = [
        a for a in annotations
        if any(h in a["cls_low"] for h in HAZARD_LABELS)
        and a["distance_m"] <= stop_threshold_m
    ]
    if hazards_near:
        nearest = min(hazards_near, key=lambda a: a["distance_m"])
        pos     = get_position_description(nearest["center"][0], frame_w)
        st      = nearest.get("state", {})
        motion  = st.get("motion", "")
        detail  = f"({motion})" if motion and motion != "unknown" else ""
        return ("STOP!",
                f"Hazard: {nearest['cls']}{detail} at {nearest['distance_m']:.2f}m ({pos})",
                (0, 0, 255))

    # ── 3. Slow-down zone ────────────────────────────────────────
    hazards_slow = [
        a for a in annotations
        if any(h in a["cls_low"] for h in HAZARD_LABELS)
        and stop_threshold_m < a["distance_m"] <= slow_threshold_m
    ]
    if hazards_slow:
        nearest = min(hazards_slow, key=lambda a: a["distance_m"])
        pos     = get_position_description(nearest["center"][0], frame_w)
        return ("Slow Down",
                f"Caution: {nearest['cls']} at {nearest['distance_m']:.2f}m ({pos})",
                (0, 165, 255))

    # ── 4. General obstacle avoidance ───────────────────────────
    finite = [a for a in annotations if a["distance_m"] != float("inf")]
    if finite:
        nearest = min(finite, key=lambda a: a["distance_m"])
        pos     = get_position_description(nearest["center"][0], frame_w)
        d       = nearest["distance_m"]
        cls     = nearest["cls"]
        st      = nearest.get("state", {})

        # Context-aware avoidance hints
        extra = ""
        if "door_state" in st:   extra = f" door={st['door_state']}"
        if "occupancy"  in st:   extra = f" {st['occupancy']}"
        if "motion"     in st:   extra = f" {st['motion']}"
        if "screen"     in st:   extra = f" screen={st['screen']}"

        if d <= slow_threshold_m:
            return ("Slow Down",
                    f"{cls}{extra} at {d:.2f}m ahead — caution ({pos})",
                    (0, 165, 255))
        if pos == "Left":
            return "Go Right", f"{cls}{extra} at {d:.2f}m on Left", (0, 255, 0)
        if pos == "Right":
            return "Go Left",  f"{cls}{extra} at {d:.2f}m on Right", (0, 255, 0)
        return "Move Forward", f"Path clear; nearest {cls} at {d:.2f}m", (0, 255, 0)

    # ── 5. All clear ─────────────────────────────────────────────
    return "Move Forward", "Path looks clear", (0, 255, 0)


# --------------------------
# Main loop
# --------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    user_cmd   = input("Enter navigation command (e.g., 'find door', blank for all): ").strip().lower()
    target_obj = None
    if user_cmd:
        for obj in OBJECT_REAL_SIZES.keys():
            if obj in user_cmd:
                target_obj = obj
                break

    print(f"[INFO] Target : {target_obj or 'None (general navigation)'}")
    print("[INFO] Mode   : Pure rule-based (no SLM / no API)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        frame_h, frame_w = frame.shape[:2]
        raw_anns         = []

        # ── YOLO detection ──────────────────────────────────────
        if YOLO_AVAILABLE:
            try:
                results = detector(frame, device=0, imgsz=640)[0]
            except Exception:
                results = detector(frame)[0]

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
                    "_crop":      crop,
                })

            # Second pass — compute relational states
            for ann in raw_anns:
                crop = ann.pop("_crop", None)
                ann["state"] = get_object_state(
                    ann["cls"], ann, raw_anns, crop, frame_h, frame_w
                )

            annotations = raw_anns

        else:
            annotations = [{
                "cls":        "unknown",
                "xyxy":       (0, 0, frame_w, frame_h),
                "center":     (frame_w//2, frame_h//2),
                "distance_m": float("inf"),
                "state":      {},
            }]

        # ── Rule-based decision ─────────────────────────────────
        decision, reason, text_color = decide_action(
            annotations, frame_w,
            target_obj=target_obj,
        )

        # ── Drawing ─────────────────────────────────────────────
        origin = screen_origin(frame_w, frame_h, y_offset=8)
        cv2.circle(frame, origin, 6, (255, 255, 255), -1)
        cv2.putText(frame, "Vehicle",
                    (origin[0] - 40, origin[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Path arrow to target
        path_target = None
        if target_obj:
            candidates = [a for a in annotations if target_obj in a["cls"]]
            if candidates:
                path_target = min(candidates, key=lambda a: a["distance_m"])

        if path_target:
            cx, cy   = path_target["center"]
            x1,y1,x2,y2 = path_target["xyxy"]
            draw_translucent_path(frame, origin, (cx, cy), x2 - x1,
                                   color=(0, 255, 255), alpha=0.25)
            draw_arrow_and_angle(frame, origin, (cx, cy))

        # Draw all detections
        for ann in annotations:
            x1, y1, x2, y2 = ann["xyxy"]
            st    = ann["state"]
            dist  = ann["distance_m"]
            label = ann["cls"]

            # Build state string from all available keys
            tags = []
            for k, v in st.items():
                if v and v not in ("unknown", ""):
                    tags.append(str(v))
            if tags:
                label += f" ({', '.join(tags)})"

            color     = (0, 0, 255) if (target_obj and target_obj in ann["cls"]) else (0, 255, 0)
            thickness = 3           if (target_obj and target_obj in ann["cls"]) else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if dist != float("inf"):
                cv2.putText(frame, f"{dist:.2f}m", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # HUD
        cv2.putText(frame, f"[RULE] {decision}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
        cv2.putText(frame, reason,
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Mode tag (top-right)
        cv2.putText(frame, "RULE-BASED",
                    (frame_w - 145, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Rule-Based Indoor Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
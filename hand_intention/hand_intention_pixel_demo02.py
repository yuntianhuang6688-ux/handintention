import cv2
import numpy as np
import time
import socket
import json
import os
from collections import deque

import mediapipe as mp

# ============================================================
# Camera settings
# ============================================================

CAMERA_INDEX = 0
USE_DSHOW = True

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# ============================================================
# Calibration files
# ============================================================

TOPVIEW_CALIB_FILE = "topview_calibration.json"
BLOCK_CENTER_FILE = "block_centers_topview.json"

# ============================================================
# UDP settings
# ============================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# ============================================================
# Block settings
# ============================================================

BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]

DRAW_COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 140, 255),
}

TEXT_COLORS = DRAW_COLORS.copy()

# ============================================================
# Intention prediction parameters
# ============================================================
# 现在全部是桌面真实距离，不是像素距离。

EARLY_MIN_DISTANCE_M = 0.030
EARLY_MAX_DISTANCE_M = 0.240
MIN_HAND_SPEED_M = 0.008

CONFIDENCE_THRESHOLD = 0.42
STABLE_TIME_REQUIRED = 0.28

SOFTMAX_TEMPERATURE = 0.85

# EARLY_MIN_DISTANCE_M = 0.045  # 4.5 cm，太近就不算提前预测
# EARLY_MAX_DISTANCE_M = 0.260  # 26 cm，太远不确认
# MIN_HAND_SPEED_M = 0.012  # 最近窗口内移动至少 1.2 cm
#
# CONFIDENCE_THRESHOLD = 0.68
# STABLE_TIME_REQUIRED = 0.32
#
# SOFTMAX_TEMPERATURE = 1.35

DISTANCE_WEIGHT = 0.35
DIRECTION_WEIGHT = 0.45
CLOSING_WEIGHT = 0.20

HAND_HISTORY_LEN = 6

PINCH_THRESHOLD_PX = 55.0

SEND_COOLDOWN = 1.5

# ============================================================
# Global runtime state
# ============================================================

homography = None
homography_inv = None
real_width_m = None
real_height_m = None
calib_image_points = None

block_points_px = {}
block_points_table = {}

remaining_blocks = set(BLOCK_COLORS)

current_click_index = 0

hand_history_table = deque(maxlen=HAND_HISTORY_LEN)
previous_distances = {}

stable_candidate = None
stable_start_time = None
locked_target = None
last_sent_time = 0.0

last_hand_seen_time = 0.0

status_text = "Load calibration and click block centers."
stage_text = "Setup"

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ============================================================
# MediaPipe
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ============================================================
# Calibration helpers
# ============================================================

def load_topview_calibration():
    global homography
    global homography_inv
    global real_width_m
    global real_height_m
    global calib_image_points

    if not os.path.exists(TOPVIEW_CALIB_FILE):
        print(f"[ERROR] Cannot find {TOPVIEW_CALIB_FILE}")
        print("Please run topview_table_calibration.py first.")
        return False

    with open(TOPVIEW_CALIB_FILE, "r") as f:
        data = json.load(f)

    homography = np.array(data["homography"], dtype=np.float32)
    homography_inv = np.linalg.inv(homography)

    real_width_m = float(data["real_width_m"])
    real_height_m = float(data["real_height_m"])
    calib_image_points = np.array(data["image_points"], dtype=np.float32)

    print("[OK] Loaded top-view calibration.")
    print("Real area size:", real_width_m, "m x", real_height_m, "m")
    print("Image points:")
    print(calib_image_points)

    return True


def pixel_to_table(pixel_point):
    if homography is None:
        return None

    pt = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
    table_pt = cv2.perspectiveTransform(pt, homography)[0, 0]

    return table_pt.astype(np.float32)


def table_to_pixel(table_point):
    if homography_inv is None:
        return None

    pt = np.array([[[float(table_point[0]), float(table_point[1])]]], dtype=np.float32)
    pixel_pt = cv2.perspectiveTransform(pt, homography_inv)[0, 0]

    return pixel_pt.astype(np.float32)


def point_inside_table_area(table_point):
    if table_point is None:
        return False

    x, y = float(table_point[0]), float(table_point[1])

    return (
            0.0 <= x <= real_width_m
            and 0.0 <= y <= real_height_m
    )


# ============================================================
# Block center save/load
# ============================================================

def save_block_centers():
    data = {
        "block_points_px": {
            color: block_points_px[color].tolist()
            for color in block_points_px
        },
        "block_points_table": {
            color: block_points_table[color].tolist()
            for color in block_points_table
        },
    }

    with open(BLOCK_CENTER_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVE] Block centers saved to {BLOCK_CENTER_FILE}")


def load_block_centers():
    global block_points_px
    global block_points_table
    global current_click_index

    if not os.path.exists(BLOCK_CENTER_FILE):
        print("[INFO] No block center file found.")
        print("[INFO] Please click block centers in this order:")
        print("       red, blue, green, yellow, orange")
        return False

    with open(BLOCK_CENTER_FILE, "r") as f:
        data = json.load(f)

    block_points_px = {
        color: np.array(value, dtype=np.float32)
        for color, value in data.get("block_points_px", {}).items()
    }

    block_points_table = {
        color: np.array(value, dtype=np.float32)
        for color, value in data.get("block_points_table", {}).items()
    }

    current_click_index = len(block_points_px)

    print("[LOAD] Loaded block centers:")
    for color in BLOCK_COLORS:
        if color in block_points_table:
            print(
                f"  {color}: pixel={block_points_px[color]}, "
                f"table={block_points_table[color]}"
            )

    return len(block_points_px) == len(BLOCK_COLORS)


def clear_block_centers():
    global block_points_px
    global block_points_table
    global current_click_index

    block_points_px = {}
    block_points_table = {}
    current_click_index = 0

    if os.path.exists(BLOCK_CENTER_FILE):
        os.remove(BLOCK_CENTER_FILE)

    print("[CLEAR] Block centers cleared.")
    print("[INFO] Click block centers again: red, blue, green, yellow, orange")


# ============================================================
# Mouse callback
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global current_click_index
    global status_text

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if current_click_index >= len(BLOCK_COLORS):
        table_xy = pixel_to_table((x, y))
        if table_xy is not None:
            print(
                f"[POINT] pixel=({x},{y}) "
                f"table=({table_xy[0]:.3f}, {table_xy[1]:.3f}) m"
            )
        return

    color = BLOCK_COLORS[current_click_index]

    px = np.array([x, y], dtype=np.float32)
    table_xy = pixel_to_table(px)

    if table_xy is None:
        print("[WARN] No calibration loaded.")
        return

    if not point_inside_table_area(table_xy):
        print("[WARN] Clicked point is outside calibrated table area.")
        print("       Still saving, but please check your calibration.")

    block_points_px[color] = px
    block_points_table[color] = table_xy

    print(
        f"[SET] {color}: pixel=({x},{y}), "
        f"table=({table_xy[0]:.3f}, {table_xy[1]:.3f}) m"
    )

    current_click_index += 1

    if current_click_index >= len(BLOCK_COLORS):
        save_block_centers()
        status_text = "All block centers set. Show your hand."
    else:
        next_color = BLOCK_COLORS[current_click_index]
        status_text = f"Click {next_color} block center."


# ============================================================
# Utility functions
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))

    if n < 1e-8:
        return np.zeros_like(v)

    return v / n


def softmax_from_scores(scores):
    if len(scores) == 0:
        return {}

    colors = list(scores.keys())
    values = np.array([scores[c] for c in colors], dtype=np.float32)

    values = values / SOFTMAX_TEMPERATURE
    values = values - np.max(values)

    exp_values = np.exp(values)
    probs = exp_values / np.sum(exp_values)

    return {
        color: float(prob)
        for color, prob in zip(colors, probs)
    }


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def compute_hand_velocity_table():
    if len(hand_history_table) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    start = np.array(hand_history_table[0], dtype=np.float32)
    end = np.array(hand_history_table[-1], dtype=np.float32)

    return end - start


def baseline_nearest_prediction(hand_table):
    candidates = [
        color for color in BLOCK_COLORS
        if color in remaining_blocks and color in block_points_table
    ]

    if len(candidates) == 0:
        return None, None

    distances = {
        color: distance(hand_table, block_points_table[color])
        for color in candidates
    }

    best_color = min(distances, key=distances.get)

    return best_color, distances[best_color]


# ============================================================
# Hand extraction
# ============================================================

def extract_hand_points(frame, hand_landmarks):
    h, w = frame.shape[:2]

    lm = hand_landmarks.landmark

    index_tip_lm = lm[8]
    thumb_tip_lm = lm[4]
    wrist_lm = lm[0]

    index_tip_px = np.array(
        [index_tip_lm.x * w, index_tip_lm.y * h],
        dtype=np.float32
    )

    thumb_tip_px = np.array(
        [thumb_tip_lm.x * w, thumb_tip_lm.y * h],
        dtype=np.float32
    )

    wrist_px = np.array(
        [wrist_lm.x * w, wrist_lm.y * h],
        dtype=np.float32
    )

    # 重点：用食指尖作为 intention point
    hand_point_px = index_tip_px.copy()

    pinch_distance_px = distance(index_tip_px, thumb_tip_px)

    hand_point_table = pixel_to_table(hand_point_px)

    return {
        "hand_point_px": hand_point_px,
        "hand_point_table": hand_point_table,
        "index_tip_px": index_tip_px,
        "thumb_tip_px": thumb_tip_px,
        "wrist_px": wrist_px,
        "pinch_distance_px": pinch_distance_px,
    }


# ============================================================
# Prediction
# ============================================================

def predict_intention(hand_table, hand_velocity_table):
    global previous_distances

    candidates = [
        color for color in BLOCK_COLORS
        if color in remaining_blocks and color in block_points_table
    ]

    if len(candidates) == 0:
        return None, 0.0, None, {}, {}, {}

    hand_speed = float(np.linalg.norm(hand_velocity_table))
    hand_vel_dir = normalize(hand_velocity_table)

    raw_scores = {}
    details = {}
    current_distances = {}

    for color in candidates:
        block_table = block_points_table[color]

        vec_to_block = block_table - hand_table
        dist_m = float(np.linalg.norm(vec_to_block))
        dir_to_block = normalize(vec_to_block)

        # 1. distance score
        # 越近越高，但会被 early factor 限制。
        dist_norm = (dist_m - EARLY_MIN_DISTANCE_M) / (
                EARLY_MAX_DISTANCE_M - EARLY_MIN_DISTANCE_M
        )
        distance_score = 1.0 - np.clip(dist_norm, 0.0, 1.0)

        # 2. direction alignment score
        # 手的运动方向是否朝向该方块。
        direction_score = float(np.dot(hand_vel_dir, dir_to_block))
        direction_score = max(0.0, direction_score)

        # 3. closing score
        # 手到方块距离是否正在变小。
        if color in previous_distances:
            distance_change = previous_distances[color] - dist_m
        else:
            distance_change = 0.0

        closing_score = np.clip(distance_change / 0.035, 0.0, 1.0)

        # Early prediction factor:
        # 太近说明已经接近抓取阶段，不应作为提前预测强触发。
        # 太远说明不确定。
        if dist_m < EARLY_MIN_DISTANCE_M:
            early_factor = 0.35
        elif dist_m > EARLY_MAX_DISTANCE_M:
            early_factor = 0.25
        else:
            early_factor = 1.0

        score = early_factor * (
                DISTANCE_WEIGHT * distance_score
                + DIRECTION_WEIGHT * direction_score
                + CLOSING_WEIGHT * closing_score
        )

        raw_scores[color] = float(score)
        current_distances[color] = dist_m

        details[color] = {
            "dist_m": dist_m,
            "distance_score": float(distance_score),
            "direction_score": float(direction_score),
            "closing_score": float(closing_score),
            "distance_change": float(distance_change),
            "score": float(score),
        }

    probabilities = softmax_from_scores(raw_scores)

    if len(probabilities) == 0:
        return None, 0.0, None, {}, details, current_distances

    best_color = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_color]
    best_distance_m = current_distances[best_color]

    previous_distances = current_distances.copy()

    return best_color, best_prob, best_distance_m, probabilities, details, current_distances


def update_stable_lock(best_color, best_prob, best_distance_m, hand_speed_m, details):
    global stable_candidate
    global stable_start_time
    global locked_target
    global stage_text

    now = time.time()

    if locked_target is not None:
        return None, 0.0, f"Already locked: {locked_target}"

    if best_color is None:
        stable_candidate = None
        stable_start_time = None
        stage_text = "No target"
        return None, 0.0, "No target"

    best_detail = details.get(best_color, {})
    direction_score = best_detail.get("direction_score", 0.0)
    closing_score = best_detail.get("closing_score", 0.0)

    if best_distance_m < EARLY_MIN_DISTANCE_M:
        stable_candidate = None
        stable_start_time = None
        stage_text = "Late stage"
        return None, 0.0, (
            f"Too close for early prediction: {best_distance_m * 100:.1f} cm"
        )

    if best_distance_m > EARLY_MAX_DISTANCE_M:
        stable_candidate = None
        stable_start_time = None
        stage_text = "Too far"
        return None, 0.0, (
            f"Waiting: hand too far {best_distance_m * 100:.1f} cm"
        )

    if hand_speed_m < MIN_HAND_SPEED_M:
        stable_candidate = None
        stable_start_time = None
        stage_text = "Slow hand"
        return None, 0.0, (
            f"Hand speed too low: {hand_speed_m * 100:.1f} cm/window"
        )

    approaching = (direction_score > 0.35) or (closing_score > 0.12)

    if not approaching:
        stable_candidate = None
        stable_start_time = None
        stage_text = "Not approaching"
        return None, 0.0, "Hand is not clearly approaching target"

    if best_prob < CONFIDENCE_THRESHOLD:
        stable_candidate = None
        stable_start_time = None
        stage_text = "Low confidence"
        return None, 0.0, (
            f"Confidence too low: {best_prob:.2f}"
        )

    if stable_candidate != best_color:
        stable_candidate = best_color
        stable_start_time = now
        stage_text = "Tracking"
        return None, 0.0, f"Tracking candidate: {best_color}"

    stable_duration = now - stable_start_time

    if stable_duration >= STABLE_TIME_REQUIRED:
        locked_target = best_color
        stage_text = "Confirmed"
        return best_color, stable_duration, f"Early confirmed: {best_color}"

    stage_text = "Stabilising"
    return None, stable_duration, (
        f"Stabilising {best_color}: {stable_duration:.2f}s"
    )


# ============================================================
# UDP
# ============================================================

def send_intention_udp(color):
    global last_sent_time
    global remaining_blocks

    now = time.time()

    if now - last_sent_time < SEND_COOLDOWN:
        return False

    msg = color.encode("utf-8")
    udp_sock.sendto(msg, (UDP_IP, UDP_PORT))

    last_sent_time = now

    if color in remaining_blocks:
        remaining_blocks.remove(color)

    print(f"[UDP] Sent intention to MuJoCo: {color}")
    print("[STATE] Remaining blocks:", sorted(remaining_blocks))

    return True


def reset_runtime_state():
    global remaining_blocks
    global stable_candidate
    global stable_start_time
    global locked_target
    global previous_distances
    global hand_history_table
    global status_text
    global stage_text

    remaining_blocks = set(BLOCK_COLORS)
    stable_candidate = None
    stable_start_time = None
    locked_target = None
    previous_distances = {}
    hand_history_table.clear()

    status_text = "Runtime reset. Show your hand."
    stage_text = "Reset"

    print("[RESET] Runtime state reset.")
    print("[STATE] Remaining blocks:", sorted(remaining_blocks))


def unlock_target():
    global locked_target
    global stable_candidate
    global stable_start_time
    global status_text
    global stage_text

    print(f"[UNLOCK] Previous locked target: {locked_target}")

    locked_target = None
    stable_candidate = None
    stable_start_time = None
    status_text = "Unlocked. Ready for next prediction."
    stage_text = "Unlocked"


# ============================================================
# Drawing
# ============================================================

def draw_calibration_area(frame):
    if calib_image_points is None:
        return

    pts = calib_image_points.astype(int)

    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(frame, p1, p2, (0, 255, 255), 2)


def draw_block_points(frame):
    for color in BLOCK_COLORS:
        if color not in block_points_px:
            continue

        px = block_points_px[color].astype(int)
        draw_color = DRAW_COLORS[color]

        cv2.circle(frame, tuple(px), 10, draw_color, -1)

        label = color
        if color not in remaining_blocks:
            label += " done"

        cv2.putText(
            frame,
            label,
            tuple(px + np.array([12, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            draw_color,
            2,
        )


def draw_hand_info(frame, hand_info, best_color=None):
    if hand_info is None:
        return

    hand_px = hand_info["hand_point_px"].astype(int)
    thumb_px = hand_info["thumb_tip_px"].astype(int)

    cv2.circle(frame, tuple(hand_px), 8, (255, 255, 255), -1)
    cv2.circle(frame, tuple(thumb_px), 6, (255, 0, 255), -1)
    cv2.line(frame, tuple(hand_px), tuple(thumb_px), (255, 255, 255), 2)

    if best_color is not None and best_color in block_points_px:
        block_px = block_points_px[best_color].astype(int)
        cv2.line(frame, tuple(hand_px), tuple(block_px), DRAW_COLORS[best_color], 2)


def draw_probability_panel(
        frame,
        probabilities,
        best_color,
        best_prob,
        best_distance_m,
        baseline_color,
        baseline_distance_m,
        hand_speed_m,
        pinch_distance_px,
        stable_duration,
):
    h, w = frame.shape[:2]

    panel_h = 190
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (0, 0, 0), -1)

    x = 15
    y = h - panel_h + 28

    cv2.putText(
        frame,
        "Probabilities:",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    y += 28

    for color in BLOCK_COLORS:
        prob = probabilities.get(color, 0.0)
        mark = " <" if color == best_color else ""

        cv2.putText(
            frame,
            f"{color}: {prob:.2f}{mark}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            DRAW_COLORS[color],
            2,
        )
        y += 23

    x2 = 260
    y2 = h - panel_h + 32

    if best_color is not None:
        dist_text = ""
        if best_distance_m is not None:
            dist_text = f"  dist={best_distance_m * 100:.1f}cm"

        cv2.putText(
            frame,
            f"Stable model: {best_color} {best_prob:.2f}{dist_text}",
            (x2, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            DRAW_COLORS[best_color],
            2,
        )
    else:
        cv2.putText(
            frame,
            "Stable model: None",
            (x2, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )

    y2 += 30

    if baseline_color is not None:
        cv2.putText(
            frame,
            f"Baseline nearest: {baseline_color} "
            f"{baseline_distance_m * 100:.1f}cm",
            (x2, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            DRAW_COLORS[baseline_color],
            2,
        )

    y2 += 28

    cv2.putText(
        frame,
        f"Stage: {stage_text}",
        (x2, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    y2 += 28

    cv2.putText(
        frame,
        f"Status: {status_text}",
        (x2, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 0),
        2,
    )

    y2 += 28

    cv2.putText(
        frame,
        f"Speed: {hand_speed_m * 100:.1f}cm/window | "
        f"Pinch: {pinch_distance_px:.1f}px | "
        f"Stable: {stable_duration:.2f}s",
        (x2, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    y2 += 28

    cv2.putText(
        frame,
        f"Remaining: {sorted(remaining_blocks)}",
        (x2, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        2,
    )


def draw_setup_text(frame):
    y = 28

    if homography is None:
        text = "No topview_calibration.json. Run calibration first."
        color = (0, 0, 255)
    elif current_click_index < len(BLOCK_COLORS):
        color_name = BLOCK_COLORS[current_click_index]
        text = f"Click {color_name} block center."
        color = DRAW_COLORS[color_name]
    else:
        text = "All block centers set. Show your hand."
        color = (255, 255, 255)

    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        color,
        2,
    )

    y += 28

    cv2.putText(
        frame,
        "Keys: c=clear block centers, r=reset runtime, u=unlock, q=quit",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


# ============================================================
# Main loop
# ============================================================

def main():
    global status_text
    global last_hand_seen_time
    global locked_target
    global stable_candidate
    global stable_start_time

    if not load_topview_calibration():
        return

    load_block_centers()

    if USE_DSHOW:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    time.sleep(1.0)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        return

    cv2.namedWindow("Hand Intention Pixel Demo")
    cv2.setMouseCallback("Hand Intention Pixel Demo", mouse_callback)

    print("")
    print("========================================")
    print("Hand Intention Pixel Demo with Top-view Calibration")
    print("Distance unit: meters on table plane")
    print("UDP target:", UDP_IP, UDP_PORT)
    print("========================================")
    print("Click block centers in order if not loaded:")
    print("red, blue, green, yellow, orange")
    print("")
    print("Keys:")
    print("  c = clear block centers")
    print("  r = reset runtime state")
    print("  u = unlock current target")
    print("  q = quit")
    print("")

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            draw_calibration_area(frame)
            draw_block_points(frame)
            draw_setup_text(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_info = None
            best_color = None
            best_prob = 0.0
            best_distance_m = None
            probabilities = {}
            details = {}
            stable_duration = 0.0
            baseline_color = None
            baseline_distance_m = None
            hand_speed_m = 0.0
            pinch_distance_px = 999.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                hand_info = extract_hand_points(frame, hand_landmarks)
                hand_table = hand_info["hand_point_table"]

                pinch_distance_px = hand_info["pinch_distance_px"]

                if hand_table is not None and point_inside_table_area(hand_table):
                    last_hand_seen_time = time.time()

                    hand_history_table.append(hand_table.copy())
                    hand_velocity_table = compute_hand_velocity_table()
                    hand_speed_m = float(np.linalg.norm(hand_velocity_table))

                    baseline_color, baseline_distance_m = baseline_nearest_prediction(
                        hand_table
                    )

                    if len(block_points_table) == len(BLOCK_COLORS):
                        (
                            best_color,
                            best_prob,
                            best_distance_m,
                            probabilities,
                            details,
                            current_distances,
                        ) = predict_intention(
                            hand_table,
                            hand_velocity_table,
                        )

                        confirmed_color, stable_duration, new_status = update_stable_lock(
                            best_color=best_color,
                            best_prob=best_prob,
                            best_distance_m=best_distance_m,
                            hand_speed_m=hand_speed_m,
                            details=details,
                        )

                        status_text = new_status

                        if confirmed_color is not None:
                            sent = send_intention_udp(confirmed_color)

                            if sent:
                                status_text = (
                                    f"Confirmed and sent to MuJoCo: {confirmed_color}"
                                )

                    else:
                        status_text = "Please click all block centers first."

                else:
                    status_text = "Hand outside calibrated table area."

            else:
                # 如果已经锁定并且手离开一段时间，自动解锁，方便下一轮。
                if locked_target is not None:
                    if time.time() - last_hand_seen_time > 1.2:
                        print("[AUTO] Hand disappeared. Unlocking for next cycle.")
                        locked_target = None
                        stable_candidate = None
                        stable_start_time = None
                        status_text = "Ready for next intention."

                hand_history_table.clear()
                previous_distances.clear()

            draw_hand_info(frame, hand_info, best_color)

            draw_probability_panel(
                frame=frame,
                probabilities=probabilities,
                best_color=best_color,
                best_prob=best_prob,
                best_distance_m=best_distance_m,
                baseline_color=baseline_color,
                baseline_distance_m=baseline_distance_m,
                hand_speed_m=hand_speed_m,
                pinch_distance_px=pinch_distance_px,
                stable_duration=stable_duration,
            )

            cv2.imshow("Hand Intention Pixel Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                clear_block_centers()

            if key == ord("r"):
                reset_runtime_state()

            if key == ord("u"):
                unlock_target()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
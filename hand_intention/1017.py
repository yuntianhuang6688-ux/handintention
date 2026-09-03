import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import time


# ============================================================
# UDP settings
# ============================================================
WSL_IP = "172.26.149.220"       # 改成 WSL 里 hostname -I 查到的 IP
WSL_PORT = 5005

WINDOWS_RECEIVE_IP = "0.0.0.0"
WINDOWS_RECEIVE_PORT = 5006

udp_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ============================================================
# Intention detection settings
# ============================================================
BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]
remaining_blocks = ["red", "blue", "green", "yellow", "orange"]

# 重要参数：防止太灵敏
CONFIDENCE_THRESHOLD = 0.75
STABLE_TIME_REQUIRED = 1.0
TARGET_DISTANCE_THRESHOLD = 90.0
PINCH_DISTANCE_THRESHOLD = 55.0
SEND_COOLDOWN = 2.0

# 轨迹参数
HAND_HISTORY_LENGTH = 8

# 预测权重
DISTANCE_WEIGHT = 4.0
DIRECTION_WEIGHT = 2.0


# ============================================================
# Global state
# ============================================================
block_points = {}
hand_history = []

stable_candidate = None
stable_start_time = None

locked_target = None
last_sent_target = None
last_send_time = 0.0

system_message = "Click block centers in order: red, blue, green, yellow, orange"


# ============================================================
# MediaPipe
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# Utility functions
# ============================================================
def softmax(scores):
    scores = np.array(scores, dtype=np.float32)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm


def send_intention_to_wsl(color):
    global last_sent_target, last_send_time, system_message

    now = time.time()

    if color == last_sent_target and now - last_send_time < SEND_COOLDOWN:
        return

    try:
        udp_send_sock.sendto(
            color.encode("utf-8"),
            (WSL_IP, WSL_PORT)
        )
        last_sent_target = color
        last_send_time = now
        system_message = f"Sent intention to WSL: {color}"
        print(system_message)

    except Exception as e:
        system_message = f"Failed to send UDP: {e}"
        print(system_message)


def udp_receive_loop():
    """
    接收 WSL 发回来的 remaining:...
    例如：
    remaining:blue,green,yellow
    """
    global remaining_blocks
    global locked_target, last_sent_target
    global stable_candidate, stable_start_time
    global system_message

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((WINDOWS_RECEIVE_IP, WINDOWS_RECEIVE_PORT))

    print(f"Windows UDP receiver listening on port {WINDOWS_RECEIVE_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode("utf-8").strip()

            print(f"Received from WSL: {message}")

            if message.startswith("remaining:"):
                payload = message.replace("remaining:", "").strip()

                if payload == "":
                    remaining_blocks = []
                else:
                    new_remaining = [
                        item.strip()
                        for item in payload.split(",")
                        if item.strip() in BLOCK_COLORS
                    ]
                    remaining_blocks = new_remaining

                locked_target = None
                last_sent_target = None
                stable_candidate = None
                stable_start_time = None

                system_message = f"Remaining blocks updated: {remaining_blocks}"

        except Exception as e:
            print(f"UDP receive error: {e}")


def get_hand_points(hand_landmarks, image_width, image_height):
    """
    返回：
    hand_center: wrist 和 index_finger_tip 的中点
    index_tip: 食指尖
    thumb_tip: 拇指尖
    """
    wrist = hand_landmarks.landmark[0]
    index_tip_lm = hand_landmarks.landmark[8]
    thumb_tip_lm = hand_landmarks.landmark[4]

    wrist_px = np.array([
        wrist.x * image_width,
        wrist.y * image_height
    ], dtype=np.float32)

    index_tip = np.array([
        index_tip_lm.x * image_width,
        index_tip_lm.y * image_height
    ], dtype=np.float32)

    thumb_tip = np.array([
        thumb_tip_lm.x * image_width,
        thumb_tip_lm.y * image_height
    ], dtype=np.float32)

    hand_center = (wrist_px + index_tip) / 2.0

    return hand_center, index_tip, thumb_tip


def compute_hand_velocity():
    if len(hand_history) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    start = np.array(hand_history[0], dtype=np.float32)
    end = np.array(hand_history[-1], dtype=np.float32)

    return end - start


def predict_intention(hand_pos, hand_vel):
    """
    只对 remaining_blocks 中的颜色进行预测。
    返回：
    best_color, best_prob, probabilities_dict, distance_to_best
    """
    available_colors = [
        color for color in remaining_blocks
        if color in block_points
    ]

    if len(available_colors) == 0:
        return None, 0.0, {}, None

    scores = []
    hand_vel_dir = normalize(hand_vel)

    for color in available_colors:
        block_pos = np.array(block_points[color], dtype=np.float32)
        vec_to_block = block_pos - hand_pos
        dist = np.linalg.norm(vec_to_block)

        # 距离分数：越近越高
        distance_score = -dist / 150.0

        # 方向分数：手的运动方向是否朝向该方块
        if np.linalg.norm(hand_vel) < 3.0:
            direction_score = 0.0
        else:
            dir_to_block = normalize(vec_to_block)
            direction_score = float(np.dot(hand_vel_dir, dir_to_block))

        score = DISTANCE_WEIGHT * distance_score + DIRECTION_WEIGHT * direction_score
        scores.append(score)

    probs = softmax(scores)

    probabilities = {
        color: float(prob)
        for color, prob in zip(available_colors, probs)
    }

    best_color = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_color]
    distance_to_best = distance(hand_pos, block_points[best_color])

    return best_color, best_prob, probabilities, distance_to_best


def update_stable_lock(best_color, best_prob, distance_to_best, pinch_distance):
    """
    稳定判定逻辑：
    - 置信度足够
    - 距离目标足够近
    - 有 pinch 抓取趋势
    - 同一颜色持续稳定 STABLE_TIME_REQUIRED 秒
    """
    global stable_candidate, stable_start_time
    global locked_target

    now = time.time()

    if best_color is None:
        stable_candidate = None
        stable_start_time = None
        return None, 0.0, "No candidate"

    confidence_ok = best_prob >= CONFIDENCE_THRESHOLD
    distance_ok = distance_to_best is not None and distance_to_best <= TARGET_DISTANCE_THRESHOLD
    pinch_ok = pinch_distance <= PINCH_DISTANCE_THRESHOLD

    if not confidence_ok:
        stable_candidate = None
        stable_start_time = None
        return None, 0.0, f"Waiting: confidence {best_prob:.2f} < {CONFIDENCE_THRESHOLD:.2f}"

    if not distance_ok:
        stable_candidate = None
        stable_start_time = None
        return None, 0.0, f"Waiting: hand too far {distance_to_best:.1f}px"

    if not pinch_ok:
        stable_candidate = None
        stable_start_time = None
        return None, 0.0, f"Waiting: pinch {pinch_distance:.1f}px"

    if stable_candidate != best_color:
        stable_candidate = best_color
        stable_start_time = now
        return None, 0.0, f"Candidate: {best_color}, stabilizing..."

    stable_duration = now - stable_start_time

    if stable_duration >= STABLE_TIME_REQUIRED:
        locked_target = best_color
        return best_color, stable_duration, f"LOCKED: {best_color}"

    return None, stable_duration, f"Stabilizing {best_color}: {stable_duration:.1f}s"


# ============================================================
# Mouse callback
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global block_points, system_message

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(block_points) >= len(BLOCK_COLORS):
        return

    color = BLOCK_COLORS[len(block_points)]
    block_points[color] = np.array([x, y], dtype=np.float32)

    system_message = f"Set {color} block center at ({x}, {y})"
    print(system_message)

    if len(block_points) < len(BLOCK_COLORS):
        next_color = BLOCK_COLORS[len(block_points)]
        system_message = f"Click {next_color} block center"
    else:
        system_message = "All block centers set. Show your hand."


# ============================================================
# Drawing functions
# ============================================================
def draw_blocks(frame):
    for color in BLOCK_COLORS:
        if color not in block_points:
            continue

        if color not in remaining_blocks:
            continue

        x, y = block_points[color].astype(int)

        bgr = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
        }[color]

        cv2.circle(frame, (x, y), 12, bgr, -1)
        cv2.circle(frame, (x, y), int(TARGET_DISTANCE_THRESHOLD), bgr, 1)
        cv2.putText(
            frame,
            color,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bgr,
            2
        )


def draw_probabilities(frame, probabilities, best_color, best_prob):
    y0 = 30

    cv2.putText(
        frame,
        "Probabilities:",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y = y0 + 25

    for color in BLOCK_COLORS:
        if color not in remaining_blocks:
            continue

        prob = probabilities.get(color, 0.0)

        text = f"{color}: {prob:.2f}"
        if color == best_color:
            text += " <"

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 25

    cv2.putText(
        frame,
        f"Best: {best_color} {best_prob:.2f}",
        (10, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


def draw_status(frame, status_text, pinch_distance):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, h - 110), (w, h), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status_text,
        (10, h - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pinch distance: {pinch_distance:.1f}px  threshold<{PINCH_DISTANCE_THRESHOLD:.1f}",
        (10, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        system_message,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# Main
# ============================================================
def main():
    global hand_history, locked_target
    global stable_candidate, stable_start_time
    global system_message

    receiver_thread = threading.Thread(
        target=udp_receive_loop,
        daemon=True
    )
    receiver_thread.start()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cv2.namedWindow("Hand Intention Pixel Demo")
    cv2.setMouseCallback("Hand Intention Pixel Demo", mouse_callback)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            draw_blocks(frame)

            probabilities = {}
            best_color = None
            best_prob = 0.0
            distance_to_best = None
            pinch_distance = 999.0
            status_text = "No hand detected"

            if len(remaining_blocks) == 0:
                status_text = "All blocks used. Task finished."

            elif len(block_points) < len(BLOCK_COLORS):
                status_text = system_message

            elif results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                hand_pos, index_tip, thumb_tip = get_hand_points(
                    hand_landmarks,
                    w,
                    h
                )

                pinch_distance = distance(index_tip, thumb_tip)

                hand_history.append(hand_pos)

                if len(hand_history) > HAND_HISTORY_LENGTH:
                    hand_history = hand_history[-HAND_HISTORY_LENGTH:]

                hand_vel = compute_hand_velocity()

                cv2.circle(
                    frame,
                    tuple(hand_pos.astype(int)),
                    8,
                    (255, 255, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    tuple(index_tip.astype(int)),
                    7,
                    (0, 255, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    tuple(thumb_tip.astype(int)),
                    7,
                    (255, 0, 255),
                    -1
                )

                best_color, best_prob, probabilities, distance_to_best = predict_intention(
                    hand_pos,
                    hand_vel
                )

                locked, stable_duration, status_text = update_stable_lock(
                    best_color,
                    best_prob,
                    distance_to_best,
                    pinch_distance
                )

                if locked is not None:
                    send_intention_to_wsl(locked)

            else:
                hand_history = []
                stable_candidate = None
                stable_start_time = None
                status_text = "No hand detected"

            draw_probabilities(frame, probabilities, best_color, best_prob)
            draw_status(frame, status_text, pinch_distance)

            cv2.imshow("Hand Intention Pixel Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                block_points.clear()
                hand_history = []
                stable_candidate = None
                stable_start_time = None
                locked_target = None
                system_message = "Reset. Click block centers in order: red, blue, green, yellow, orange"
                print(system_message)

            elif key == ord("u"):
                # 手动清除当前 lock/stable 状态
                stable_candidate = None
                stable_start_time = None
                locked_target = None
                system_message = "Unlocked current target."
                print(system_message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
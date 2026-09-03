import cv2
import time
import socket
import threading
import numpy as np
import mediapipe as mp


# ==========================
# 1. UDP 设置：Windows -> WSL
# ==========================
WSL_IP = "172.26.149.220"   # 改成你在 WSL 中 hostname -I 显示的 IP
WSL_PORT = 5005

udp_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_sent_target = None


# ==========================
# 2. UDP 设置：WSL -> Windows
# 接收 WSL 发回的剩余方块列表
# ==========================
WINDOWS_RECEIVE_IP = "0.0.0.0"
WINDOWS_RECEIVE_PORT = 5006

remaining_blocks = ["red", "blue", "green", "yellow", "orange"]


# ==========================
# 3. 五个积木像素坐标
# 运行后用鼠标依次点击：
# red, blue, green, yellow, orange
# ==========================
BLOCKS_PIXEL = {
    "red": np.array([100, 300], dtype=np.float32),
    "blue": np.array([200, 300], dtype=np.float32),
    "green": np.array([300, 300], dtype=np.float32),
    "yellow": np.array([400, 300], dtype=np.float32),
    "orange": np.array([500, 300], dtype=np.float32),
}

BLOCK_COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
}

block_names = ["red", "blue", "green", "yellow", "orange"]
click_index = 0


# ==========================
# 4. 意图锁定状态
# ==========================
locked_target = None
lock_threshold = 0.55


def udp_receive_loop():
    """
    接收 WSL 发回来的剩余方块列表。
    消息格式：
        remaining:red,blue,yellow
        remaining:orange
        remaining:
    """
    global remaining_blocks, locked_target, last_sent_target

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((WINDOWS_RECEIVE_IP, WINDOWS_RECEIVE_PORT))

    print(f"Listening for remaining blocks on UDP port {WINDOWS_RECEIVE_PORT}...")

    while True:
        try:
            data, addr = recv_sock.recvfrom(1024)
            message = data.decode("utf-8").strip()

            if message.startswith("remaining:"):
                payload = message.replace("remaining:", "").strip()

                if payload == "":
                    remaining_blocks = []
                else:
                    remaining_blocks = [
                        item.strip()
                        for item in payload.split(",")
                        if item.strip() in block_names
                    ]

                print(f"Updated remaining blocks from WSL: {remaining_blocks}")

                # 收到任务状态更新后，自动重置本轮锁定，准备下一轮
                locked_target = None
                last_sent_target = None

        except Exception as e:
            print(f"UDP receive error: {e}")


def mouse_callback(event, x, y, flags, param):
    """
    用鼠标依次点击 red, blue, green, yellow, orange 五个积木中心。
    """
    global click_index

    if event == cv2.EVENT_LBUTTONDOWN:
        name = block_names[click_index % len(block_names)]
        BLOCKS_PIXEL[name] = np.array([x, y], dtype=np.float32)
        print(f"Set {name} block position to: ({x}, {y})")
        click_index += 1


def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_intention(hand_pos, hand_vel):
    """
    只对 remaining_blocks 中的颜色进行预测。

    规则：
    1. 手离哪个积木越近，分数越高
    2. 手运动方向越朝向哪个积木，分数越高
    """
    scores = []
    names = []

    speed = np.linalg.norm(hand_vel)

    for name, block_pos in BLOCKS_PIXEL.items():
        if name not in remaining_blocks:
            continue

        names.append(name)

        vec_to_block = block_pos - hand_pos
        distance = np.linalg.norm(vec_to_block)

        distance_score = -distance / 300.0

        if speed > 1e-6 and distance > 1e-6:
            direction_score = np.dot(hand_vel / speed, vec_to_block / distance)
        else:
            direction_score = 0.0

        score = 4.0 * distance_score + 2.0 * direction_score
        scores.append(score)

    if len(scores) == 0:
        return {}

    probs = softmax(scores)
    return dict(zip(names, probs))


# ==========================
# 5. 初始化 MediaPipe Hands
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# ==========================
# 6. 打开摄像头
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera. Try changing camera index 0 to 1.")
    exit()


cv2.namedWindow("Hand Intention Pixel Demo")
cv2.setMouseCallback("Hand Intention Pixel Demo", mouse_callback)


last_hand_pos = None
last_time = time.time()


# 启动接收 WSL 剩余方块的线程
receiver_thread = threading.Thread(target=udp_receive_loop, daemon=True)
receiver_thread.start()


print("\nInstructions:")
print("1. 在画面中依次点击 red, blue, green, yellow, orange 五个积木中心")
print("2. 把手伸向其中一个剩余积木")
print("3. 系统锁定后会通过 UDP 发送给 WSL/ROS2")
print("4. WSL 任务管理器完成一轮后，会发回 remaining blocks")
print("5. 画面只显示剩余方块")
print("6. 按 r 可以手动重置锁定")
print("7. 按 q 或 ESC 退出")
print(f"UDP send target: {WSL_IP}:{WSL_PORT}")
print(f"UDP receive port: {WINDOWS_RECEIVE_PORT}\n")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    now = time.time()
    dt = max(now - last_time, 1e-6)
    last_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_pos = None
    hand_vel = np.array([0.0, 0.0], dtype=np.float32)

    # ==========================
    # 7. 检测手部关键点
    # ==========================
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        wrist_lm = hand_landmarks.landmark[0]
        index_lm = hand_landmarks.landmark[8]

        wrist = np.array([wrist_lm.x * w, wrist_lm.y * h], dtype=np.float32)
        index_tip = np.array([index_lm.x * w, index_lm.y * h], dtype=np.float32)

        hand_pos = 0.5 * (wrist + index_tip)

        if last_hand_pos is not None:
            hand_vel = (hand_pos - last_hand_pos) / dt

        last_hand_pos = hand_pos.copy()

        cv2.circle(frame, tuple(hand_pos.astype(int)), 10, (255, 0, 255), -1)

    # ==========================
    # 8. 意图预测
    # ==========================
    probs = None
    predicted = None
    confidence = 0.0

    if hand_pos is not None and len(remaining_blocks) > 0:
        probs = predict_intention(hand_pos, hand_vel)

        if len(probs) > 0:
            raw_predicted = max(probs, key=probs.get)
            raw_confidence = probs[raw_predicted]

            if locked_target is None and raw_confidence >= lock_threshold:
                locked_target = raw_predicted
                print(f">>> Intention locked: {locked_target}, confidence={raw_confidence:.2f}")

            if locked_target is not None:
                predicted = locked_target
                confidence = probs.get(locked_target, 0.0)
            else:
                predicted = raw_predicted
                confidence = raw_confidence

    # ==========================
    # 9. UDP 发送锁定结果到 WSL
    # ==========================
    if locked_target is not None and locked_target != last_sent_target:
        udp_send_sock.sendto(
            locked_target.encode("utf-8"),
            (WSL_IP, WSL_PORT)
        )
        last_sent_target = locked_target
        print(f"Sent locked target to WSL: {locked_target}")

    # ==========================
    # 10. 画剩余积木点
    # ==========================
    for name, pos in BLOCKS_PIXEL.items():
        if name not in remaining_blocks:
            continue

        color = BLOCK_COLORS[name]
        center = tuple(pos.astype(int))

        cv2.circle(frame, center, 12, color, -1)
        cv2.putText(
            frame,
            name,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # ==========================
    # 11. 显示预测结果
    # ==========================
    if len(remaining_blocks) == 0:
        cv2.putText(
            frame,
            "All blocks used. Task finished.",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    elif probs is not None and predicted is not None:
        y0 = 40

        cv2.putText(
            frame,
            f"Prediction: {predicted}  conf={confidence:.2f}  locked={locked_target}",
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        y = y0 + 40
        for name in remaining_blocks:
            if probs is not None and name in probs:
                text = f"{name}: {probs[name]:.2f}"
            else:
                text = f"{name}: --"

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                BLOCK_COLORS[name],
                2,
            )
            y += 30

    # ==========================
    # 12. 显示剩余方块状态
    # ==========================
    remaining_text = "Remaining: " + ", ".join(remaining_blocks)
    cv2.putText(
        frame,
        remaining_text,
        (20, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Click blocks: red, blue, green, yellow, orange | r: reset | q: quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Hand Intention Pixel Demo", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key == ord("r"):
        locked_target = None
        last_sent_target = None
        last_hand_pos = None
        print("Manual reset. Ready for next intention.")


cap.release()
cv2.destroyAllWindows()
udp_send_sock.close()
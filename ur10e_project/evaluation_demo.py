import time
import numpy as np
import mujoco
import mujoco.viewer


BLOCKS = {
    "red": np.array([-0.30, 0.15, 0.09]),
    "blue": np.array([-0.10, 0.15, 0.09]),
    "green": np.array([0.10, 0.15, 0.09]),
    "yellow": np.array([0.30, 0.15, 0.09]),
}

# 人真实想拿的方块，用来生成“人手轨迹”
HUMAN_TARGET_BLOCK = "blue"   # 可改成 red / blue / green / yellow

HAND_START = np.array([0.0, -0.45, 0.16])
ROBOT_WAIT_POS = np.array([0.0, -0.25, 0.32])
ROBOT_START = ROBOT_WAIT_POS.copy()

# 搭塔区域
TOWER_BASE_POS = np.array([0.0, -0.18, 0.09])

# 方块边长 0.08，所以每层高度增加 0.08
BLOCK_HEIGHT = 0.08


def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def smoothstep(alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return 3 * alpha**2 - 2 * alpha**3


def predict_intention(hand_pos, hand_vel):
    """
    根据人手位置和速度预测人要拿哪个方块。

    特征：
    1. 距离特征：人手离哪个方块越近，该方块得分越高
    2. 方向特征：人手运动方向越朝向哪个方块，该方块得分越高
    """
    scores = []
    names = []

    speed = np.linalg.norm(hand_vel)

    for name, block_pos in BLOCKS.items():
        names.append(name)

        vec_to_block = block_pos - hand_pos
        distance = np.linalg.norm(vec_to_block)

        distance_score = -distance

        if speed > 1e-6 and distance > 1e-6:
            direction_score = np.dot(hand_vel / speed, vec_to_block / distance)
        else:
            direction_score = 0.0

        score = 4.0 * distance_score + 2.0 * direction_score
        scores.append(score)

    probs = softmax(scores)
    return dict(zip(names, probs))


def choose_robot_target(human_target):
    """
    当前简单策略：
    1. 避开人类目标
    2. 选择离机器人等待位置最近的方块
    """
    candidates = []

    for name, pos in BLOCKS.items():
        if name == human_target:
            continue

        dist_to_robot = np.linalg.norm(pos - ROBOT_START)
        candidates.append((dist_to_robot, name))

    candidates.sort()
    return candidates[0][1]


def human_motion_plan(t):
    """
    人手运动：
    1. 从起点移动到目标方块上方
    2. 下降到方块
    3. 搬到搭塔区域上方
    4. 放到搭塔第一层
    """
    block_pos = BLOCKS[HUMAN_TARGET_BLOCK]

    above_block = block_pos + np.array([0.0, 0.0, 0.18])
    grasp_pos = block_pos + np.array([0.0, 0.0, 0.08])

    human_place_pos = TOWER_BASE_POS
    above_tower = human_place_pos + np.array([0.0, 0.0, 0.20])
    place_pos = human_place_pos + np.array([0.0, 0.0, 0.08])

    segment_time = 1.5

    if t < segment_time:
        a = smoothstep(t / segment_time)
        return (1 - a) * HAND_START + a * above_block, "human_move_to_block"

    elif t < 2 * segment_time:
        a = smoothstep((t - segment_time) / segment_time)
        return (1 - a) * above_block + a * grasp_pos, "human_lower_to_grasp"

    elif t < 3 * segment_time:
        a = smoothstep((t - 2 * segment_time) / segment_time)
        return (1 - a) * grasp_pos + a * above_tower, "human_carry_to_tower"

    elif t < 4 * segment_time:
        a = smoothstep((t - 3 * segment_time) / segment_time)
        return (1 - a) * above_tower + a * place_pos, "human_place_block"

    else:
        return place_pos, "human_finished"


def robot_motion_plan(t, robot_target_name, decision_time):
    """
    机器人运动：
    决策前等待；
    决策后去拿自己的目标方块，并放到搭塔第二层。
    """
    if robot_target_name is None:
        return ROBOT_WAIT_POS, "robot_waiting"

    elapsed = t - decision_time

    block_pos = BLOCKS[robot_target_name]

    above_block = block_pos + np.array([0.0, 0.0, 0.22])
    grasp_pos = block_pos + np.array([0.0, 0.0, 0.08])

    # 机器人放第二层，比人放的方块高一层
    robot_place_base = TOWER_BASE_POS + np.array([0.0, 0.0, BLOCK_HEIGHT])
    above_tower = robot_place_base + np.array([0.0, 0.0, 0.22])
    place_pos = robot_place_base + np.array([0.0, 0.0, 0.08])

    segment_time = 1.5

    if elapsed < 0:
        return ROBOT_WAIT_POS, "robot_waiting"

    elif elapsed < segment_time:
        a = smoothstep(elapsed / segment_time)
        return (1 - a) * ROBOT_WAIT_POS + a * above_block, "robot_move_to_block"

    elif elapsed < 2 * segment_time:
        a = smoothstep((elapsed - segment_time) / segment_time)
        return (1 - a) * above_block + a * grasp_pos, "robot_lower_to_grasp"

    elif elapsed < 3 * segment_time:
        a = smoothstep((elapsed - 2 * segment_time) / segment_time)
        return (1 - a) * grasp_pos + a * above_tower, "robot_carry_to_tower"

    elif elapsed < 4 * segment_time:
        a = smoothstep((elapsed - 3 * segment_time) / segment_time)
        return (1 - a) * above_tower + a * place_pos, "robot_place_block"

    else:
        return place_pos, "robot_finished"


xml = """
<mujoco model="evaluation_demo">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="main_cam" pos="0 -1.8 1.2" xyaxes="1 0 0 0 0.6 1"/>

    <geom name="table" type="box"
          pos="0 0 0"
          size="0.7 0.5 0.03"
          material="grid_mat"/>

    <body name="red_block" mocap="true" pos="-0.3 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="1 0 0 1" mass="0.1"/>
    </body>

    <body name="blue_block" mocap="true" pos="-0.1 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="0 0.2 1 1" mass="0.1"/>
    </body>

    <body name="green_block" mocap="true" pos="0.1 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="0 0.8 0.2 1" mass="0.1"/>
    </body>

    <body name="yellow_block" mocap="true" pos="0.3 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="1 0.9 0 1" mass="0.1"/>
    </body>

    <geom name="tower_target" type="cylinder"
          pos="0 -0.18 0.035"
          size="0.08 0.005"
          rgba="1 1 1 0.35"
          contype="0" conaffinity="0"/>

    <!-- 紫色小球：人手 -->
    <body name="human_hand" mocap="true" pos="0 -0.45 0.16">
      <geom type="sphere" size="0.035" rgba="0.8 0 1 1"
            contype="0" conaffinity="0"/>
    </body>

    <!-- 青色小球：机器人末端 -->
    <body name="robot_ee" mocap="true" pos="0 -0.35 0.32">
      <geom type="sphere" size="0.035" rgba="0 0.9 1 1"
            contype="0" conaffinity="0"/>
    </body>

  </worldbody>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

human_mocap_id = model.body("human_hand").mocapid[0]
robot_mocap_id = model.body("robot_ee").mocapid[0]

block_mocap_ids = {
    "red": model.body("red_block").mocapid[0],
    "blue": model.body("blue_block").mocapid[0],
    "green": model.body("green_block").mocapid[0],
    "yellow": model.body("yellow_block").mocapid[0],
}

last_hand_pos = HAND_START.copy()
last_print_time = 0.0

# ==========================
# 人类意图锁定
# ==========================
locked_target = None
lock_threshold = 0.55

# ==========================
# 机器人决策
# ==========================
robot_target = None
decision_time = None
decision_made = False

# ==========================
# 抓取状态
# ==========================
human_attached_block = None
human_released_block = False

robot_attached_block = None
robot_released_block = False

# ==========================
# 评估指标
# ==========================
prediction_time = None
human_grasp_time = None
task_finish_time = None

intention_correct = False
conflict_count = 0
min_human_robot_distance = float("inf")

# 记录每个方块当前位置
block_positions = {name: pos.copy() for name, pos in BLOCKS.items()}

evaluation_printed = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.4
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25

    start_time = time.time()

    while viewer.is_running():
        now = time.time()
        t = now - start_time

        # ==========================
        # 1. 人手运动
        # ==========================
        hand_pos, human_phase = human_motion_plan(t)
        data.mocap_pos[human_mocap_id] = hand_pos

        dt = 0.002
        hand_vel = (hand_pos - last_hand_pos) / dt
        last_hand_pos = hand_pos.copy()

        # ==========================
        # 2. 人类意图识别
        # ==========================
        probs = predict_intention(hand_pos, hand_vel)

        raw_predicted = max(probs, key=probs.get)
        raw_confidence = probs[raw_predicted]

        if locked_target is None and raw_confidence >= lock_threshold:
            locked_target = raw_predicted
            prediction_time = t
            intention_correct = locked_target == HUMAN_TARGET_BLOCK

            print(f"\n>>> Human intention locked: {locked_target} at {prediction_time:.2f}s <<<")
            print(f">>> Intention correct: {intention_correct} <<<\n")

        if locked_target is not None:
            predicted = locked_target
            confidence = probs[locked_target]
            is_locked = True
        else:
            predicted = raw_predicted
            confidence = raw_confidence
            is_locked = False

        # ==========================
        # 3. 机器人协同决策
        # ==========================
        if locked_target is not None and not decision_made:
            robot_target = choose_robot_target(locked_target)
            decision_time = t
            decision_made = True

            if robot_target == HUMAN_TARGET_BLOCK:
                conflict_count += 1

            print(f"\n>>> Robot avoids {locked_target}, chooses {robot_target} <<<")
            print(f">>> Conflict count: {conflict_count} <<<\n")

        # ==========================
        # 4. 机器人运动
        # ==========================
        robot_pos, robot_phase = robot_motion_plan(
            t,
            robot_target,
            decision_time if decision_time is not None else 0.0
        )
        data.mocap_pos[robot_mocap_id] = robot_pos

        # 记录人手与机器人末端的最小距离
        human_robot_distance = np.linalg.norm(hand_pos - robot_pos)
        min_human_robot_distance = min(min_human_robot_distance, human_robot_distance)

        # ==========================
        # 5. 人类抓取和搬运
        # ==========================
        if human_phase in ["human_carry_to_tower", "human_place_block"] and human_attached_block is None:
            human_attached_block = HUMAN_TARGET_BLOCK
            human_grasp_time = t
            print(f"\n>>> Human picked up {human_attached_block} block at {human_grasp_time:.2f}s <<<\n")

        if human_attached_block is not None and not human_released_block:
            block_positions[human_attached_block] = hand_pos - np.array([0.0, 0.0, 0.08])

        if human_phase == "human_finished" and human_attached_block is not None and not human_released_block:
            block_positions[human_attached_block] = TOWER_BASE_POS.copy()
            human_released_block = True
            print(f"\n>>> Human placed {human_attached_block} block at tower base <<<\n")

        # ==========================
        # 6. 机器人抓取和搬运
        # ==========================
        if robot_target is not None:
            if robot_phase in ["robot_carry_to_tower", "robot_place_block"] and robot_attached_block is None:
                robot_attached_block = robot_target
                print(f"\n>>> Robot picked up {robot_attached_block} block <<<\n")

            if robot_attached_block is not None and not robot_released_block:
                block_positions[robot_attached_block] = robot_pos - np.array([0.0, 0.0, 0.08])

            if robot_phase == "robot_finished" and robot_attached_block is not None and not robot_released_block:
                block_positions[robot_attached_block] = TOWER_BASE_POS + np.array([0.0, 0.0, BLOCK_HEIGHT])
                robot_released_block = True
                task_finish_time = t

                print(f"\n>>> Robot placed {robot_attached_block} block on second layer <<<\n")

        # ==========================
        # 7. 更新方块位置
        # ==========================
        for name, pos in block_positions.items():
            data.mocap_pos[block_mocap_ids[name]] = pos

        # ==========================
        # 8. 打印状态
        # ==========================
        if now - last_print_time > 0.3:
            last_print_time = now
            print(
                f"Human intention={predicted:6s} | "
                f"locked={is_locked} | "
                f"conf={confidence:.2f} | "
                f"human_phase={human_phase} | "
                f"robot_target={robot_target} | "
                f"robot_phase={robot_phase} | "
                f"min_dist={min_human_robot_distance:.3f}m"
            )

        # ==========================
        # 9. 最终评估结果
        # ==========================
        if robot_released_block and human_released_block and not evaluation_printed:
            evaluation_printed = True

            if prediction_time is not None and human_grasp_time is not None:
                lead_time = human_grasp_time - prediction_time
            else:
                lead_time = None

            print("\n========== Evaluation Results ==========")
            print(f"Human target block: {HUMAN_TARGET_BLOCK}")
            print(f"Predicted target: {locked_target}")
            print(f"Intention correct: {intention_correct}")

            if prediction_time is not None:
                print(f"Prediction time: {prediction_time:.2f}s")
            else:
                print("Prediction time: None")

            if human_grasp_time is not None:
                print(f"Human grasp time: {human_grasp_time:.2f}s")
            else:
                print("Human grasp time: None")

            if lead_time is not None:
                print(f"Lead time: {lead_time:.2f}s")
            else:
                print("Lead time: None")

            print(f"Conflict count: {conflict_count}")
            print(f"Minimum human-robot distance: {min_human_robot_distance:.3f} m")

            if task_finish_time is not None:
                print(f"Task finish time: {task_finish_time:.2f}s")
            else:
                print("Task finish time: None")

            print("========================================\n")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
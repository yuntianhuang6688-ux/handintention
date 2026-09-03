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
HUMAN_TARGET_BLOCK = "green"   # 可改成 red / blue / green / yellow

HAND_START = np.array([0.0, -0.45, 0.16])
HAND_END = BLOCKS[HUMAN_TARGET_BLOCK] + np.array([0.0, 0.0, 0.08])

ROBOT_START = np.array([0.0, -0.35, 0.32])
ROBOT_WAIT_POS = np.array([0.0, -0.25, 0.32])

# 搭塔目标位置
TOWER_POS = np.array([0.0, -0.18, 0.10])


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
    机器人协同决策：
    避开人类已经锁定的目标方块，选择另一个方块。
    
    目前用简单规则：
    1. 不选择 human_target
    2. 优先选择离机器人起点最近的方块
    """
    candidates = []

    for name, pos in BLOCKS.items():
        if name == human_target:
            continue

        dist_to_robot = np.linalg.norm(pos - ROBOT_START)
        candidates.append((dist_to_robot, name))

    candidates.sort()
    return candidates[0][1]


def robot_motion_plan(t, robot_target_name, decision_time):
    """
    简化机器人末端运动：
    决策前：停在等待位置
    决策后：
      1. 移动到目标方块上方
      2. 下降到方块附近
      3. 移动到搭塔区域上方
      4. 下降到搭塔区域
    """
    if robot_target_name is None:
        return ROBOT_WAIT_POS

    elapsed = t - decision_time

    block_pos = BLOCKS[robot_target_name]

    above_block = block_pos + np.array([0.0, 0.0, 0.22])
    grasp_pos = block_pos + np.array([0.0, 0.0, 0.08])
    above_tower = TOWER_POS + np.array([0.0, 0.0, 0.22])
    place_pos = TOWER_POS + np.array([0.0, 0.0, 0.08])

    # 每段运动时间
    segment_time = 1.5

    if elapsed < 0:
        return ROBOT_WAIT_POS

    elif elapsed < segment_time:
        a = smoothstep(elapsed / segment_time)
        return (1 - a) * ROBOT_WAIT_POS + a * above_block

    elif elapsed < 2 * segment_time:
        a = smoothstep((elapsed - segment_time) / segment_time)
        return (1 - a) * above_block + a * grasp_pos

    elif elapsed < 3 * segment_time:
        a = smoothstep((elapsed - 2 * segment_time) / segment_time)
        return (1 - a) * grasp_pos + a * above_tower

    elif elapsed < 4 * segment_time:
        a = smoothstep((elapsed - 3 * segment_time) / segment_time)
        return (1 - a) * above_tower + a * place_pos

    else:
        return place_pos


xml = """
<mujoco model="cooperation_demo">
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

    <body name="red_block" pos="-0.3 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="1 0 0 1" mass="0.1"/>
    </body>

    <body name="blue_block" pos="-0.1 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="0 0.2 1 1" mass="0.1"/>
    </body>

    <body name="green_block" pos="0.1 0.15 0.09">
      <geom type="box" size="0.04 0.04 0.04" rgba="0 0.8 0.2 1" mass="0.1"/>
    </body>

    <body name="yellow_block" pos="0.3 0.15 0.09">
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

    <!-- 青色小球：简化机器人末端/夹爪 -->
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

last_hand_pos = HAND_START.copy()
last_print_time = 0.0

# 意图锁定
locked_target = None
lock_threshold = 0.55

# 机器人决策
robot_target = None
decision_time = None
decision_made = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.4
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25

    start_time = time.time()

    while viewer.is_running():
        now = time.time()
        t = now - start_time

        # ==========================
        # 1. 生成人手轨迹
        # ==========================
        alpha = min(t / 4.0, 1.0)
        alpha_smooth = smoothstep(alpha)
        hand_pos = (1 - alpha_smooth) * HAND_START + alpha_smooth * HAND_END

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
            print(f"\n>>> Human intention locked: {locked_target} <<<\n")

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
            print(f"\n>>> Robot avoids {locked_target}, chooses {robot_target} <<<\n")

        # ==========================
        # 4. 机器人末端运动
        # ==========================
        robot_pos = robot_motion_plan(t, robot_target, decision_time if decision_time is not None else 0.0)
        data.mocap_pos[robot_mocap_id] = robot_pos

        # ==========================
        # 5. 打印状态
        # ==========================
        if now - last_print_time > 0.3:
            last_print_time = now
            print(
                f"Human: {predicted:6s} | "
                f"locked={is_locked} | "
                f"conf={confidence:.2f} | "
                f"Robot target={robot_target} | "
                f"red={probs['red']:.2f}, "
                f"blue={probs['blue']:.2f}, "
                f"green={probs['green']:.2f}, "
                f"yellow={probs['yellow']:.2f}"
            )

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)

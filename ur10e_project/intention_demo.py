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
TARGET_BLOCK = "green"   # 可以改成 red / blue / green / yellow

HAND_START = np.array([0.0, -0.45, 0.16])
HAND_END = BLOCKS[TARGET_BLOCK] + np.array([0.0, 0.0, 0.08])


def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


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

        # 距离分数：距离越小，分数越高
        distance_score = -distance

        # 方向分数：速度方向与“手到方块方向”越一致，分数越高
        if speed > 1e-6 and distance > 1e-6:
            direction_score = np.dot(hand_vel / speed, vec_to_block / distance)
        else:
            direction_score = 0.0

        # 综合分数
        score = 4.0 * distance_score + 2.0 * direction_score
        scores.append(score)

    probs = softmax(scores)
    return dict(zip(names, probs))


xml = """
<mujoco model="intention_demo">
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

    <!-- 简化的人手，用一个紫色小球表示 -->
    <body name="human_hand" mocap="true" pos="0 -0.45 0.16">
      <geom type="sphere" size="0.035" rgba="0.8 0 1 1"
            contype="0" conaffinity="0"/>
    </body>

  </worldbody>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

hand_mocap_id = model.body("human_hand").mocapid[0]

last_hand_pos = HAND_START.copy()
last_print_time = 0.0

# ==========================
# 意图锁定参数
# ==========================
locked_target = None
lock_threshold = 0.55

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.4
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25

    start_time = time.time()

    while viewer.is_running():
        now = time.time()
        t = now - start_time

        # 让人手在 4 秒内从起点移动到目标方块
        alpha = min(t / 4.0, 1.0)

        # 平滑轨迹
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        hand_pos = (1 - alpha_smooth) * HAND_START + alpha_smooth * HAND_END

        # 更新 MuJoCo 中的人手位置
        data.mocap_pos[hand_mocap_id] = hand_pos

        # 估计速度
        dt = 0.002
        hand_vel = (hand_pos - last_hand_pos) / dt
        last_hand_pos = hand_pos.copy()

        # 计算意图概率
        probs = predict_intention(hand_pos, hand_vel)

        raw_predicted = max(probs, key=probs.get)
        raw_confidence = probs[raw_predicted]

        # ==========================
        # 意图锁定逻辑
        # ==========================
        if locked_target is None and raw_confidence >= lock_threshold:
            locked_target = raw_predicted
            print(f"\n>>> Intention locked: {locked_target} <<<\n")

        if locked_target is not None:
            predicted = locked_target
            confidence = probs[locked_target]
            is_locked = True
        else:
            predicted = raw_predicted
            confidence = raw_confidence
            is_locked = False

        # 每 0.3 秒打印一次预测结果
        if now - last_print_time > 0.3:
            last_print_time = now
            print(
                f"Prediction: {predicted:6s} | "
                f"locked={is_locked} | "
                f"confidence={confidence:.2f} | "
                f"red={probs['red']:.2f}, "
                f"blue={probs['blue']:.2f}, "
                f"green={probs['green']:.2f}, "
                f"yellow={probs['yellow']:.2f}"
            )

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
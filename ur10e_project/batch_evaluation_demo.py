import csv
import numpy as np


BLOCKS = {
    "red": np.array([-0.30, 0.15, 0.09]),
    "blue": np.array([-0.10, 0.15, 0.09]),
    "green": np.array([0.10, 0.15, 0.09]),
    "yellow": np.array([0.30, 0.15, 0.09]),
}

TARGET_BLOCKS = ["red", "blue", "green", "yellow"]

HAND_START = np.array([0.0, -0.45, 0.16])
ROBOT_WAIT_POS = np.array([0.0, -0.25, 0.32])
ROBOT_START = ROBOT_WAIT_POS.copy()

TOWER_BASE_POS = np.array([0.0, -0.18, 0.09])
BLOCK_HEIGHT = 0.08

LOCK_THRESHOLD = 0.55
DT = 0.002
MAX_TIME = 10.0


def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def smoothstep(alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return 3 * alpha**2 - 2 * alpha**3


def predict_intention(hand_pos, hand_vel):
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
    当前策略：
    避开人类目标，选择离机器人等待位置最近的方块。
    """
    candidates = []

    for name, pos in BLOCKS.items():
        if name == human_target:
            continue

        dist_to_robot = np.linalg.norm(pos - ROBOT_START)
        candidates.append((dist_to_robot, name))

    candidates.sort()
    return candidates[0][1]


def human_motion_plan(t, human_target_block):
    block_pos = BLOCKS[human_target_block]

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
    if robot_target_name is None:
        return ROBOT_WAIT_POS, "robot_waiting"

    elapsed = t - decision_time

    block_pos = BLOCKS[robot_target_name]

    above_block = block_pos + np.array([0.0, 0.0, 0.22])
    grasp_pos = block_pos + np.array([0.0, 0.0, 0.08])

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


def run_one_trial(human_target_block):
    locked_target = None
    prediction_time = None
    human_grasp_time = None
    task_finish_time = None

    intention_correct = False
    conflict_count = 0
    min_human_robot_distance = float("inf")

    robot_target = None
    decision_time = None
    decision_made = False

    human_attached_block = None
    human_released_block = False

    robot_attached_block = None
    robot_released_block = False

    last_hand_pos = HAND_START.copy()

    t = 0.0

    while t <= MAX_TIME:
        # 人手轨迹
        hand_pos, human_phase = human_motion_plan(t, human_target_block)
        hand_vel = (hand_pos - last_hand_pos) / DT
        last_hand_pos = hand_pos.copy()

        # 意图识别
        probs = predict_intention(hand_pos, hand_vel)
        raw_predicted = max(probs, key=probs.get)
        raw_confidence = probs[raw_predicted]

        if locked_target is None and raw_confidence >= LOCK_THRESHOLD:
            locked_target = raw_predicted
            prediction_time = t
            intention_correct = locked_target == human_target_block

        # 机器人决策
        if locked_target is not None and not decision_made:
            robot_target = choose_robot_target(locked_target)
            decision_time = t
            decision_made = True

            if robot_target == human_target_block:
                conflict_count += 1

        # 机器人运动
        robot_pos, robot_phase = robot_motion_plan(
            t,
            robot_target,
            decision_time if decision_time is not None else 0.0
        )

        # 人机距离
        human_robot_distance = np.linalg.norm(hand_pos - robot_pos)
        min_human_robot_distance = min(min_human_robot_distance, human_robot_distance)

        # 人类抓取时间
        if human_phase in ["human_carry_to_tower", "human_place_block"] and human_attached_block is None:
            human_attached_block = human_target_block
            human_grasp_time = t

        if human_phase == "human_finished" and human_attached_block is not None and not human_released_block:
            human_released_block = True

        # 机器人抓取和放置
        if robot_target is not None:
            if robot_phase in ["robot_carry_to_tower", "robot_place_block"] and robot_attached_block is None:
                robot_attached_block = robot_target

            if robot_phase == "robot_finished" and robot_attached_block is not None and not robot_released_block:
                robot_released_block = True
                task_finish_time = t

        # 实验结束
        if human_released_block and robot_released_block:
            break

        t += DT

    if prediction_time is not None and human_grasp_time is not None:
        lead_time = human_grasp_time - prediction_time
    else:
        lead_time = None

    return {
        "human_target": human_target_block,
        "predicted_target": locked_target,
        "intention_correct": intention_correct,
        "prediction_time": prediction_time,
        "human_grasp_time": human_grasp_time,
        "lead_time": lead_time,
        "robot_target": robot_target,
        "conflict_count": conflict_count,
        "min_human_robot_distance": min_human_robot_distance,
        "task_finish_time": task_finish_time,
    }


def main():
    results = []

    print("\nRunning batch evaluation...\n")

    for target in TARGET_BLOCKS:
        result = run_one_trial(target)
        results.append(result)

        print(
            f"Target={result['human_target']:6s} | "
            f"Predicted={str(result['predicted_target']):6s} | "
            f"Correct={result['intention_correct']} | "
            f"Lead time={result['lead_time']:.2f}s | "
            f"Robot target={result['robot_target']} | "
            f"Conflict={result['conflict_count']} | "
            f"Min distance={result['min_human_robot_distance']:.3f}m | "
            f"Finish={result['task_finish_time']:.2f}s"
        )

    csv_file = "experiment_results.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "human_target",
                "predicted_target",
                "intention_correct",
                "prediction_time",
                "human_grasp_time",
                "lead_time",
                "robot_target",
                "conflict_count",
                "min_human_robot_distance",
                "task_finish_time",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    correct_count = sum(1 for r in results if r["intention_correct"])
    accuracy = correct_count / len(results)

    lead_times = [r["lead_time"] for r in results if r["lead_time"] is not None]
    avg_lead_time = sum(lead_times) / len(lead_times)

    avg_conflict = sum(r["conflict_count"] for r in results) / len(results)
    avg_min_distance = sum(r["min_human_robot_distance"] for r in results) / len(results)
    avg_finish_time = sum(r["task_finish_time"] for r in results) / len(results)

    print("\n========== Batch Evaluation Summary ==========")
    print(f"Number of trials: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average lead time: {avg_lead_time:.2f}s")
    print(f"Average conflict count: {avg_conflict:.2f}")
    print(f"Average minimum human-robot distance: {avg_min_distance:.3f}m")
    print(f"Average task finish time: {avg_finish_time:.2f}s")
    print(f"CSV saved to: {csv_file}")
    print("==============================================\n")


if __name__ == "__main__":
    main()

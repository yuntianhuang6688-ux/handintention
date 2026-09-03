import os
import time
import json
import socket
import argparse
import tempfile
import random
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/home/yuntian/ur10e_project"

UR10E_MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "mujoco_menagerie",
    "universal_robots_ur10e",
)

UR10E_XML = os.path.join(UR10E_MODEL_DIR, "ur10e.xml")
WAYPOINT_FILE = os.path.join(PROJECT_DIR, "waypoints.json")


# ============================================================
# Basic setup
# ============================================================

TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.06

BLOCK_SIZE = 0.04
BLOCK_CENTER_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.0005

BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]
VALID_TARGETS = set(BLOCK_COLORS)

UDP_PORT = 5005

END_EFFECTOR_SITE = "attachment_site"
GRIPPER_FREEJOINT_NAME = "gripper_visual_freejoint"

# This version uses a world-vertical visual gripper.
# The gripper follows the UR10e end-effector position, but not its tilted rotation.
VERTICAL_GRIPPER_MOUNT_WORLD_OFFSET = np.array([0.0, 0.0, -0.080], dtype=np.float64)
VERTICAL_GRIPPER_TIP_WORLD_OFFSET = np.array([0.0, 0.0, -0.170], dtype=np.float64)

GRASP_DISTANCE_THRESHOLD = 0.20


# ============================================================
# Globals loaded from waypoints.json
# ============================================================

READY_Q = None
MID_Q = None
PICK_APPROACH_Q = None
PICK_DOWN_Q = None
PLACE_APPROACH_Q = None
PLACE_DOWN_Q = None

GRID_POINTS = []
PLACE_POINT = None
PLACE_POSITIONS = {}

BLOCKS = {}


@dataclass
class PickPlaceResult:
    human_target: str
    robot_target: str
    success: bool
    conflict: bool
    duration: float


# ============================================================
# Load generated waypoints
# ============================================================

def load_waypoints():
    global READY_Q, MID_Q
    global PICK_APPROACH_Q, PICK_DOWN_Q
    global PLACE_APPROACH_Q, PLACE_DOWN_Q
    global GRID_POINTS, PLACE_POINT, PLACE_POSITIONS

    if not os.path.exists(WAYPOINT_FILE):
        raise FileNotFoundError(
            f"Cannot find waypoint file:\n{WAYPOINT_FILE}\n"
            "Please run auto_generate_waypoints.py first."
        )

    with open(WAYPOINT_FILE, "r") as f:
        wp = json.load(f)

    READY_Q = np.array(wp["ready"], dtype=np.float64)
    MID_Q = np.array(wp["mid"], dtype=np.float64)
    PICK_APPROACH_Q = np.array(wp["pick_approach"], dtype=np.float64)
    PICK_DOWN_Q = np.array(wp["pick_down"], dtype=np.float64)
    PLACE_APPROACH_Q = np.array(wp["place_approach"], dtype=np.float64)
    PLACE_DOWN_Q = np.array(wp["place_down"], dtype=np.float64)

    raw_grid_points = [np.array(p, dtype=np.float64) for p in wp["grid_points"]]
    GRID_POINTS = []
    for p in raw_grid_points:
        p_fixed = p.copy()
        p_fixed[2] = BLOCK_CENTER_Z
        GRID_POINTS.append(p_fixed)

    PLACE_POINT = np.array(wp["place_point"], dtype=np.float64)
    PLACE_POINT[2] = BLOCK_CENTER_Z

    PLACE_POSITIONS = {
        "red":    PLACE_POINT + np.array([-0.08, 0.00, 0.00], dtype=np.float64),
        "blue":   PLACE_POINT + np.array([-0.04, 0.00, 0.00], dtype=np.float64),
        "green":  PLACE_POINT + np.array([0.00, 0.00, 0.00], dtype=np.float64),
        "yellow": PLACE_POINT + np.array([0.04, 0.00, 0.00], dtype=np.float64),
        "orange": PLACE_POINT + np.array([0.08, 0.00, 0.00], dtype=np.float64),
    }

    print("")
    print("========================================")
    print("Loaded waypoints from:", WAYPOINT_FILE)
    print("READY_Q:", np.round(READY_Q, 3))
    print("MID_Q:", np.round(MID_Q, 3))
    print("PICK_APPROACH_Q:", np.round(PICK_APPROACH_Q, 3))
    print("PICK_DOWN_Q:", np.round(PICK_DOWN_Q, 3))
    print("PLACE_APPROACH_Q:", np.round(PLACE_APPROACH_Q, 3))
    print("PLACE_DOWN_Q:", np.round(PLACE_DOWN_Q, 3))
    print("BLOCK_CENTER_Z:", BLOCK_CENTER_Z)
    print("========================================")


# ============================================================
# Block layout
# ============================================================

def randomize_blocks_in_grid(seed=None, fixed=False):
    global BLOCKS

    if fixed:
        selected_ids = [0, 1, 2, 3, 4]
    else:
        rng = random.Random(seed)
        selected_ids = rng.sample(list(range(len(GRID_POINTS))), k=5)

    BLOCKS = {}

    for i, color in enumerate(BLOCK_COLORS):
        p = GRID_POINTS[selected_ids[i]].copy()
        p[2] = BLOCK_CENTER_Z
        BLOCKS[color] = p

    print("")
    print("========================================")
    print("Block positions")
    for color in BLOCK_COLORS:
        print(f"{color:>6}: {np.round(BLOCKS[color], 3)}")
    print("========================================")


# ============================================================
# XML scene
# ============================================================

def create_scene_xml() -> str:
    rgba_map = {
        "red": "1 0.05 0.05 1",
        "blue": "0.05 0.25 1 1",
        "green": "0.05 0.8 0.15 1",
        "yellow": "1 0.85 0.05 1",
        "orange": "1 0.45 0.05 1",
    }

    marker_rgba_map = {
        "red": "1 0.05 0.05 0.35",
        "blue": "0.05 0.25 1 0.35",
        "green": "0.05 0.8 0.15 0.35",
        "yellow": "1 0.85 0.05 0.35",
        "orange": "1 0.45 0.05 0.35",
    }

    block_xml = ""
    for name, pos in BLOCKS.items():
        block_xml += f"""
        <body name="{name}_block" pos="{pos[0]} {pos[1]} {BLOCK_CENTER_Z}">
            <freejoint name="{name}_freejoint"/>
            <geom name="{name}_geom"
                  type="box"
                  size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0}"
                  rgba="{rgba_map[name]}"
                  mass="0.05"
                  friction="1.2 0.01 0.001"
                  condim="3"/>
        </body>
        """

    place_marker_xml = ""
    for name, pos in PLACE_POSITIONS.items():
        place_marker_xml += f"""
        <geom name="{name}_place_marker"
              type="box"
              pos="{pos[0]} {pos[1]} {TABLE_TOP_Z + 0.004}"
              size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} 0.004"
              rgba="{marker_rgba_map[name]}"
              contype="0"
              conaffinity="0"/>
        """

    grid_marker_xml = ""
    for i, p in enumerate(GRID_POINTS):
        grid_marker_xml += f"""
        <geom name="grid_marker_{i}"
              type="box"
              pos="{p[0]} {p[1]} {TABLE_TOP_Z + 0.003}"
              size="0.025 0.025 0.003"
              rgba="0.02 0.02 0.02 1"
              contype="0"
              conaffinity="0"/>
        """

    grid_center = np.mean(np.array(GRID_POINTS), axis=0)
    place_center = np.mean(np.array(list(PLACE_POSITIONS.values())), axis=0)

    table_center = 0.5 * (grid_center + place_center)

    xml = f"""
<mujoco model="ur10e_camera_intention_pick_place">

    <include file="ur10e.xml"/>

    <statistic center="{table_center[0]} {table_center[1]} 0.25" extent="1.8"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <visual>
        <headlight diffuse="0.65 0.65 0.65"
                   ambient="0.35 0.35 0.35"
                   specular="0.1 0.1 0.1"/>
        <global azimuth="130" elevation="-26"/>
        <rgba haze="0.15 0.20 0.25 1"/>
    </visual>

    <worldbody>
        <light name="top_light"
               pos="{table_center[0]} {table_center[1]} 2.2"
               dir="0 0 -1"
               diffuse="0.85 0.85 0.85"/>

        <geom name="floor"
              type="plane"
              size="2.5 2.5 0.01"
              rgba="0.80 0.80 0.80 1"/>

        <body name="table" pos="{table_center[0]} {table_center[1]} {-TABLE_THICKNESS / 2.0}">
            <geom name="table_top"
                  type="box"
                  size="1.00 0.65 {TABLE_THICKNESS / 2.0}"
                  rgba="0.90 0.90 0.90 1"
                  friction="1.0 0.01 0.001"/>
        </body>

        <geom name="front_black_rail"
              type="box"
              pos="{table_center[0]} {table_center[1] - 0.67} -0.02"
              size="1.05 0.025 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="back_black_rail"
              type="box"
              pos="{table_center[0]} {table_center[1] + 0.67} -0.02"
              size="1.05 0.025 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="left_black_rail"
              type="box"
              pos="{table_center[0] - 1.05} {table_center[1]} -0.02"
              size="0.025 0.67 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="right_black_rail"
              type="box"
              pos="{table_center[0] + 1.05} {table_center[1]} -0.02"
              size="0.025 0.67 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="robot_base_plate"
              type="box"
              pos="0.0 0.0 {TABLE_TOP_Z + 0.005}"
              size="0.15 0.15 0.005"
              rgba="0.12 0.12 0.15 1"
              contype="0"
              conaffinity="0"/>

        <geom name="blue_baseplate"
              type="box"
              pos="{place_center[0]} {place_center[1]} {TABLE_TOP_Z + 0.006}"
              size="0.30 0.10 0.006"
              rgba="0.02 0.28 0.95 0.75"
              contype="0"
              conaffinity="0"/>

        <geom name="place_area"
              type="box"
              pos="{place_center[0]} {place_center[1]} {TABLE_TOP_Z + 0.012}"
              size="0.34 0.12 0.003"
              rgba="0.1 0.8 0.1 0.20"
              contype="0"
              conaffinity="0"/>

        {place_marker_xml}

        <geom name="marker_board"
              type="box"
              pos="{grid_center[0]} {grid_center[1]} {TABLE_TOP_Z + 0.001}"
              size="0.24 0.22 0.001"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        {grid_marker_xml}

        {block_xml}

        <!-- World-vertical simplified gripper.
             It follows the end-effector position but stays vertical. -->
        <body name="gripper_visual" pos="0 0 0">
            <freejoint name="{GRIPPER_FREEJOINT_NAME}"/>

            <geom name="gripper_palm"
                  type="box"
                  pos="0 0 0"
                  size="0.035 0.028 0.022"
                  rgba="0.15 0.15 0.16 1"
                  contype="0"
                  conaffinity="0"/>

            <geom name="gripper_left_finger"
                  type="box"
                  pos="0 0.020 -0.055"
                  size="0.008 0.006 0.055"
                  rgba="0.05 0.05 0.05 1"
                  contype="0"
                  conaffinity="0"/>

            <geom name="gripper_right_finger"
                  type="box"
                  pos="0 -0.020 -0.055"
                  size="0.008 0.006 0.055"
                  rgba="0.05 0.05 0.05 1"
                  contype="0"
                  conaffinity="0"/>

            <geom name="gripper_tip_pad"
                  type="box"
                  pos="0 0 -0.110"
                  size="0.020 0.030 0.006"
                  rgba="0.02 0.02 0.02 1"
                  contype="0"
                  conaffinity="0"/>

            <site name="gripper_tip_visual"
                  pos="0 0 -0.115"
                  size="0.006"
                  rgba="1 0 1 1"/>
        </body>

        <camera name="overview"
                pos="{table_center[0] + 0.9} {table_center[1] - 1.1} 1.05"
                xyaxes="0.70 0.71 0 -0.35 0.35 0.87"/>
    </worldbody>

</mujoco>
"""
    return xml


# ============================================================
# MuJoCo helpers
# ============================================================

def name_to_id(model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Cannot find MuJoCo object: {name}")
    return idx


def get_site_pos(model, data, site_name: str) -> np.ndarray:
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def set_freejoint_pose(model, data, joint_name: str, pos: np.ndarray, quat: np.ndarray):
    joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    qvel_addr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_addr:qpos_addr + 7] = np.array([
        pos[0], pos[1], pos[2],
        quat[0], quat[1], quat[2], quat[3]
    ], dtype=np.float64)

    data.qvel[qvel_addr:qvel_addr + 6] = 0.0


def set_robot_qpos(model, data, q):
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def set_block_pose(model, data, block_name: str, pos: np.ndarray):
    safe_pos = pos.copy()
    safe_pos[2] = BLOCK_CENTER_Z

    set_freejoint_pose(
        model,
        data,
        f"{block_name}_freejoint",
        safe_pos,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )


def get_block_pos(model, data, block_name: str) -> np.ndarray:
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{block_name}_block")
    return data.xpos[body_id].copy()


def reset_blocks(model, data):
    for name, pos in BLOCKS.items():
        set_block_pose(model, data, name, pos)
    mujoco.mj_forward(model, data)


def update_gripper_visual(model, data):
    site_pos = get_site_pos(model, data, END_EFFECTOR_SITE)

    gripper_world_pos = site_pos + VERTICAL_GRIPPER_MOUNT_WORLD_OFFSET

    # Identity quaternion: keep gripper vertical in world frame.
    world_vertical_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    set_freejoint_pose(
        model,
        data,
        GRIPPER_FREEJOINT_NAME,
        gripper_world_pos,
        world_vertical_quat
    )


def get_gripper_tip_world(model, data) -> np.ndarray:
    site_pos = get_site_pos(model, data, END_EFFECTOR_SITE)
    return site_pos + VERTICAL_GRIPPER_TIP_WORLD_OFFSET


# ============================================================
# Simulation movement
# ============================================================

def sync_viewer(viewer, model, data, realtime=True):
    if viewer is not None and viewer.is_running():
        viewer.sync()

    if realtime:
        time.sleep(model.opt.timestep)


def hold(model, data, viewer=None, duration=0.5,
         attached_block=None,
         attach_offset_world=None,
         realtime=True):

    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        data.ctrl[:6] = data.qpos[:6]
        update_gripper_visual(model, data)

        if attached_block is not None:
            tip_world = get_gripper_tip_world(model, data)
            set_block_pose(model, data, attached_block, tip_world + attach_offset_world)

        mujoco.mj_step(model, data)
        sync_viewer(viewer, model, data, realtime=realtime)


def smooth_move_joints(model, data, viewer, q_target,
                       duration=2.0,
                       attached_block=None,
                       attach_offset_world=None,
                       realtime=True):

    q_start = data.qpos[:6].copy()
    steps = max(1, int(duration / model.opt.timestep))

    for step in range(steps):
        alpha = (step + 1) / steps
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

        q_cmd = (1.0 - alpha) * q_start + alpha * q_target
        data.ctrl[:6] = q_cmd

        update_gripper_visual(model, data)

        if attached_block is not None:
            tip_world = get_gripper_tip_world(model, data)
            set_block_pose(model, data, attached_block, tip_world + attach_offset_world)

        mujoco.mj_step(model, data)
        sync_viewer(viewer, model, data, realtime=realtime)


# ============================================================
# UDP input
# ============================================================

def wait_for_human_target_udp(port=UDP_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))

    print("")
    print("========================================")
    print(f"[UDP] Waiting for human target on port {port}")
    print(f"[UDP] Valid targets: {sorted(VALID_TARGETS)}")
    print("========================================")

    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode("utf-8").strip().lower()

        print(f"[UDP] Received from {addr}: {message}")

        if message in VALID_TARGETS:
            print(f"[UDP] Confirmed human target: {message}")
            sock.close()
            return message

        print(f"[UDP] Ignored invalid message: {message}")


# ============================================================
# Decision
# ============================================================

def choose_robot_target(human_target: str, strategy="closest_to_trained_pick") -> str:
    candidates = [c for c in BLOCK_COLORS if c != human_target]

    if len(candidates) == 0:
        raise RuntimeError("No available block after excluding human target.")

    if strategy == "fixed_order":
        return candidates[0]

    trained_pick_point = GRID_POINTS[4][:2]

    return min(
        candidates,
        key=lambda c: np.linalg.norm(BLOCKS[c][:2] - trained_pick_point)
    )


# ============================================================
# Pick and place
# ============================================================

def run_pick_place(model, data, viewer,
                   human_target: str,
                   strategy="closest_to_trained_pick",
                   realtime=True) -> PickPlaceResult:

    start_time = time.time()

    robot_target = choose_robot_target(human_target, strategy=strategy)
    conflict = robot_target == human_target

    print("")
    print("========================================")
    print("Collaboration decision")
    print("Human target:", human_target)
    print("Robot target:", robot_target)
    print("Conflict:", conflict)
    print("========================================")

    reset_blocks(model, data)
    set_robot_qpos(model, data, READY_Q)
    update_gripper_visual(model, data)
    hold(model, data, viewer, duration=0.8, realtime=realtime)

    print("[1] Move to mid pose")
    smooth_move_joints(model, data, viewer, MID_Q, duration=1.2, realtime=realtime)

    print("[2] Move to pick approach")
    smooth_move_joints(model, data, viewer, PICK_APPROACH_Q, duration=1.6, realtime=realtime)

    print("[3] Move down to grasp")
    smooth_move_joints(model, data, viewer, PICK_DOWN_Q, duration=1.0, realtime=realtime)
    hold(model, data, viewer, duration=0.25, realtime=realtime)

    print("[4] Attach target block")
    attached_block = robot_target

    tip_world = get_gripper_tip_world(model, data)
    block_world = get_block_pos(model, data, robot_target)

    grasp_distance = np.linalg.norm(tip_world[:2] - block_world[:2])

    print("Vertical gripper tip:", np.round(tip_world, 3))
    print("Block pos:", np.round(block_world, 3))
    print("Grasp XY distance:", round(float(grasp_distance), 4))

    if grasp_distance > GRASP_DISTANCE_THRESHOLD:
        print(
            f"[WARN] Computed gripper-tip distance is {grasp_distance:.3f} m, "
            f"larger than threshold {GRASP_DISTANCE_THRESHOLD:.3f} m. "
            "Continuing with snap attach for demo stability."
        )

    attach_offset_world = block_world - tip_world

    print("[5] Lift block")
    smooth_move_joints(
        model,
        data,
        viewer,
        PICK_APPROACH_Q,
        duration=1.2,
        attached_block=attached_block,
        attach_offset_world=attach_offset_world,
        realtime=realtime
    )

    print("[6] Move to place approach")
    smooth_move_joints(
        model,
        data,
        viewer,
        MID_Q,
        duration=1.0,
        attached_block=attached_block,
        attach_offset_world=attach_offset_world,
        realtime=realtime
    )

    smooth_move_joints(
        model,
        data,
        viewer,
        PLACE_APPROACH_Q,
        duration=1.6,
        attached_block=attached_block,
        attach_offset_world=attach_offset_world,
        realtime=realtime
    )

    print("[7] Move down to release")
    smooth_move_joints(
        model,
        data,
        viewer,
        PLACE_DOWN_Q,
        duration=1.0,
        attached_block=attached_block,
        attach_offset_world=attach_offset_world,
        realtime=realtime
    )

    print("[8] Release block at designated place position")
    place_pos = PLACE_POSITIONS[robot_target].copy()
    place_pos[2] = BLOCK_CENTER_Z

    set_block_pose(model, data, robot_target, place_pos)
    mujoco.mj_forward(model, data)
    update_gripper_visual(model, data)
    hold(model, data, viewer, duration=0.5, realtime=realtime)

    print("[9] Return to ready")
    smooth_move_joints(model, data, viewer, PLACE_APPROACH_Q, duration=0.8, realtime=realtime)
    smooth_move_joints(model, data, viewer, MID_Q, duration=1.0, realtime=realtime)
    smooth_move_joints(model, data, viewer, READY_Q, duration=1.2, realtime=realtime)
    hold(model, data, viewer, duration=0.5, realtime=realtime)

    final_pos = get_block_pos(model, data, robot_target)
    xy_error = np.linalg.norm(final_pos[:2] - PLACE_POSITIONS[robot_target][:2])

    success = (xy_error < 0.06) and (not conflict)
    duration = time.time() - start_time

    print("")
    print("========================================")
    print("Task result")
    print("Success:", success)
    print("Human target:", human_target)
    print("Robot target:", robot_target)
    print("Final block position:", np.round(final_pos, 3))
    print("Target place position:", np.round(PLACE_POSITIONS[robot_target], 3))
    print("XY error:", round(float(xy_error), 4))
    print("Duration:", round(duration, 2), "s")
    print("========================================")

    return PickPlaceResult(
        human_target=human_target,
        robot_target=robot_target,
        success=success,
        conflict=conflict,
        duration=duration,
    )


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UR10e MuJoCo collaborative pick-and-place using generated waypoints."
    )

    parser.add_argument("--manual", type=str, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fixed-grid", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--strategy",
        type=str,
        default="closest_to_trained_pick",
        choices=["closest_to_trained_pick", "fixed_order"],
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not os.path.exists(UR10E_XML):
        raise FileNotFoundError(f"Cannot find UR10e XML: {UR10E_XML}")

    load_waypoints()
    randomize_blocks_in_grid(seed=args.seed, fixed=args.fixed_grid)

    scene_xml = create_scene_xml()

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".xml",
        delete=False,
        dir=UR10E_MODEL_DIR
    ) as f:
        f.write(scene_xml)
        scene_path = f.name

    print("Temporary MuJoCo scene:", scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Model loaded successfully.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    reset_blocks(model, data)
    set_robot_qpos(model, data, READY_Q)
    update_gripper_visual(model, data)
    mujoco.mj_forward(model, data)

    if args.manual is not None:
        human_target = args.manual.strip().lower()

        if human_target not in VALID_TARGETS:
            raise ValueError(
                f"Invalid manual target: {human_target}. "
                f"Valid targets: {sorted(VALID_TARGETS)}"
            )

        print(f"[MANUAL] Human target set to: {human_target}")
    else:
        human_target = wait_for_human_target_udp(port=UDP_PORT)

    realtime = not args.fast

    if args.no_viewer:
        result = run_pick_place(
            model,
            data,
            viewer=None,
            human_target=human_target,
            strategy=args.strategy,
            realtime=False
        )
        print(result)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        result = run_pick_place(
            model,
            data,
            viewer=viewer,
            human_target=human_target,
            strategy=args.strategy,
            realtime=realtime
        )

        print("Demo finished. Close viewer to exit.")

        while viewer.is_running():
            data.ctrl[:6] = data.qpos[:6]
            update_gripper_visual(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
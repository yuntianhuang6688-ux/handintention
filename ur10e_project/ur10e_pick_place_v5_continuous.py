import os
import time
import socket
import argparse
import tempfile
import select

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


# ============================================================
# Basic scene parameters
# ============================================================

TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.06

BLOCK_SIZE = 0.04
BLOCK_CENTER_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.001

END_EFFECTOR_SITE = "attachment_site"

UDP_PORT = 5005

BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]

COLOR_RGBA = {
    "red": "1.0 0.05 0.05 1",
    "blue": "0.05 0.25 1.0 1",
    "green": "0.05 0.8 0.15 1",
    "yellow": "1.0 0.85 0.05 1",
    "orange": "1.0 0.45 0.05 1",
}

COLOR_RGBA_TRANSPARENT = {
    "red": "1.0 0.05 0.05 0.35",
    "blue": "0.05 0.25 1.0 0.35",
    "green": "0.05 0.8 0.15 0.35",
    "yellow": "1.0 0.85 0.05 0.35",
    "orange": "1.0 0.45 0.05 0.35",
}

PICK_AREA_CENTER = np.array([-0.62, -0.18, TABLE_TOP_Z + 0.004], dtype=np.float64)
PLACE_AREA_CENTER = np.array([-0.62, 0.28, TABLE_TOP_Z + 0.004], dtype=np.float64)

GRID_POINTS = [
    np.array([-0.72, -0.28, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.62, -0.28, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.52, -0.28, BLOCK_CENTER_Z], dtype=np.float64),

    np.array([-0.72, -0.18, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.62, -0.18, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.52, -0.18, BLOCK_CENTER_Z], dtype=np.float64),

    np.array([-0.72, -0.08, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.62, -0.08, BLOCK_CENTER_Z], dtype=np.float64),
    np.array([-0.52, -0.08, BLOCK_CENTER_Z], dtype=np.float64),
]

BLOCK_START_POSITIONS = {
    "red": GRID_POINTS[4],
    "blue": GRID_POINTS[5],
    "green": GRID_POINTS[3],
    "yellow": GRID_POINTS[1],
    "orange": GRID_POINTS[7],
}

# Blue finished area slots.
# Both human-taken blocks and robot-placed blocks will be moved here.
PLACE_POSITIONS = {
    "red": np.array([-0.74, 0.28, BLOCK_CENTER_Z], dtype=np.float64),
    "blue": np.array([-0.68, 0.28, BLOCK_CENTER_Z], dtype=np.float64),
    "green": np.array([-0.62, 0.28, BLOCK_CENTER_Z], dtype=np.float64),
    "yellow": np.array([-0.56, 0.28, BLOCK_CENTER_Z], dtype=np.float64),
    "orange": np.array([-0.50, 0.28, BLOCK_CENTER_Z], dtype=np.float64),
}


# ============================================================
# Tuned target heights
# ============================================================

PICK_APPROACH_Z = 0.24
PICK_DOWN_Z = 0.110

PLACE_APPROACH_Z = 0.24
PLACE_DOWN_Z = 0.150

MID_POS = np.array([-0.62, 0.05, 0.34], dtype=np.float64)


# ============================================================
# Robot joint settings
# ============================================================

READY_Q = np.array([
    -0.70,
    -1.35,
     1.55,
    -1.75,
    -1.57,
     0.00,
], dtype=np.float64)

NATURAL_Q = np.array([
    -0.75,
    -1.35,
     1.55,
    -1.80,
    -1.57,
     0.00,
], dtype=np.float64)

JOINT_LOWER = np.array([-2.20, -2.15, 0.45, -2.90, -2.50, -3.14])
JOINT_UPPER = np.array([ 0.80, -0.35, 2.60, -0.25,  0.40,  3.14])


# ============================================================
# Runtime state
# ============================================================

available_blocks = set(BLOCK_COLORS)
human_finished_blocks = set()
robot_finished_blocks = set()


# ============================================================
# Scene XML
# ============================================================

def create_scene_xml() -> str:
    table_center_x = -0.45
    table_center_y = 0.02

    grid_xml = ""
    for i, p in enumerate(GRID_POINTS):
        grid_xml += f"""
        <geom name="pick_grid_{i}"
              type="box"
              pos="{p[0]} {p[1]} {TABLE_TOP_Z + 0.003}"
              size="0.025 0.025 0.003"
              rgba="0.02 0.02 0.02 1"
              contype="0"
              conaffinity="0"/>
        """

    block_xml = ""
    for color in BLOCK_COLORS:
        pos = BLOCK_START_POSITIONS[color]
        block_xml += f"""
        <body name="{color}_cube" pos="{pos[0]} {pos[1]} {pos[2]}">
            <freejoint name="{color}_cube_freejoint"/>
            <geom name="{color}_cube_geom"
                  type="box"
                  size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0}"
                  rgba="{COLOR_RGBA[color]}"
                  mass="0.05"
                  friction="1.2 0.01 0.001"
                  condim="3"/>
        </body>
        """

    place_slot_xml = ""
    for color in BLOCK_COLORS:
        pos = PLACE_POSITIONS[color]
        place_slot_xml += f"""
        <geom name="{color}_place_slot"
              type="box"
              pos="{pos[0]} {pos[1]} {TABLE_TOP_Z + 0.008}"
              size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} 0.004"
              rgba="{COLOR_RGBA_TRANSPARENT[color]}"
              contype="0"
              conaffinity="0"/>
        """

    xml = f"""
<mujoco model="ur10e_pick_place_v5_continuous">

    <include file="ur10e.xml"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <statistic center="-0.45 0.02 0.35" extent="1.8"/>

    <visual>
        <headlight diffuse="0.65 0.65 0.65"
                   ambient="0.35 0.35 0.35"
                   specular="0.1 0.1 0.1"/>
        <global azimuth="130" elevation="-25"/>
        <rgba haze="0.15 0.20 0.25 1"/>
    </visual>

    <worldbody>

        <light name="top_light"
               pos="-0.45 0.02 2.2"
               dir="0 0 -1"
               diffuse="0.85 0.85 0.85"/>

        <geom name="floor"
              type="plane"
              size="3 3 0.01"
              rgba="0.80 0.80 0.80 1"/>

        <body name="table" pos="{table_center_x} {table_center_y} {-TABLE_THICKNESS / 2.0}">
            <geom name="table_top"
                  type="box"
                  size="0.95 0.65 {TABLE_THICKNESS / 2.0}"
                  rgba="0.90 0.90 0.90 1"
                  friction="1.0 0.01 0.001"/>
        </body>

        <geom name="front_black_rail"
              type="box"
              pos="{table_center_x} {table_center_y - 0.67} -0.02"
              size="1.00 0.025 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="back_black_rail"
              type="box"
              pos="{table_center_x} {table_center_y + 0.67} -0.02"
              size="1.00 0.025 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="left_black_rail"
              type="box"
              pos="{table_center_x - 1.00} {table_center_y} -0.02"
              size="0.025 0.67 0.025"
              rgba="0.02 0.025 0.03 1"
              contype="0"
              conaffinity="0"/>

        <geom name="right_black_rail"
              type="box"
              pos="{table_center_x + 1.00} {table_center_y} -0.02"
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

        <geom name="pick_area"
              type="box"
              pos="{PICK_AREA_CENTER[0]} {PICK_AREA_CENTER[1]} {PICK_AREA_CENTER[2]}"
              size="0.22 0.22 0.004"
              rgba="0.10 0.80 0.10 0.25"
              contype="0"
              conaffinity="0"/>

        <geom name="place_area"
              type="box"
              pos="{PLACE_AREA_CENTER[0]} {PLACE_AREA_CENTER[1]} {PLACE_AREA_CENTER[2]}"
              size="0.30 0.12 0.004"
              rgba="0.02 0.28 0.95 0.45"
              contype="0"
              conaffinity="0"/>

        <geom name="marker_board"
              type="box"
              pos="{PICK_AREA_CENTER[0]} {PICK_AREA_CENTER[1]} {TABLE_TOP_Z + 0.001}"
              size="0.28 0.28 0.001"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        {grid_xml}

        {place_slot_xml}

        {block_xml}

        <camera name="overview"
                pos="0.35 -1.20 1.05"
                xyaxes="0.72 0.70 0 -0.35 0.36 0.86"/>

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


def set_robot_qpos(model, data, q):
    q = np.asarray(q, dtype=np.float64)
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def get_site_pos(model, data, site_name: str) -> np.ndarray:
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def get_ee_pos(model, data) -> np.ndarray:
    return get_site_pos(model, data, END_EFFECTOR_SITE)


def clamp_q(q):
    return np.minimum(np.maximum(q, JOINT_LOWER), JOINT_UPPER)


def set_freejoint_pose(model, data, joint_name: str, pos: np.ndarray):
    joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    qvel_addr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_addr:qpos_addr + 7] = np.array([
        pos[0], pos[1], pos[2],
        1.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)

    data.qvel[qvel_addr:qvel_addr + 6] = 0.0


def get_cube_pos(model, data, color: str) -> np.ndarray:
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube")
    return data.xpos[body_id].copy()


def set_cube_pos(model, data, color: str, pos: np.ndarray):
    p = np.asarray(pos, dtype=np.float64).copy()
    p[2] = max(p[2], BLOCK_CENTER_Z)

    set_freejoint_pose(model, data, f"{color}_cube_freejoint", p)
    mujoco.mj_forward(model, data)


def reset_all_cubes(model, data):
    for color in BLOCK_COLORS:
        set_cube_pos(model, data, color, BLOCK_START_POSITIONS[color])


def move_human_cube_to_finished_area(model, data, color: str):
    """
    Simulate that the human has taken this cube.
    Instead of hiding it, move it to its corresponding slot on the blue finished area.
    """
    finished_pos = PLACE_POSITIONS[color].copy()
    set_cube_pos(model, data, color, finished_pos)


# ============================================================
# UDP helpers
# ============================================================

def create_udp_socket(port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)

    print("")
    print("========================================")
    print(f"[UDP] Listening continuously on port {port}")
    print("[UDP] Valid messages:", ", ".join(BLOCK_COLORS))
    print("========================================")

    return sock


def poll_udp_target(sock):
    """
    Non-blocking UDP polling.
    Returns a color string or None.
    """
    readable, _, _ = select.select([sock], [], [], 0.0)

    if not readable:
        return None

    data, addr = sock.recvfrom(1024)
    msg = data.decode("utf-8").strip().lower()

    print(f"[UDP] Received from {addr}: {msg}")

    if msg in BLOCK_COLORS:
        return msg

    print("[UDP] Ignored invalid message:", msg)
    return None


# ============================================================
# Decision logic
# ============================================================

def choose_robot_target(human_target: str, strategy="first_available") -> str | None:
    candidates = [
        c for c in BLOCK_COLORS
        if c != human_target and c in available_blocks
    ]

    if len(candidates) == 0:
        return None

    if strategy == "first_available":
        return candidates[0]

    if strategy == "nearest_to_center":
        center = PICK_AREA_CENTER[:2]
        return min(
            candidates,
            key=lambda c: np.linalg.norm(BLOCK_START_POSITIONS[c][:2] - center)
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# Jacobian IK
# ============================================================

def solve_ik_position(
    model,
    data,
    target_pos,
    seed_q,
    label,
    q_ref=None,
    max_iters=300,
    pos_tol=0.025,
):
    if q_ref is None:
        q_ref = NATURAL_Q.copy()

    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)

    q = clamp_q(seed_q.copy())
    set_robot_qpos(model, data, q)

    best_q = q.copy()
    best_err = 999.0

    damping = 1e-3
    posture_weight = 0.045

    print("")
    print("========================================")
    print(f"Solving IK: {label}")
    print("Target:", np.round(target_pos, 3))
    print("Seed:", np.round(seed_q, 3))

    for it in range(max_iters):
        set_robot_qpos(model, data, q)

        ee_pos = get_ee_pos(model, data)
        err = target_pos - ee_pos
        err_norm = float(np.linalg.norm(err))

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()

        if err_norm <= pos_tol:
            break

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J_pos = jacp[:, :6]

        J_posture = posture_weight * np.eye(6)
        e_posture = posture_weight * (q_ref - q)

        J = np.vstack([J_pos, J_posture])
        e = np.concatenate([err, e_posture])

        A = J.T @ J + damping * np.eye(6)
        b = J.T @ e

        try:
            dq = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(A, b, rcond=None)[0]

        max_step = 0.040
        dq_norm = np.linalg.norm(dq)

        if dq_norm > max_step:
            dq = dq / dq_norm * max_step

        q = clamp_q(q + dq)

    set_robot_qpos(model, data, best_q)
    final_ee = get_ee_pos(model, data)
    final_err = np.linalg.norm(target_pos - final_ee)

    print("Result q:", np.round(best_q, 3))
    print("Final ee:", np.round(final_ee, 3))
    print("Final error:", round(float(final_err), 4))
    print("Iterations:", it + 1)
    print("========================================")

    return best_q


def build_waypoints(model, data, robot_target: str, current_q: np.ndarray):
    pick_pos = BLOCK_START_POSITIONS[robot_target]
    place_pos = PLACE_POSITIONS[robot_target]

    pick_approach_pos = np.array([pick_pos[0], pick_pos[1], PICK_APPROACH_Z], dtype=np.float64)
    pick_down_pos = np.array([pick_pos[0], pick_pos[1], PICK_DOWN_Z], dtype=np.float64)

    place_approach_pos = np.array([place_pos[0], place_pos[1], PLACE_APPROACH_Z], dtype=np.float64)
    place_down_pos = np.array([place_pos[0], place_pos[1], PLACE_DOWN_Z], dtype=np.float64)

    pick_approach_q = solve_ik_position(
        model=model,
        data=data,
        target_pos=pick_approach_pos,
        seed_q=current_q,
        label=f"pick_approach_{robot_target}",
        q_ref=NATURAL_Q,
    )

    pick_down_q = solve_ik_position(
        model=model,
        data=data,
        target_pos=pick_down_pos,
        seed_q=pick_approach_q,
        label=f"pick_down_{robot_target}",
        q_ref=pick_approach_q,
    )

    mid_q = solve_ik_position(
        model=model,
        data=data,
        target_pos=MID_POS,
        seed_q=pick_approach_q,
        label="mid",
        q_ref=NATURAL_Q,
    )

    place_approach_q = solve_ik_position(
        model=model,
        data=data,
        target_pos=place_approach_pos,
        seed_q=mid_q,
        label=f"place_approach_{robot_target}",
        q_ref=NATURAL_Q,
    )

    place_down_q = solve_ik_position(
        model=model,
        data=data,
        target_pos=place_down_pos,
        seed_q=place_approach_q,
        label=f"place_down_{robot_target}",
        q_ref=place_approach_q,
    )

    return {
        "current": current_q,
        "pick_approach": pick_approach_q,
        "pick_down": pick_down_q,
        "mid": mid_q,
        "place_approach": place_approach_q,
        "place_down": place_down_q,
        "ready": READY_Q.copy(),
    }


# ============================================================
# Motion
# ============================================================

def sync_viewer(viewer, model, data, realtime=True):
    if viewer is not None and viewer.is_running():
        viewer.sync()

    if realtime:
        time.sleep(model.opt.timestep)


def hold_pose(
    model,
    data,
    viewer,
    duration=0.4,
    realtime=True,
    attached=False,
    attach_offset=None,
    color=None,
):
    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        data.ctrl[:6] = data.qpos[:6]

        if attached and color is not None:
            ee_pos = get_ee_pos(model, data)
            set_cube_pos(model, data, color, ee_pos + attach_offset)

        mujoco.mj_step(model, data)
        sync_viewer(viewer, model, data, realtime=realtime)


def smooth_move_joints(
    model,
    data,
    viewer,
    q_target,
    duration,
    realtime=True,
    attached=False,
    attach_offset=None,
    color=None,
):
    q_start = data.qpos[:6].copy()
    steps = max(1, int(duration / model.opt.timestep))

    for i in range(steps):
        alpha = (i + 1) / steps
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

        q_cmd = (1.0 - alpha) * q_start + alpha * q_target
        data.ctrl[:6] = q_cmd

        if attached and color is not None:
            ee_pos = get_ee_pos(model, data)
            set_cube_pos(model, data, color, ee_pos + attach_offset)

        mujoco.mj_step(model, data)
        sync_viewer(viewer, model, data, realtime=realtime)


def run_one_pick_place(
    model,
    data,
    viewer,
    human_target,
    robot_target,
    strategy,
    realtime=True,
):
    global available_blocks
    global human_finished_blocks
    global robot_finished_blocks

    print("")
    print("========================================")
    print("New collaboration cycle")
    print("Human target:", human_target)
    print("Robot target:", robot_target)
    print("Conflict:", human_target == robot_target)
    print("Available before:", sorted(available_blocks))
    print("========================================")

    # Human target is considered taken by human.
    # Move it directly to the blue finished area.
    if human_target in available_blocks:
        available_blocks.remove(human_target)
        human_finished_blocks.add(human_target)
        move_human_cube_to_finished_area(model, data, human_target)
        print(f"[STATE] Human took {human_target}. Moved to blue finished area.")

    current_q = data.qpos[:6].copy()

    print("")
    print("Building IK waypoints for robot target:", robot_target)
    waypoints = build_waypoints(
        model=model,
        data=data,
        robot_target=robot_target,
        current_q=current_q,
    )

    print("[1] Move to pick approach")
    smooth_move_joints(
        model, data, viewer,
        waypoints["pick_approach"],
        duration=2.0,
        realtime=realtime,
    )

    print("[2] Move down to pick")
    smooth_move_joints(
        model, data, viewer,
        waypoints["pick_down"],
        duration=1.2,
        realtime=realtime,
    )

    hold_pose(model, data, viewer, duration=0.3, realtime=realtime)

    print("[3] Attach selected cube")
    ee_pos = get_ee_pos(model, data)
    cube_pos = get_cube_pos(model, data, robot_target)
    attach_offset = cube_pos - ee_pos

    print("EE pos:", np.round(ee_pos, 3))
    print(f"{robot_target} cube pos:", np.round(cube_pos, 3))
    print("Attach offset:", np.round(attach_offset, 3))

    print("[4] Lift cube")
    smooth_move_joints(
        model, data, viewer,
        waypoints["pick_approach"],
        duration=1.2,
        realtime=realtime,
        attached=True,
        attach_offset=attach_offset,
        color=robot_target,
    )

    print("[5] Move through mid")
    smooth_move_joints(
        model, data, viewer,
        waypoints["mid"],
        duration=1.5,
        realtime=realtime,
        attached=True,
        attach_offset=attach_offset,
        color=robot_target,
    )

    print("[6] Move to place approach")
    smooth_move_joints(
        model, data, viewer,
        waypoints["place_approach"],
        duration=2.0,
        realtime=realtime,
        attached=True,
        attach_offset=attach_offset,
        color=robot_target,
    )

    print("[7] Move down to place")
    smooth_move_joints(
        model, data, viewer,
        waypoints["place_down"],
        duration=1.2,
        realtime=realtime,
        attached=True,
        attach_offset=attach_offset,
        color=robot_target,
    )

    hold_pose(
        model,
        data,
        viewer,
        duration=0.3,
        realtime=realtime,
        attached=True,
        attach_offset=attach_offset,
        color=robot_target,
    )

    print("[8] Release robot cube on blue finished area")
    set_cube_pos(model, data, robot_target, PLACE_POSITIONS[robot_target])
    hold_pose(model, data, viewer, duration=0.5, realtime=realtime)

    print("[9] Return to ready")
    smooth_move_joints(
        model, data, viewer,
        waypoints["place_approach"],
        duration=1.0,
        realtime=realtime,
    )

    smooth_move_joints(
        model, data, viewer,
        waypoints["mid"],
        duration=1.2,
        realtime=realtime,
    )

    smooth_move_joints(
        model, data, viewer,
        waypoints["ready"],
        duration=1.8,
        realtime=realtime,
    )

    if robot_target in available_blocks:
        available_blocks.remove(robot_target)

    robot_finished_blocks.add(robot_target)

    final_cube = get_cube_pos(model, data, robot_target)
    target_place = PLACE_POSITIONS[robot_target]
    place_error = np.linalg.norm(final_cube[:2] - target_place[:2])

    print("")
    print("========================================")
    print("Cycle finished")
    print("Human target:", human_target)
    print("Robot target:", robot_target)
    print("Final robot cube pos:", np.round(final_cube, 3))
    print("Target place:", np.round(target_place, 3))
    print("XY place error:", round(float(place_error), 4))
    print("Available after:", sorted(available_blocks))
    print("Human finished:", sorted(human_finished_blocks))
    print("Robot finished:", sorted(robot_finished_blocks))
    print("========================================")


def continuous_loop(model, data, viewer, udp_sock, strategy, realtime=True):
    print("")
    print("========================================")
    print("Continuous collaboration mode started")
    print("Send UDP color messages to trigger each cycle.")
    print("Example:")
    print("  echo -n red | nc -u -w1 127.0.0.1 5005")
    print("========================================")

    set_robot_qpos(model, data, READY_Q)
    reset_all_cubes(model, data)

    while viewer is None or viewer.is_running():
        human_target = poll_udp_target(udp_sock)

        if human_target is None:
            data.ctrl[:6] = data.qpos[:6]
            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            if realtime:
                time.sleep(model.opt.timestep)

            continue

        if human_target not in available_blocks:
            print(f"[STATE] Ignored {human_target}: not available anymore.")
            continue

        robot_target = choose_robot_target(
            human_target=human_target,
            strategy=strategy,
        )

        if robot_target is None:
            print("[STATE] No available robot target. Task completed or no blocks left.")
            # Still move the human-selected block to blue area if it is available.
            if human_target in available_blocks:
                available_blocks.remove(human_target)
                human_finished_blocks.add(human_target)
                move_human_cube_to_finished_area(model, data, human_target)
            continue

        run_one_pick_place(
            model=model,
            data=data,
            viewer=viewer,
            human_target=human_target,
            robot_target=robot_target,
            strategy=strategy,
            realtime=realtime,
        )

        if len(available_blocks) == 0:
            print("")
            print("========================================")
            print("All blocks have been used.")
            print("Continuous loop is still running. Close viewer or Ctrl+C to exit.")
            print("========================================")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UR10e MuJoCo v5 continuous UDP collaboration demo."
    )

    parser.add_argument(
        "--udp-port",
        type=int,
        default=UDP_PORT,
        help="UDP port for camera intention input."
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="first_available",
        choices=["first_available", "nearest_to_center"],
        help="Robot target selection strategy."
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run without realtime sleep."
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without MuJoCo viewer."
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not os.path.exists(UR10E_XML):
        raise FileNotFoundError(
            f"Cannot find UR10e XML:\n{UR10E_XML}\n"
            "Check your mujoco_menagerie path."
        )

    scene_xml = create_scene_xml()

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".xml",
        delete=False,
        dir=UR10E_MODEL_DIR
    ) as f:
        f.write(scene_xml)
        scene_path = f.name

    print("Temporary v5 continuous scene:", scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Model loaded.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    udp_sock = create_udp_socket(args.udp_port)

    realtime = not args.fast

    if args.no_viewer:
        continuous_loop(
            model=model,
            data=data,
            viewer=None,
            udp_sock=udp_sock,
            strategy=args.strategy,
            realtime=False,
        )
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        continuous_loop(
            model=model,
            data=data,
            viewer=viewer,
            udp_sock=udp_sock,
            strategy=args.strategy,
            realtime=realtime,
        )


if __name__ == "__main__":
    main()
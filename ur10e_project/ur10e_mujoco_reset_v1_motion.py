import os
import time
import argparse
import tempfile

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
# Basic parameters
# ============================================================

TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.06

BLOCK_SIZE = 0.04
BLOCK_CENTER_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.001

END_EFFECTOR_SITE = "attachment_site"

BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]

COLOR_RGBA = {
    "red": "1.0 0.05 0.05 1",
    "blue": "0.05 0.25 1.0 1",
    "green": "0.05 0.8 0.15 1",
    "yellow": "1.0 0.85 0.05 1",
    "orange": "1.0 0.45 0.05 1",
}


# ============================================================
# Clean layout
# ============================================================

GRID_POINTS = [
    np.array([-0.72, -0.25, BLOCK_CENTER_Z]),
    np.array([-0.62, -0.25, BLOCK_CENTER_Z]),
    np.array([-0.52, -0.25, BLOCK_CENTER_Z]),

    np.array([-0.72, -0.15, BLOCK_CENTER_Z]),
    np.array([-0.62, -0.15, BLOCK_CENTER_Z]),
    np.array([-0.52, -0.15, BLOCK_CENTER_Z]),

    np.array([-0.72, -0.05, BLOCK_CENTER_Z]),
    np.array([-0.62, -0.05, BLOCK_CENTER_Z]),
    np.array([-0.52, -0.05, BLOCK_CENTER_Z]),
]

BLOCK_POSITIONS = {
    "red": GRID_POINTS[0],
    "blue": GRID_POINTS[1],
    "green": GRID_POINTS[2],
    "yellow": GRID_POINTS[3],
    "orange": GRID_POINTS[4],
}

PICK_CENTER = GRID_POINTS[4].copy()
PLACE_CENTER = np.array([-0.62, 0.30, TABLE_TOP_Z + 0.006])


# ============================================================
# IK target positions
# ============================================================
# These are attachment_site target positions, not block positions.
# We keep them above the table so the motion is natural.

PICK_HOVER_TARGET = np.array([PICK_CENTER[0], PICK_CENTER[1], 0.36])
PICK_LOWER_TARGET = np.array([PICK_CENTER[0], PICK_CENTER[1], 0.23])

PLACE_HOVER_TARGET = np.array([PLACE_CENTER[0], PLACE_CENTER[1], 0.36])
PLACE_LOWER_TARGET = np.array([PLACE_CENTER[0], PLACE_CENTER[1], 0.23])

MID_TARGET = np.array([-0.62, 0.05, 0.45])


# ============================================================
# Natural joint settings
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

JOINT_LOWER = np.array([-2.20, -2.10, 0.60, -2.80, -2.40, -3.14])
JOINT_UPPER = np.array([ 1.00, -0.45, 2.50, -0.40,  0.30,  3.14])


# ============================================================
# Scene XML
# ============================================================

def create_scene_xml() -> str:
    block_xml = ""
    for color, pos in BLOCK_POSITIONS.items():
        block_xml += f"""
        <body name="{color}_block" pos="{pos[0]} {pos[1]} {pos[2]}">
            <freejoint name="{color}_freejoint"/>
            <geom name="{color}_geom"
                  type="box"
                  size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0}"
                  rgba="{COLOR_RGBA[color]}"
                  mass="0.05"
                  friction="1.2 0.01 0.001"
                  condim="3"/>
        </body>
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

    table_center_x = -0.50
    table_center_y = 0.02

    xml = f"""
<mujoco model="ur10e_clean_motion_v1c_jacobian_ik">

    <include file="ur10e.xml"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <statistic center="-0.45 0.0 0.35" extent="1.7"/>

    <visual>
        <headlight diffuse="0.65 0.65 0.65"
                   ambient="0.35 0.35 0.35"
                   specular="0.1 0.1 0.1"/>
        <global azimuth="130" elevation="-25"/>
        <rgba haze="0.15 0.20 0.25 1"/>
    </visual>

    <worldbody>

        <light name="top_light"
               pos="-0.45 0.0 2.2"
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

        <geom name="blue_baseplate"
              type="box"
              pos="{PLACE_CENTER[0]} {PLACE_CENTER[1]} {PLACE_CENTER[2]}"
              size="0.30 0.11 0.006"
              rgba="0.02 0.28 0.95 0.75"
              contype="0"
              conaffinity="0"/>

        <geom name="green_place_area"
              type="box"
              pos="{PLACE_CENTER[0]} {PLACE_CENTER[1]} {TABLE_TOP_Z + 0.013}"
              size="0.34 0.13 0.003"
              rgba="0.1 0.8 0.1 0.20"
              contype="0"
              conaffinity="0"/>

        <geom name="marker_board"
              type="box"
              pos="-0.62 -0.15 {TABLE_TOP_Z + 0.001}"
              size="0.25 0.23 0.001"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        {grid_marker_xml}

        {block_xml}

        <!-- IK target markers -->
        <site name="pick_hover_target"
              pos="{PICK_HOVER_TARGET[0]} {PICK_HOVER_TARGET[1]} {PICK_HOVER_TARGET[2]}"
              size="0.018"
              rgba="1 0 1 1"/>

        <site name="pick_lower_target"
              pos="{PICK_LOWER_TARGET[0]} {PICK_LOWER_TARGET[1]} {PICK_LOWER_TARGET[2]}"
              size="0.014"
              rgba="1 0.4 1 1"/>

        <site name="place_hover_target"
              pos="{PLACE_HOVER_TARGET[0]} {PLACE_HOVER_TARGET[1]} {PLACE_HOVER_TARGET[2]}"
              size="0.018"
              rgba="0 1 0 1"/>

        <site name="place_lower_target"
              pos="{PLACE_LOWER_TARGET[0]} {PLACE_LOWER_TARGET[1]} {PLACE_LOWER_TARGET[2]}"
              size="0.014"
              rgba="0.4 1 0.4 1"/>

        <site name="mid_target"
              pos="{MID_TARGET[0]} {MID_TARGET[1]} {MID_TARGET[2]}"
              size="0.014"
              rgba="1 1 0 1"/>

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


def get_site_pos(model, data, site_name: str) -> np.ndarray:
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def get_site_rot(model, data, site_name: str) -> np.ndarray:
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def set_robot_qpos(model, data, q):
    q = np.asarray(q, dtype=np.float64)
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def get_ee_pos(model, data):
    return get_site_pos(model, data, END_EFFECTOR_SITE)


def get_ee_rot(model, data):
    return get_site_rot(model, data, END_EFFECTOR_SITE)


def clamp_q(q):
    return np.minimum(np.maximum(q, JOINT_LOWER), JOINT_UPPER)


# ============================================================
# Jacobian IK
# ============================================================

def solve_ik(
    model,
    data,
    target_pos,
    seed_q,
    label,
    q_ref=None,
    max_iters=250,
    pos_tol=0.015,
):
    """
    Damped least-squares Jacobian IK for UR10e attachment_site.

    Main goal:
        attachment_site position -> target_pos

    Secondary goals:
        keep posture close to q_ref / NATURAL_Q
        keep wrist_2 reasonable
    """

    if q_ref is None:
        q_ref = NATURAL_Q.copy()

    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)

    q = clamp_q(seed_q.copy())
    set_robot_qpos(model, data, q)

    damping = 1e-3
    posture_weight = 0.035

    best_q = q.copy()
    best_err = 999.0

    print("")
    print("========================================")
    print(f"Solving IK for {label}")
    print("Target:", np.round(target_pos, 3))
    print("Seed q:", np.round(seed_q, 3))

    for it in range(max_iters):
        set_robot_qpos(model, data, q)

        ee_pos = get_ee_pos(model, data)
        pos_err = target_pos - ee_pos
        err_norm = float(np.linalg.norm(pos_err))

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()

        if err_norm < pos_tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        # We only use first 6 DoFs because UR10e has 6 actuated joints.
        J_pos = jacp[:, :6]

        # Posture regularisation row:
        # keep close to q_ref and avoid extreme poses.
        J_posture = posture_weight * np.eye(6)
        e_posture = posture_weight * (q_ref - q)

        J = np.vstack([
            J_pos,
            J_posture,
        ])

        e = np.concatenate([
            pos_err,
            e_posture,
        ])

        # Damped least squares
        A = J.T @ J + damping * np.eye(6)
        b = J.T @ e

        try:
            dq = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(A, b, rcond=None)[0]

        # Limit step size to avoid sudden weird jumps.
        max_step = 0.045
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq / dq_norm * max_step

        q = q + dq
        q = clamp_q(q)

    set_robot_qpos(model, data, best_q)
    final_ee = get_ee_pos(model, data)
    final_err = np.linalg.norm(target_pos - final_ee)

    print("Result q:", np.round(best_q, 3))
    print("Final ee:", np.round(final_ee, 3))
    print("Final error:", round(float(final_err), 4))
    print("Iterations:", it + 1)
    print("========================================")

    return best_q


def build_trajectory_waypoints(model, data):
    """
    Build a clean motion trajectory with Jacobian IK.
    Each point is solved from the previous point, so the motion stays continuous.
    """

    ready_q = READY_Q.copy()
    set_robot_qpos(model, data, ready_q)

    pick_hover_q = solve_ik(
        model=model,
        data=data,
        target_pos=PICK_HOVER_TARGET,
        seed_q=ready_q,
        label="pick_hover",
        q_ref=NATURAL_Q,
    )

    pick_lower_q = solve_ik(
        model=model,
        data=data,
        target_pos=PICK_LOWER_TARGET,
        seed_q=pick_hover_q,
        label="pick_lower",
        q_ref=pick_hover_q,
    )

    mid_q = solve_ik(
        model=model,
        data=data,
        target_pos=MID_TARGET,
        seed_q=pick_hover_q,
        label="mid",
        q_ref=NATURAL_Q,
    )

    place_hover_q = solve_ik(
        model=model,
        data=data,
        target_pos=PLACE_HOVER_TARGET,
        seed_q=mid_q,
        label="place_hover",
        q_ref=NATURAL_Q,
    )

    place_lower_q = solve_ik(
        model=model,
        data=data,
        target_pos=PLACE_LOWER_TARGET,
        seed_q=place_hover_q,
        label="place_lower",
        q_ref=place_hover_q,
    )

    trajectory = [
        ("ready", ready_q, 0.8),
        ("move_to_pick_hover", pick_hover_q, 2.0),
        ("lower_near_pick_area", pick_lower_q, 1.2),
        ("back_to_pick_hover", pick_hover_q, 1.1),
        ("move_to_mid", mid_q, 1.5),
        ("move_to_place_hover", place_hover_q, 2.0),
        ("lower_near_place_area", place_lower_q, 1.2),
        ("back_to_place_hover", place_hover_q, 1.1),
        ("move_to_mid", mid_q, 1.5),
        ("return_ready", ready_q, 1.8),
    ]

    return trajectory


# ============================================================
# Motion
# ============================================================

def smooth_move_joints(
    model,
    data,
    viewer,
    q_target,
    duration,
    realtime=True,
):
    q_start = data.qpos[:6].copy()
    steps = max(1, int(duration / model.opt.timestep))

    for i in range(steps):
        alpha = (i + 1) / steps
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

        q_cmd = (1.0 - alpha) * q_start + alpha * q_target

        data.ctrl[:6] = q_cmd
        mujoco.mj_step(model, data)

        if viewer is not None and viewer.is_running():
            viewer.sync()

        if realtime:
            time.sleep(model.opt.timestep)


def hold_pose(
    model,
    data,
    viewer,
    duration=0.5,
    realtime=True,
):
    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        data.ctrl[:6] = data.qpos[:6]
        mujoco.mj_step(model, data)

        if viewer is not None and viewer.is_running():
            viewer.sync()

        if realtime:
            time.sleep(model.opt.timestep)


def run_motion_demo(model, data, viewer, trajectory, cycles=1, realtime=True):
    print("")
    print("========================================")
    print("Running v1c Jacobian IK motion demo")
    print("No grasping. No block movement. No fake attach.")
    print("========================================")

    set_robot_qpos(model, data, trajectory[0][1])
    hold_pose(model, data, viewer, duration=0.8, realtime=realtime)

    for cycle in range(cycles):
        print("")
        print(f"Cycle {cycle + 1}/{cycles}")

        for name, q_target, duration in trajectory:
            print(f"  -> {name}")
            smooth_move_joints(
                model=model,
                data=data,
                viewer=viewer,
                q_target=q_target,
                duration=duration,
                realtime=realtime,
            )
            hold_pose(
                model=model,
                data=data,
                viewer=viewer,
                duration=0.25,
                realtime=realtime,
            )

    print("")
    print("v1c motion demo finished.")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UR10e MuJoCo clean v1c Jacobian IK motion demo."
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of trajectory cycles."
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
            "Check mujoco_menagerie path."
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

    print("Temporary clean v1c scene:", scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Model loaded.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    print("")
    print("Building Jacobian IK trajectory...")
    trajectory = build_trajectory_waypoints(model, data)

    realtime = not args.fast

    if args.no_viewer:
        run_motion_demo(
            model=model,
            data=data,
            viewer=None,
            trajectory=trajectory,
            cycles=args.cycles,
            realtime=False,
        )
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run_motion_demo(
            model=model,
            data=data,
            viewer=viewer,
            trajectory=trajectory,
            cycles=args.cycles,
            realtime=realtime,
        )

        print("Close the viewer to exit.")

        while viewer.is_running():
            data.ctrl[:6] = data.qpos[:6]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
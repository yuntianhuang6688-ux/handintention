import os
import time
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

BLOCK_COLORS = ["red", "blue", "green", "yellow", "orange"]

COLOR_RGBA = {
    "red": "1.0 0.05 0.05 1",
    "blue": "0.05 0.25 1.0 1",
    "green": "0.05 0.8 0.15 1",
    "yellow": "1.0 0.85 0.05 1",
    "orange": "1.0 0.45 0.05 1",
}


# ============================================================
# New clean layout
# ============================================================
# Robot base stays at world origin.
# Blocks are placed in front-left workspace, not too close to the base.
# This should avoid the previous folded, ugly arm posture later.

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

PLACE_CENTER = np.array([-0.62, 0.30, TABLE_TOP_Z + 0.006])


# ============================================================
# Robot initial pose
# ============================================================
# This is only a natural-looking standby pose.
# No picking yet.

READY_Q = np.array([
    -0.70,   # shoulder_pan
    -1.30,   # shoulder_lift
     1.55,   # elbow
    -1.80,   # wrist_1
    -1.57,   # wrist_2
     0.00,   # wrist_3
], dtype=np.float64)


# ============================================================
# XML scene creation
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
<mujoco model="ur10e_clean_scene_v0">

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

        <!-- Main tabletop -->
        <body name="table" pos="{table_center_x} {table_center_y} {-TABLE_THICKNESS / 2.0}">
            <geom name="table_top"
                  type="box"
                  size="0.95 0.65 {TABLE_THICKNESS / 2.0}"
                  rgba="0.90 0.90 0.90 1"
                  friction="1.0 0.01 0.001"/>
        </body>

        <!-- Black frame rails, like the real table -->
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

        <!-- Robot base plate -->
        <geom name="robot_base_plate"
              type="box"
              pos="0.0 0.0 {TABLE_TOP_Z + 0.005}"
              size="0.15 0.15 0.005"
              rgba="0.12 0.12 0.15 1"
              contype="0"
              conaffinity="0"/>

        <!-- Blue baseplate / placing area -->
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

        <!-- White marker board -->
        <geom name="marker_board"
              type="box"
              pos="-0.62 -0.15 {TABLE_TOP_Z + 0.001}"
              size="0.25 0.23 0.001"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        {grid_marker_xml}

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

def set_robot_qpos(model, data, q):
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def main():
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

    print("Temporary clean scene:", scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Model loaded.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    set_robot_qpos(model, data, READY_Q)

    print("")
    print("Clean scene v0 loaded.")
    print("This version only checks layout.")
    print("No IK, no fake grasp, no snap attach, no moving blocks.")
    print("Close the viewer to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:6] = data.qpos[:6]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
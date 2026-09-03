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
# Basic scene parameters
# ============================================================

TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.06

BLOCK_SIZE = 0.04
BLOCK_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.001

# Robot base is at world origin.
# Put the pick area in front-left of the robot, not too close to the base.
PICK_AREA_CENTER = np.array([-0.62, -0.18, TABLE_TOP_Z + 0.004])
PLACE_AREA_CENTER = np.array([-0.62, 0.28, TABLE_TOP_Z + 0.004])

# One cube to pick.
CUBE_POS = np.array([-0.62, -0.18, BLOCK_Z])

# A natural standby pose for UR10e.
READY_Q = np.array([
    -0.70,   # shoulder_pan
    -1.35,   # shoulder_lift
     1.55,   # elbow
    -1.75,   # wrist_1
    -1.57,   # wrist_2
     0.00,   # wrist_3
], dtype=np.float64)


# ============================================================
# Scene XML
# ============================================================

def create_scene_xml() -> str:
    table_center_x = -0.45
    table_center_y = 0.02

    # 3x3 grid in the pick area.
    grid_points = []
    spacing = 0.10
    for row in range(3):
        for col in range(3):
            x = PICK_AREA_CENTER[0] + (col - 1) * spacing
            y = PICK_AREA_CENTER[1] + (row - 1) * spacing
            grid_points.append((x, y))

    grid_xml = ""
    for i, (x, y) in enumerate(grid_points):
        grid_xml += f"""
        <geom name="pick_grid_{i}"
              type="box"
              pos="{x} {y} {TABLE_TOP_Z + 0.003}"
              size="0.025 0.025 0.003"
              rgba="0.02 0.02 0.02 1"
              contype="0"
              conaffinity="0"/>
        """

    xml = f"""
<mujoco model="ur10e_clean_pick_place_scene">

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

        <!-- Table -->
        <body name="table" pos="{table_center_x} {table_center_y} {-TABLE_THICKNESS / 2.0}">
            <geom name="table_top"
                  type="box"
                  size="0.95 0.65 {TABLE_THICKNESS / 2.0}"
                  rgba="0.90 0.90 0.90 1"
                  friction="1.0 0.01 0.001"/>
        </body>

        <!-- Black frame rails, similar to the real table -->
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

        <!-- Pick area -->
        <geom name="pick_area"
              type="box"
              pos="{PICK_AREA_CENTER[0]} {PICK_AREA_CENTER[1]} {PICK_AREA_CENTER[2]}"
              size="0.22 0.22 0.004"
              rgba="0.10 0.80 0.10 0.25"
              contype="0"
              conaffinity="0"/>

        <!-- Place area -->
        <geom name="place_area"
              type="box"
              pos="{PLACE_AREA_CENTER[0]} {PLACE_AREA_CENTER[1]} {PLACE_AREA_CENTER[2]}"
              size="0.30 0.12 0.004"
              rgba="0.02 0.28 0.95 0.45"
              contype="0"
              conaffinity="0"/>

        <!-- White board under pick grid -->
        <geom name="marker_board"
              type="box"
              pos="{PICK_AREA_CENTER[0]} {PICK_AREA_CENTER[1]} {TABLE_TOP_Z + 0.001}"
              size="0.28 0.28 0.001"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        {grid_xml}

        <!-- Cube object -->
        <body name="red_cube" pos="{CUBE_POS[0]} {CUBE_POS[1]} {CUBE_POS[2]}">
            <freejoint name="red_cube_freejoint"/>
            <geom name="red_cube_geom"
                  type="box"
                  size="{BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0} {BLOCK_SIZE / 2.0}"
                  rgba="1.0 0.05 0.05 1"
                  mass="0.05"
                  friction="1.2 0.01 0.001"
                  condim="3"/>
        </body>

        <!-- Visual target markers -->
        <site name="pick_target_marker"
              pos="{CUBE_POS[0]} {CUBE_POS[1]} 0.18"
              size="0.012"
              rgba="1 0 1 1"/>

        <site name="place_target_marker"
              pos="{PLACE_AREA_CENTER[0]} {PLACE_AREA_CENTER[1]} 0.18"
              size="0.012"
              rgba="0 1 1 1"/>

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


# ============================================================
# Main
# ============================================================

def main():
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

    print("Temporary clean pick-place scene:", scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Model loaded.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    set_robot_qpos(model, data, READY_Q)

    print("")
    print("Clean UR10e pick-and-place scene loaded.")
    print("This version includes:")
    print("  - UR10e robot")
    print("  - table")
    print("  - green pick area")
    print("  - blue place area")
    print("  - red cube")
    print("  - target markers")
    print("")
    print("No robot motion yet. Close the viewer to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:6] = data.qpos[:6]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
import os
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT_DIR = "/home/yuntian/ur10e_project"

UR10E_MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "mujoco_menagerie",
    "universal_robots_ur10e",
)

UR10E_XML = os.path.join(UR10E_MODEL_DIR, "ur10e.xml")

OUTPUT_FILE = os.path.join(PROJECT_DIR, "waypoints.json")


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


DEFAULT_Q = np.array([0.0, -1.3, 1.5, -1.8, -1.57, 0.0], dtype=np.float64)


SCENE_XML = f"""
<mujoco model="ur10e_teach_waypoints">

    <include file="ur10e.xml"/>

    <statistic center="0 0 0.4" extent="1.8"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <visual>
        <headlight diffuse="0.65 0.65 0.65"
                   ambient="0.35 0.35 0.35"
                   specular="0.1 0.1 0.1"/>
        <global azimuth="130" elevation="-26"/>
    </visual>

    <worldbody>
        <light name="top_light"
               pos="0 0 2.2"
               dir="0 0 -1"
               diffuse="0.85 0.85 0.85"/>

        <geom name="floor"
              type="plane"
              size="2.5 2.5 0.01"
              rgba="0.80 0.80 0.80 1"/>

        <body name="table" pos="0.0 0.0 -0.03">
            <geom name="table_top"
                  type="box"
                  size="1.0 0.65 0.03"
                  rgba="0.90 0.90 0.90 1"/>
        </body>

        <geom name="robot_base_plate"
              type="box"
              pos="0 0 0.005"
              size="0.15 0.15 0.005"
              rgba="0.12 0.12 0.15 1"
              contype="0"
              conaffinity="0"/>

        <!-- Left-side grid reference -->
        <geom name="grid_board"
              type="box"
              pos="-0.38 -0.12 0.002"
              size="0.25 0.23 0.002"
              rgba="0.96 0.96 0.94 1"
              contype="0"
              conaffinity="0"/>

        <!-- Blue place area -->
        <geom name="place_board"
              type="box"
              pos="-0.35 0.24 0.006"
              size="0.32 0.11 0.006"
              rgba="0.02 0.28 0.95 0.75"
              contype="0"
              conaffinity="0"/>

        <camera name="overview"
                pos="0.7 -1.1 1.0"
                xyaxes="0.70 0.71 0 -0.35 0.35 0.87"/>
    </worldbody>

</mujoco>
"""


def write_temp_scene():
    path = os.path.join(UR10E_MODEL_DIR, "teach_scene_tmp.xml")
    with open(path, "w") as f:
        f.write(SCENE_XML)
    return path


def print_help():
    print("")
    print("Commands:")
    print("  show")
    print("  j <joint_id> <delta>")
    print("     joint_id: 0 shoulder_pan")
    print("               1 shoulder_lift")
    print("               2 elbow")
    print("               3 wrist_1")
    print("               4 wrist_2")
    print("               5 wrist_3")
    print("     example: j 0 0.1")
    print("              j 1 -0.1")
    print("")
    print("  set <joint_id> <value>")
    print("     example: set 0 0.5")
    print("")
    print("  save <name>")
    print("     required names:")
    print("       ready")
    print("       mid")
    print("       pick_approach")
    print("       pick_down")
    print("       place_approach")
    print("       place_down")
    print("")
    print("  list")
    print("  reset")
    print("  quit")
    print("")


def apply_q(model, data, q):
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def main():
    if not os.path.exists(UR10E_XML):
        raise FileNotFoundError(f"Cannot find UR10e XML: {UR10E_XML}")

    scene_path = write_temp_scene()

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    q = DEFAULT_Q.copy()
    waypoints = {}

    apply_q(model, data, q)

    print("")
    print("Teach waypoint tool started.")
    print("Use terminal commands to move joints and save waypoints.")
    print_help()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)

            cmd = input("teach> ").strip()

            if cmd == "":
                continue

            parts = cmd.split()

            try:
                if parts[0] == "help":
                    print_help()

                elif parts[0] == "show":
                    print("")
                    for i, name in enumerate(JOINT_NAMES):
                        print(f"{i} {name:15s}: {q[i]: .4f}")
                    print("q =", np.round(q, 4).tolist())

                elif parts[0] == "j":
                    if len(parts) != 3:
                        print("Usage: j <joint_id> <delta>")
                        continue

                    jid = int(parts[1])
                    delta = float(parts[2])

                    if jid < 0 or jid >= 6:
                        print("joint_id must be 0 to 5")
                        continue

                    q[jid] += delta
                    apply_q(model, data, q)

                    print(f"{JOINT_NAMES[jid]} = {q[jid]:.4f}")

                elif parts[0] == "set":
                    if len(parts) != 3:
                        print("Usage: set <joint_id> <value>")
                        continue

                    jid = int(parts[1])
                    value = float(parts[2])

                    if jid < 0 or jid >= 6:
                        print("joint_id must be 0 to 5")
                        continue

                    q[jid] = value
                    apply_q(model, data, q)

                    print(f"{JOINT_NAMES[jid]} = {q[jid]:.4f}")

                elif parts[0] == "save":
                    if len(parts) != 2:
                        print("Usage: save <name>")
                        continue

                    name = parts[1]
                    waypoints[name] = q.copy().tolist()

                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(waypoints, f, indent=4)

                    print(f"Saved waypoint '{name}' to {OUTPUT_FILE}")
                    print("q =", np.round(q, 4).tolist())

                elif parts[0] == "list":
                    print("")
                    print("Saved waypoints:")
                    for name, saved_q in waypoints.items():
                        print(name, "=", np.round(saved_q, 4).tolist())

                elif parts[0] == "reset":
                    q = DEFAULT_Q.copy()
                    apply_q(model, data, q)
                    print("Reset to default q.")

                elif parts[0] == "quit":
                    print("Exiting.")
                    break

                else:
                    print("Unknown command. Type 'help'.")

            except Exception as e:
                print("Command error:", e)


if __name__ == "__main__":
    main()
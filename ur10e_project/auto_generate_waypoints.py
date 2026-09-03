import os
import json
import numpy as np
import mujoco


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
OUTPUT_FILE = os.path.join(PROJECT_DIR, "waypoints.json")

END_EFFECTOR_SITE = "attachment_site"


# ============================================================
# Task geometry
# ============================================================

TABLE_TOP_Z = 0.0
BLOCK_SIZE = 0.04
BLOCK_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.004

# Visual gripper tip offset relative to UR10e attachment_site.
# The gripper tip is along local -Z of attachment_site.
GRIPPER_TIP_OFFSET_LOCAL = np.array([0.0, 0.0, -0.145], dtype=np.float64)


# ============================================================
# Workspace layout
# ============================================================
#
# Grid is placed farther away from the base to reduce folded-arm motion.
# ============================================================

GRID_POINTS = [
    np.array([-0.75, -0.28, BLOCK_Z], dtype=np.float64),
    np.array([-0.65, -0.28, BLOCK_Z], dtype=np.float64),
    np.array([-0.55, -0.28, BLOCK_Z], dtype=np.float64),

    np.array([-0.75, -0.18, BLOCK_Z], dtype=np.float64),
    np.array([-0.65, -0.18, BLOCK_Z], dtype=np.float64),
    np.array([-0.55, -0.18, BLOCK_Z], dtype=np.float64),

    np.array([-0.75, -0.08, BLOCK_Z], dtype=np.float64),
    np.array([-0.65, -0.08, BLOCK_Z], dtype=np.float64),
    np.array([-0.55, -0.08, BLOCK_Z], dtype=np.float64),
]

PLACE_POINT = np.array([-0.65, 0.32, BLOCK_Z], dtype=np.float64)


SCENE_XML = """
<mujoco model="ur10e_auto_waypoint_search">
    <include file="ur10e.xml"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <worldbody>
        <geom name="floor"
              type="plane"
              size="2.5 2.5 0.01"
              rgba="0.8 0.8 0.8 1"/>
    </worldbody>
</mujoco>
"""


# ============================================================
# MuJoCo helpers
# ============================================================

def write_temp_scene():
    path = os.path.join(UR10E_MODEL_DIR, "auto_waypoint_scene_tmp.xml")
    with open(path, "w") as f:
        f.write(SCENE_XML)
    return path


def name_to_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Cannot find MuJoCo object: {name}")
    return idx


def get_site_pos(model, data, site_name):
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def get_site_rot(model, data, site_name):
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def get_tip_pos_and_rot(model, data, q):
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)

    site_pos = get_site_pos(model, data, END_EFFECTOR_SITE)
    site_rot = get_site_rot(model, data, END_EFFECTOR_SITE)

    tip_pos = site_pos + site_rot @ GRIPPER_TIP_OFFSET_LOCAL

    return tip_pos, site_rot


# ============================================================
# Cost functions
# ============================================================

def posture_cost(q):
    shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 = q

    natural = np.array([
        -0.75,
        -1.35,
        1.55,
        -1.80,
        -1.57,
        0.0
    ], dtype=np.float64)

    cost = 0.0

    # Keep close to natural tabletop-reaching pose.
    cost += 2.5 * np.linalg.norm(q - natural)

    # Avoid folded elbow.
    if elbow > 1.95:
        cost += 25.0 * abs(elbow - 1.95)

    if elbow < 1.00:
        cost += 25.0 * abs(1.00 - elbow)

    # Avoid shoulder being too high.
    if shoulder_lift > -1.00:
        cost += 25.0 * abs(shoulder_lift + 1.00)

    # Avoid wrist_1 being too shallow.
    if wrist_1 > -1.30:
        cost += 25.0 * abs(wrist_1 + 1.30)

    # Avoid wrist_1 being over-bent.
    if wrist_1 < -2.35:
        cost += 15.0 * abs(wrist_1 + 2.35)

    # Keep wrist_2 near -pi/2.
    cost += 2.0 * abs(wrist_2 + 1.57)

    return float(cost)


def orientation_cost(site_rot):
    """
    We want the gripper tip direction to point down.

    Because the visual gripper tip is along local -Z of attachment_site:
        tip direction = -site_z

    To make tip direction point downward [0, 0, -1],
    we want:
        site_z ≈ [0, 0, 1]
    """
    site_z_world = site_rot[:, 2]
    desired_site_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot = float(np.dot(site_z_world, desired_site_z))
    dot = np.clip(dot, -1.0, 1.0)

    # 0 means perfectly vertical; larger means more tilted.
    return 1.0 - dot


def evaluate_q(model, data, q, target_xy, target_z):
    tip, site_rot = get_tip_pos_and_rot(model, data, q)

    xy_error = np.linalg.norm(tip[:2] - target_xy)
    z_error = abs(tip[2] - target_z)

    orient_error = orientation_cost(site_rot)
    pose_error = posture_cost(q)

    # Position + vertical end-effector + natural posture.
    # Orientation has a strong weight because your current issue is tilted gripper.
    score = (
        1.00 * xy_error
        + 0.35 * z_error
        + 0.85 * orient_error
        + 0.08 * pose_error
    )

    return {
        "score": float(score),
        "xy_error": float(xy_error),
        "z_error": float(z_error),
        "orientation_error": float(orient_error),
        "posture_error": float(pose_error),
        "tip": tip.copy(),
        "site_z": site_rot[:, 2].copy(),
        "q": q.copy(),
    }


# ============================================================
# Search
# ============================================================

def search_grid(model, data, target_xy, target_z, ranges, label):
    best = None
    count = 0

    for shoulder_pan in ranges["shoulder_pan"]:
        for shoulder_lift in ranges["shoulder_lift"]:
            for elbow in ranges["elbow"]:
                for wrist_1 in ranges["wrist_1"]:
                    for wrist_2 in ranges["wrist_2"]:
                        q = np.array([
                            shoulder_pan,
                            shoulder_lift,
                            elbow,
                            wrist_1,
                            wrist_2,
                            0.0,
                        ], dtype=np.float64)

                        result = evaluate_q(
                            model=model,
                            data=data,
                            q=q,
                            target_xy=target_xy,
                            target_z=target_z,
                        )

                        if best is None or result["score"] < best["score"]:
                            best = result

                        count += 1

    print(f"{label}: searched {count} candidates")
    return best


def search_best_q(model, data, target_xy, target_z, label):
    print("")
    print("========================================")
    print(f"Searching waypoint for {label}")
    print("Target XY:", np.round(target_xy, 3), "target z:", round(float(target_z), 3))

    # Coarse search.
    # Wrist_2 is now searched too, because wrist_2 strongly affects tool direction.
    coarse_ranges = {
        "shoulder_pan": np.linspace(-1.45, -0.25, 23),
        "shoulder_lift": np.linspace(-1.60, -1.05, 15),
        "elbow": np.linspace(1.05, 1.95, 17),
        "wrist_1": np.linspace(-2.30, -1.35, 17),
        "wrist_2": np.linspace(-2.20, -0.90, 15),
    }

    coarse_best = search_grid(
        model,
        data,
        target_xy,
        target_z,
        coarse_ranges,
        label="coarse"
    )

    q0 = coarse_best["q"]

    print("Coarse best:")
    print("  q:", np.round(q0, 4).tolist())
    print("  tip:", np.round(coarse_best["tip"], 4).tolist())
    print("  site_z:", np.round(coarse_best["site_z"], 4).tolist())
    print("  xy_error:", round(coarse_best["xy_error"], 4))
    print("  z_error:", round(coarse_best["z_error"], 4))
    print("  orientation_error:", round(coarse_best["orientation_error"], 4))
    print("  score:", round(coarse_best["score"], 4))

    # Fine search around coarse result.
    fine_ranges = {
        "shoulder_pan": np.linspace(q0[0] - 0.10, q0[0] + 0.10, 11),
        "shoulder_lift": np.linspace(q0[1] - 0.08, q0[1] + 0.08, 9),
        "elbow": np.linspace(q0[2] - 0.10, q0[2] + 0.10, 9),
        "wrist_1": np.linspace(q0[3] - 0.12, q0[3] + 0.12, 9),
        "wrist_2": np.linspace(q0[4] - 0.15, q0[4] + 0.15, 11),
    }

    fine_best = search_grid(
        model,
        data,
        target_xy,
        target_z,
        fine_ranges,
        label="fine"
    )

    print("Fine best:")
    print("  q:", np.round(fine_best["q"], 4).tolist())
    print("  tip:", np.round(fine_best["tip"], 4).tolist())
    print("  site_z:", np.round(fine_best["site_z"], 4).tolist())
    print("  xy_error:", round(fine_best["xy_error"], 4))
    print("  z_error:", round(fine_best["z_error"], 4))
    print("  orientation_error:", round(fine_best["orientation_error"], 4))
    print("  score:", round(fine_best["score"], 4))
    print("========================================")

    return fine_best["q"]


def make_approach_from_down(q_down):
    q = q_down.copy()

    # Lift approach pose while keeping orientation mostly vertical.
    q[1] -= 0.10
    q[2] -= 0.07
    q[3] += 0.12

    return q


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(UR10E_XML):
        raise FileNotFoundError(f"Cannot find UR10e XML: {UR10E_XML}")

    scene_path = write_temp_scene()

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Use centre grid point as trained pick point.
    pick_target = GRID_POINTS[4]

    # Tip should be slightly above block centre/top.
    pick_tip_z = BLOCK_Z + 0.035
    place_tip_z = BLOCK_Z + 0.035

    q_pick_down = search_best_q(
        model=model,
        data=data,
        target_xy=pick_target[:2],
        target_z=pick_tip_z,
        label="pick_down"
    )

    q_place_down = search_best_q(
        model=model,
        data=data,
        target_xy=PLACE_POINT[:2],
        target_z=place_tip_z,
        label="place_down"
    )

    q_pick_approach = make_approach_from_down(q_pick_down)
    q_place_approach = make_approach_from_down(q_place_down)

    # Mid pose between pick and place.
    q_mid = 0.5 * (q_pick_approach + q_place_approach)
    q_mid[1] -= 0.08
    q_mid[2] -= 0.05
    q_mid[3] += 0.10

    # Ready pose is slightly more lifted.
    q_ready = q_mid.copy()
    q_ready[1] -= 0.10
    q_ready[2] -= 0.07
    q_ready[3] += 0.12

    waypoints = {
        "ready": q_ready.tolist(),
        "mid": q_mid.tolist(),
        "pick_approach": q_pick_approach.tolist(),
        "pick_down": q_pick_down.tolist(),
        "place_approach": q_place_approach.tolist(),
        "place_down": q_place_down.tolist(),

        "grid_points": [p.tolist() for p in GRID_POINTS],
        "place_point": PLACE_POINT.tolist(),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=4)

    print("")
    print("========================================")
    print("Saved waypoints to:")
    print(OUTPUT_FILE)
    print("========================================")
    print(json.dumps(waypoints, indent=4))


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import json
import os
import time


# ============================================================
# Camera settings
# ============================================================

CAMERA_INDEX = 0
USE_DSHOW = True

WINDOW_NAME = "Top View Table Calibration"

CALIB_FILE = "topview_calibration.json"


# ============================================================
# Real table / pick-area size
# ============================================================
# 这里先假设你点击的是 3x3 抓取区域的四个角。
# 例如真实抓取区域宽 30 cm，高 30 cm。
# 如果你的实际区域不是 30cm x 30cm，改这里。

REAL_WIDTH_M = 0.40
REAL_HEIGHT_M = 0.40


# ============================================================
# Global state
# ============================================================

clicked_points = []
homography = None
image_points = None
mouse_pos = None


# ============================================================
# Homography helpers
# ============================================================

def compute_homography(points):
    """
    points order:
    1. top-left
    2. top-right
    3. bottom-right
    4. bottom-left
    """

    src = np.array(points, dtype=np.float32)

    dst = np.array([
        [0.0, 0.0],
        [REAL_WIDTH_M, 0.0],
        [REAL_WIDTH_M, REAL_HEIGHT_M],
        [0.0, REAL_HEIGHT_M],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    return H


def pixel_to_table(pixel_point):
    """
    Convert one image pixel point to table coordinate in meters.
    """

    global homography

    if homography is None:
        return None

    pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    table_pt = cv2.perspectiveTransform(pt, homography)[0, 0]

    return table_pt


def table_to_pixel(table_point):
    """
    Convert one table coordinate point in meters back to image pixel.
    Useful for drawing real-distance markers.
    """

    global homography

    if homography is None:
        return None

    H_inv = np.linalg.inv(homography)

    pt = np.array([[[table_point[0], table_point[1]]]], dtype=np.float32)
    pixel_pt = cv2.perspectiveTransform(pt, H_inv)[0, 0]

    return pixel_pt


def save_calibration(points, H):
    data = {
        "real_width_m": REAL_WIDTH_M,
        "real_height_m": REAL_HEIGHT_M,
        "image_points": np.array(points, dtype=float).tolist(),
        "homography": H.tolist(),
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVE] Calibration saved to {CALIB_FILE}")


def load_calibration():
    global homography
    global image_points

    if not os.path.exists(CALIB_FILE):
        print("[INFO] No calibration file found.")
        return False

    with open(CALIB_FILE, "r") as f:
        data = json.load(f)

    image_points = np.array(data["image_points"], dtype=np.float32)
    homography = np.array(data["homography"], dtype=np.float32)

    print("[LOAD] Calibration loaded.")
    print("Image points:")
    print(image_points)
    print("Real size:")
    print(data["real_width_m"], "m x", data["real_height_m"], "m")

    return True


# ============================================================
# Mouse callback
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    global homography
    global image_points
    global mouse_pos

    mouse_pos = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if homography is not None:
        table_xy = pixel_to_table((x, y))
        if table_xy is not None:
            print(
                f"[POINT] pixel=({x}, {y})  "
                f"table=({table_xy[0]:.3f}, {table_xy[1]:.3f}) m"
            )
        return

    clicked_points.append([x, y])
    print(f"[CLICK] Corner {len(clicked_points)}: ({x}, {y})")

    if len(clicked_points) == 4:
        image_points = np.array(clicked_points, dtype=np.float32)
        homography = compute_homography(image_points)
        save_calibration(image_points, homography)

        print("[OK] Calibration completed.")
        print("Now click anywhere inside the area to see table coordinates in meters.")

        clicked_points = []


# ============================================================
# Drawing helpers
# ============================================================

def draw_calibration_polygon(frame):
    global image_points

    if image_points is not None:
        pts = image_points.astype(int)

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

        for i, p in enumerate(pts):
            cv2.circle(frame, tuple(p), 7, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(i + 1),
                tuple(p + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    for i, p in enumerate(clicked_points):
        p = np.array(p, dtype=int)
        cv2.circle(frame, tuple(p), 7, (255, 255, 0), -1)
        cv2.putText(
            frame,
            str(i + 1),
            tuple(p + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )


def draw_real_grid(frame):
    """
    Draw a real 3x3 grid after calibration.
    """

    if homography is None:
        return

    for i in range(4):
        x = REAL_WIDTH_M * i / 3.0
        p1 = table_to_pixel((x, 0.0))
        p2 = table_to_pixel((x, REAL_HEIGHT_M))

        if p1 is not None and p2 is not None:
            cv2.line(
                frame,
                tuple(p1.astype(int)),
                tuple(p2.astype(int)),
                (0, 180, 0),
                2,
            )

    for i in range(4):
        y = REAL_HEIGHT_M * i / 3.0
        p1 = table_to_pixel((0.0, y))
        p2 = table_to_pixel((REAL_WIDTH_M, y))

        if p1 is not None and p2 is not None:
            cv2.line(
                frame,
                tuple(p1.astype(int)),
                tuple(p2.astype(int)),
                (0, 180, 0),
                2,
            )


def draw_distance_circle(frame, center_table, radius_m):
    """
    Draw a real-distance circle projected to image.
    This circle represents a real radius on the table plane.
    """

    if homography is None:
        return

    points = []

    for angle in np.linspace(0, 2 * np.pi, 80):
        x = center_table[0] + radius_m * np.cos(angle)
        y = center_table[1] + radius_m * np.sin(angle)

        p = table_to_pixel((x, y))

        if p is not None:
            points.append(p.astype(int))

    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)


def draw_status(frame):
    y = 30

    if homography is None:
        text = "Click 4 corners: top-left, top-right, bottom-right, bottom-left"
        color = (0, 255, 255)
    else:
        text = "Calibration ready. Click any point to print real table coordinate."
        color = (255, 255, 255)

    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    y += 30

    cv2.putText(
        frame,
        "r = reset calibration, q = quit",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    y += 30

    if mouse_pos is not None and homography is not None:
        table_xy = pixel_to_table(mouse_pos)
        if table_xy is not None:
            text = f"Mouse table coordinate: x={table_xy[0]:.3f} m, y={table_xy[1]:.3f} m"
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )


# ============================================================
# Main
# ============================================================

def main():
    global clicked_points
    global homography
    global image_points

    load_calibration()

    if USE_DSHOW:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(1.0)

    if not cap.isOpened():
        print(f"Cannot open camera index {CAMERA_INDEX}")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("")
    print("Instructions:")
    print("1. Put the top-view camera in a fixed position.")
    print("2. Click 4 corners of the pick area.")
    print("   Order: top-left, top-right, bottom-right, bottom-left.")
    print("3. After calibration, click any point to see table coordinates in meters.")
    print("4. Press r to reset calibration.")
    print("5. Press q to quit.")
    print("")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        draw_calibration_polygon(frame)
        draw_real_grid(frame)
        draw_status(frame)

        if homography is not None:
            # Example: draw a real 8 cm radius circle around table center
            center_table = np.array([REAL_WIDTH_M / 2.0, REAL_HEIGHT_M / 2.0], dtype=np.float32)
            draw_distance_circle(frame, center_table, radius_m=0.08)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            print("[RESET] Calibration cleared.")
            clicked_points = []
            homography = None
            image_points = None

            if os.path.exists(CALIB_FILE):
                os.remove(CALIB_FILE)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
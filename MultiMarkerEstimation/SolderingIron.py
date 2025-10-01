import cv2
import numpy as np
from cv2 import aruco
import math
import time
import csv
import os

PLANE_ID = 7              # bord型 平面マーカの ID
PLANE_MARKER_LENGTH = 0.064        # bord型 マーカのサイズ
CUBE_IDS = [0,1,2,3]      # キューブに貼った 4 マーカの ID (前, 右, 後, 左)
CUBE_MARKER_LENGTH = 0.0315    # 31.5 mm (m単位)
CUBE_WIDTH  = 0.047       # 47 mm
CUBE_DEPTH  = 0.040       # 40 mm

# コテ先の位置
# これは "キューブの board座標系" における座標 (x,y,z) [meters]
TIP_OFFSET = np.array([0.0, -0.20, 0.00], dtype=np.float32)

# 平滑化 (EMA) の係数（0=no smoothing, 1=very slow）
POS_ALPHA = 0.4
ROT_ALPHA = 0.4

# ログ出力
LOG_CSV = True
LOG_PATH = "tip_log.csv"
# ----------------------------------

AXIS_LEN_CUBE = 0.03
AXIS_LEN_PLANE = 0.02

def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {file_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def marker_corners_on_face(center, u_vec, v_vec, marker_length):
    half = marker_length / 2.0
    tl = center - u_vec*half + v_vec*half
    tr = center + u_vec*half + v_vec*half
    br = center + u_vec*half - v_vec*half
    bl = center - u_vec*half - v_vec*half
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = tvec.reshape(3,)
    return T

def transform_inverse(T):
    R = T[:3,:3]; t = T[:3,3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3,:3] = R.T
    Tinv[:3,3] = -R.T @ t
    return Tinv

def transform_to_euler(T):
    R = T[:3,:3]
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def build_cube_board(marker_length, width, depth):
    half_w = width/2.0; half_d = depth/2.0
    X = np.array([1.0,0.0,0.0]); Y = np.array([0.0,1.0,0.0]); Z = np.array([0.0,0.0,1.0])
    c0 = np.array([0.0, 0.0, +half_d], dtype=np.float32)   # front (ID0)
    u0, v0 = X, Y
    c1 = np.array([+half_w, 0.0, 0.0], dtype=np.float32)    # right (ID1): outside right-> -Z
    u1, v1 = -Z, Y
    c2 = np.array([0.0, 0.0, -half_d], dtype=np.float32)    # back (ID2): outside right -> -X
    u2, v2 = -X, Y
    c3 = np.array([-half_w, 0.0, 0.0], dtype=np.float32)    # left (ID3): outside right -> +Z
    u3, v3 = Z, Y
    objPoints = [
        marker_corners_on_face(c0, u0, v0, marker_length),
        marker_corners_on_face(c1, u1, v1, marker_length),
        marker_corners_on_face(c2, u2, v2, marker_length),
        marker_corners_on_face(c3, u3, v3, marker_length),
    ]
    ids = np.array([[CUBE_IDS[0]],[CUBE_IDS[1]],[CUBE_IDS[2]],[CUBE_IDS[3]]], dtype=np.int32)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.Board(objPoints=objPoints, ids=ids, dictionary=dictionary)
    return board

def project_point_and_draw(img, point3d, rvec, tvec, camera_matrix, dist_coeffs, color=(0,0,255), radius=6):
    # project in the usual cv2 sense (point given in object's local coords using rvec,tvec)
    pts2d, _ = cv2.projectPoints(np.array([point3d], dtype=np.float32), rvec.reshape(3,1), tvec.reshape(3,1), camera_matrix, dist_coeffs)
    p = tuple(pts2d.ravel().astype(int))
    cv2.circle(img, p, radius, color, -1)
    return p

def main():
    camera_matrix, dist_coeffs = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(0)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    cube_board = build_cube_board(CUBE_MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH)

    # logging
    if LOG_CSV:
        header = ["time","cam_x_m","cam_y_m","cam_z_m","plane_x_m","plane_y_m","plane_z_m",
                  "roll_deg","pitch_deg","yaw_deg","dist3_m"]
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    ema_pos = None
    ema_rot_mat = None
    prev_T_plane_to_tip = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

        plane_avail = False; cube_avail = False
        plane_rvec = plane_tvec = None
        cube_rvec = cube_tvec = None

        if ids is not None:
            ids_list = ids.flatten().tolist()

            # plane single marker pose
            if PLANE_ID in ids_list:
                idx = ids_list.index(PLANE_ID)
                rvecs_p, tvecs_p, _ = aruco.estimatePoseSingleMarkers([corners[idx]], PLANE_MARKER_LENGTH, camera_matrix, dist_coeffs)
                plane_rvec = rvecs_p[0].reshape(3); plane_tvec = tvecs_p[0].reshape(3)
                plane_avail = True
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, plane_rvec, plane_tvec, AXIS_LEN_PLANE)

            # cube board pose: pass all detected corners/ids, estimatePoseBoard will use subset corresponding to board ids
            retval, rvec_b, tvec_b = aruco.estimatePoseBoard(corners, ids, cube_board, camera_matrix, dist_coeffs, None, None)
            if retval > 0:
                cube_rvec = rvec_b.reshape(3); cube_tvec = tvec_b.reshape(3)
                cube_avail = True
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, cube_rvec, cube_tvec, AXIS_LEN_CUBE)

            # draw detected markers for debug
            aruco.drawDetectedMarkers(frame, corners, ids)

        # if cube detected -> compute tip coords in camera frame
        if cube_avail:
            # Project tip (in cube local coords) into image using cube rvec/tvec
            # Note: cv2.projectPoints expects point(s) in the same coordinate frame as rvec/tvec (i.e., cube local coords)
            tip_2d = None
            try:
                # point in cube local coords (TIP_OFFSET)
                tip_2d = project_point_and_draw(frame, TIP_OFFSET, cube_rvec, cube_tvec, camera_matrix, dist_coeffs, color=(0,0,255), radius=6)
            except Exception as e:
                print("project error:", e)

        # if both plane and cube available, compute tip in plane frame and metrics
        if plane_avail and cube_avail:
            T_plane = rvec_tvec_to_transform(plane_rvec, plane_tvec)  # plane -> camera
            T_cube  = rvec_tvec_to_transform(cube_rvec, cube_tvec)    # cube  -> camera
            T_plane_inv = transform_inverse(T_plane)
            T_plane_to_cube = T_plane_inv @ T_cube

            # tip in cube local homogeneous
            tip_local_h = np.hstack([TIP_OFFSET.reshape(3,), 1.0])
            # tip in camera frame: cam_tip = T_cube @ tip_local_h
            cam_tip_h = (T_cube @ tip_local_h)
            cam_tip = cam_tip_h[:3]
            # tip in plane frame: plane_tip = inv(T_plane) * cam_tip
            plane_tip_h = (T_plane_inv @ cam_tip_h)
            plane_tip = plane_tip_h[:3]

            # rotation plane->cube
            R_plane_to_cube = T_plane_to_cube[:3,:3]
            euler = np.degrees(transform_to_euler(T_plane_to_cube))

            # smoothing EMA for position (camera frame)
            if ema_pos is None:
                ema_pos = cam_tip.copy()
            else:
                ema_pos = POS_ALPHA * cam_tip + (1-POS_ALPHA) * ema_pos

            # smoothing for rotation: convert to matrix and EMA on matrices (simple lerp then orthonormalize)
            if ema_rot_mat is None:
                ema_rot_mat = R_plane_to_cube.copy()
            else:
                M = ROT_ALPHA * R_plane_to_cube + (1-ROT_ALPHA) * ema_rot_mat
                # orthonormalize via SVD
                U, _, Vt = np.linalg.svd(M)
                ema_rot_mat = U @ Vt

            # compute distances
            dist3 = np.linalg.norm(plane_tip)
            dist_xy = np.linalg.norm(plane_tip[[0,1]])
            # show text
            cv2.putText(frame, f"Tip cam (m): x={cam_tip[0]:.3f}, y={cam_tip[1]:.3f}, z={cam_tip[2]:.3f}",
                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Tip plane (m): x={plane_tip[0]:.3f}, y={plane_tip[1]:.3f}, z={plane_tip[2]:.3f}",
                        (8,48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            cv2.putText(frame, f"Rot plane->cube (deg): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}",
                        (8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,80), 2)
            cv2.putText(frame, f"Dist3: {dist3*100:.1f} cm | DistXY: {dist_xy*100:.1f} cm",
                        (8,96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,80), 2)

            # show projected tip in image (re-project using cam rvec/tvec=identity)
            # For visibility, also project tip using plane or camera rvec/tvec: here we already drew tip via cube rvec/tvec above.

            # compute velocity if previous exists
            now = time.time()
            if prev_T_plane_to_tip is not None:
                dt = now - prev_time if prev_time else 1.0
                vel = (plane_tip - prev_T_plane_to_tip) / dt
                cv2.putText(frame, f"Vel plane(m/s): vx={vel[0]:.3f}, vy={vel[1]:.3f}, vz={vel[2]:.3f}",
                            (8,120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,200,120), 1)
            prev_T_plane_to_tip = plane_tip.copy()
            prev_time = now

            # logging
            if LOG_CSV:
                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(),
                                     cam_tip[0], cam_tip[1], cam_tip[2],
                                     plane_tip[0], plane_tip[1], plane_tip[2],
                                     euler[0], euler[1], euler[2],
                                     dist3])

        # show
        cv2.imshow("Tip Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

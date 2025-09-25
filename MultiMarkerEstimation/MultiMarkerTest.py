# MarkerRigidWithPlane.py
import cv2
import numpy as np
from cv2 import aruco
import math

# ---- (ユーザ設定) ----
PLANE_ID = 10        # 固定平面のマーカID（変更してください）
CUBE_IDS = [0,1,2,3] # キューブに貼った4つのマーカID (前, 右, 後, 左)
MARKER_LENGTH = 0.0315  # 31.5 mm (m単位)
CUBE_WIDTH  = 0.047     # 47 mm
CUBE_HEIGHT = 0.040     # 40 mm (高さは主に表示用)
CUBE_DEPTH  = 0.040     # 40 mm

AXIS_LEN_PLANE = 0.02   # 平面用軸表示長 (m)
AXIS_LEN_CUBE  = 0.03   # キューブ用軸表示長 (m)

# -----------------------

def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {file_path}")
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
    # rvec: (3,) Rodrigues, tvec: (3,)
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = tvec.reshape(3,)
    return T

def transform_inverse(T):
    R = T[:3,:3]
    t = T[:3,3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3,:3] = R.T
    Tinv[:3,3] = -R.T @ t
    return Tinv

def transform_to_euler(T):
    # returns roll, pitch, yaw in radians from rotation matrix (ZYX convention -> yaw around Z)
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
    return np.array([x, y, z])  # roll(x), pitch(y), yaw(z)

def build_cube_board(marker_length, width, depth):
    half_w = width/2.0
    half_d = depth/2.0
    m = marker_length

    X = np.array([1.0,0.0,0.0])
    Y = np.array([0.0,1.0,0.0])
    Z = np.array([0.0,0.0,1.0])

    # centers and u/v such that corners order = top-left, top-right, bottom-right, bottom-left
    c0 = np.array([0.0, 0.0, +half_d], dtype=np.float32)  # front (ID 0)
    u0, v0 = X, Y

    c1 = np.array([+half_w, 0.0, 0.0], dtype=np.float32)   # right (ID 1)外側から見た右 = -Z
    u1, v1 = -Z, Y

    c2 = np.array([0.0, 0.0, -half_d], dtype=np.float32)   # back (ID 2)外側から見た右 = -X
    u2, v2 = -X, Y

    c3 = np.array([-half_w, 0.0, 0.0], dtype=np.float32)   # left (ID 3)外側から見た右 = +Z
    u3, v3 = Z, Y

    objPoints = [
        marker_corners_on_face(c0, u0, v0, marker_length),
        marker_corners_on_face(c1, u1, v1, marker_length),
        marker_corners_on_face(c2, u2, v2, marker_length),
        marker_corners_on_face(c3, u3, v3, marker_length),
    ]
    ids = np.array([[0],[1],[2],[3]], dtype=np.int32)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.Board(objPoints=objPoints, ids=ids, dictionary=dictionary)
    return board

def main():
    camera_matrix, dist_coeffs = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(0)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # build cube board
    cube_board = build_cube_board(MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH)

    print("PLANE_ID:", PLANE_ID, "CUBE_IDS:", CUBE_IDS)
    prev_rel_T = None  # optional: store previous relative transform for motion calc

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

        plane_pose_available = False
        cube_pose_available = False
        plane_rvec = plane_tvec = None
        cube_rvec = cube_tvec = None

        if ids is not None:
            # --- single marker pose estimation for plane marker (if detected) ---
            ids_list = ids.flatten().tolist()
            # find plane marker index in detected list
            if PLANE_ID in ids_list:
                idx = ids_list.index(PLANE_ID)
                # corners[idx] corresponds to plane marker
                rvecs_p, tvecs_p, _ = aruco.estimatePoseSingleMarkers([corners[idx]], MARKER_LENGTH, camera_matrix, dist_coeffs)
                plane_rvec = rvecs_p[0].reshape(3)
                plane_tvec = tvecs_p[0].reshape(3)
                plane_pose_available = True
                # draw plane marker axis (green)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, plane_rvec, plane_tvec, AXIS_LEN_PLANE)

            # --- cube board pose estimation (using board) ---
            # But ensure we only pass cube-related corners & ids to estimatePoseBoard
            # We pass all detected corners/ids; estimatePoseBoard will use matching ids in board
            retval, rvec_b, tvec_b = aruco.estimatePoseBoard(corners, ids, cube_board, camera_matrix, dist_coeffs, None, None)
            if retval > 0:
                cube_rvec = rvec_b.reshape(3)
                cube_tvec = tvec_b.reshape(3)
                cube_pose_available = True
                # draw cube axes (blue)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, cube_rvec, cube_tvec, AXIS_LEN_CUBE)

            # also draw all detected marker outlines and ids
            aruco.drawDetectedMarkers(frame, corners, ids)

        # If both poses available, compute transform plane -> cube (cube expressed in plane frame)
        if plane_pose_available and cube_pose_available:
            T_plane = rvec_tvec_to_transform(plane_rvec, plane_tvec)   # plane -> camera
            T_cube  = rvec_tvec_to_transform(cube_rvec, cube_tvec)     # cube  -> camera
            # convert to camera<-plane, camera<-cube then plane->cube = inv(T_plane) * T_cube
            T_plane_inv = transform_inverse(T_plane)
            T_plane_to_cube = T_plane_inv @ T_cube

            # translation (in meters), rotation (radians -> degrees)
            rel_t = T_plane_to_cube[:3,3]
            rel_angles = np.degrees(transform_to_euler(T_plane_to_cube))

            # show numeric info
            txt1 = f"Rel Pos (plane->cube) (m): x={rel_t[0]:.3f}, y={rel_t[1]:.3f}, z={rel_t[2]:.3f}"
            txt2 = f"Rel Rot (deg): roll={rel_angles[0]:.1f}, pitch={rel_angles[1]:.1f}, yaw={rel_angles[2]:.1f}"
            cv2.putText(frame, txt1, (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, txt2, (8,56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # compute distance in XY plane or full 3D
            dist3 = np.linalg.norm(rel_t)
            dist_xy = np.linalg.norm(rel_t[[0,1]])
            cv2.putText(frame, f"Dist 3D: {dist3*100:.1f} cm  |  Dist XY: {dist_xy*100:.1f} cm",
                        (8,82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 2)

            # optionally compute velocity if previous transform exists
            if prev_rel_T is not None:
                dt_translation = (T_plane_to_cube[:3,3] - prev_rel_T[:3,3])
                dt_dist = np.linalg.norm(dt_translation)
                cv2.putText(frame, f"Delta (m): {dt_translation[0]:.3f},{dt_translation[1]:.3f},{dt_translation[2]:.3f}",
                            (8,108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,200,150), 1)

            prev_rel_T = T_plane_to_cube.copy()
        else:
            prev_rel_T = None

        cv2.imshow("Plane+RigidCube Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

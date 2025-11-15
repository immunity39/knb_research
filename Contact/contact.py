import cv2
import numpy as np
from cv2 import aruco
import math
import time
import csv
import os
import mediapipe as mp

# ======== 1. 定数の統合と設定 ========

# --- カメラ・キャリブレーション ---
CALIBRATION_FILE = "calibration.yaml"
CAM_ID = 0
FLIP_FRAME = False # 必要に応じてフレームを反転

# --- ArUco設定 ---
ARUCO_DICT = aruco.DICT_4X4_50

# --- 基板 (Plane) の設定 (hand_contact.py より) ---
# [重要] IDをコテと重複しないように変更 (例: 10, 11, 12, 13)
PLANE_MARKER_IDS = [10, 11, 12, 13] # [top-left, top-right, bottom-right, bottom-left]
PLANE_MARKER_LENGTH = 0.04     # 基板マーカのサイズ (m)
PLANE_MARKER_GAP_X = 0.20    # 基板マーカ間の横ギャップ (m)
PLANE_MARKER_GAP_Y = 0.15    # 基板マーカ間の縦ギャップ (m)
AXIS_LEN_PLANE = 0.03

# --- ハンダコテ (Cube) の設定 (SolderingIron.py より) ---
CUBE_IDS = [0, 1, 2, 3]      # キューブに貼った 4 マーカの ID (前, 右, 後, 左)
CUBE_MARKER_LENGTH = 0.0315    # 31.5 mm (m単位)
CUBE_WIDTH  = 0.047       # 47 mm
CUBE_DEPTH  = 0.040       # 40 mm
AXIS_LEN_CUBE = 0.03
# コテ先の位置 (キューブのローカル座標系)
TIP_OFFSET = np.array([0.0, -0.20, 0.00], dtype=np.float32)

# --- MediaPipe (Hand/Solder) の設定 ---
MP_MAX_HANDS = 2
MP_DETECT_CONF = 0.5
MP_TRACK_CONF = 0.5

# --- 平滑化 (EMA) の設定 ---
USE_SMOOTHING_PLANE = True   # 基板ポーズの平滑化
SMOOTH_ALPHA_PLANE = 0.3     # 基板ポーズのEMA係数
USE_SMOOTHING_TIP = True     # コテ先座標・姿勢の平滑化
SMOOTH_ALPHA_TIP_POS = 0.4   # コテ先位置のEMA係数
SMOOTH_ALPHA_TIP_ROT = 0.4   # コテ先姿勢のEMA係数

# --- 接触判定の閾値とデバウンス ---
# (A) コテ先 vs 基板
TIP_BOARD_Z_THRESH = 0.015   # 1.5 cm
TIP_BOARD_XY_THRESH = 0.05   # 基板中心から 5 cm 以内
# (B) 仮想ハンダ vs 基板
SOLDER_BOARD_XY_THRESH = 0.05 # 基板中心から 5 cm 以内
# (C) コテ先 vs 仮想ハンダ
TIP_SOLDER_DIST_THRESH = 0.02 # 2 cm
# 確認フレーム数（デバウンス）
TOUCH_CONFIRM_FRAMES = 3

# --- ログ出力 ---
LOG_CSV = True
LOG_PATH = "soldering_log.csv"
# ------------------------------------------


# ======== 2. ヘルパー関数のマージ ========

def load_camera_calibration(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {file_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

# --- 座標変換系 (SolderingIron.py より) ---
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

# --- コテ (Cube) の定義 (SolderingIron.py より) ---
def marker_corners_on_face(center, u_vec, v_vec, marker_length):
    half = marker_length / 2.0
    tl = center - u_vec*half + v_vec*half
    tr = center + u_vec*half + v_vec*half
    br = center + u_vec*half - v_vec*half
    bl = center - u_vec*half - v_vec*half
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def build_cube_board(marker_length, width, depth, dictionary):
    half_w = width/2.0; half_d = depth/2.0
    X = np.array([1.0,0.0,0.0]); Y = np.array([0.0,1.0,0.0]); Z = np.array([0.0,0.0,1.0])
    c0 = np.array([0.0, 0.0, +half_d], dtype=np.float32)   # front (ID0)
    u0, v0 = X, Y
    c1 = np.array([+half_w, 0.0, 0.0], dtype=np.float32)    # right (ID1)
    u1, v1 = -Z, Y
    c2 = np.array([0.0, 0.0, -half_d], dtype=np.float32)    # back (ID2)
    u2, v2 = -X, Y
    c3 = np.array([-half_w, 0.0, 0.0], dtype=np.float32)    # left (ID3)
    u3, v3 = Z, Y
    objPoints = [
        marker_corners_on_face(c0, u0, v0, marker_length),
        marker_corners_on_face(c1, u1, v1, marker_length),
        marker_corners_on_face(c2, u2, v2, marker_length),
        marker_corners_on_face(c3, u3, v3, marker_length),
    ]
    ids = np.array([[CUBE_IDS[0]],[CUBE_IDS[1]],[CUBE_IDS[2]],[CUBE_IDS[3]]], dtype=np.int32)
    return aruco.Board(objPoints=objPoints, ids=ids, dictionary=dictionary)

# --- 基板 (Plane) の定義 (hand_contact.py より) ---
def make_marker_object_points(x, y, z=0, marker_length=PLANE_MARKER_LENGTH):
    """指定座標(x, y)を中心としたマーカの4隅座標を返す"""
    half = marker_length / 2
    return np.array([
        [x - half, y + half, z],
        [x + half, y + half, z],
        [x + half, y - half, z],
        [x - half, y - half, z]
    ], dtype=np.float32)

def build_plane_board(dictionary):
    """4隅マーカで基板を定義する"""
    half_x = PLANE_MARKER_GAP_X / 2
    half_y = PLANE_MARKER_GAP_Y / 2
    coords = {
        "tl": (-half_x,  half_y), # ID 10
        "tr": ( half_x,  half_y), # ID 11
        "br": ( half_x, -half_y), # ID 12
        "bl": (-half_x, -half_y)  # ID 13
    }
    obj_points = [
        make_marker_object_points(*coords["tl"]),
        make_marker_object_points(*coords["tr"]),
        make_marker_object_points(*coords["br"]),
        make_marker_object_points(*coords["bl"])
    ]
    ids = np.array([[i] for i in PLANE_MARKER_IDS], dtype=np.int32)
    return aruco.Board(objPoints=obj_points, ids=ids, dictionary=dictionary)

# --- 描画・計算系 (両方から) ---
def project_point_and_draw(img, point3d, rvec, tvec, camera_matrix, dist_coeffs, color=(0,0,255), radius=6):
    """オブジェクトローカル座標系の点を画像に投影描画"""
    try:
        pts2d, _ = cv2.projectPoints(np.array([point3d], dtype=np.float32), rvec.reshape(3,1), tvec.reshape(3,1), camera_matrix, dist_coeffs)
        p = tuple(pts2d.ravel().astype(int))
        cv2.circle(img, p, radius, color, -1)
        return p
    except Exception:
        return None

def project_point_cam_coord(img, point3d_cam, camera_matrix, dist_coeffs, color=(255,255,0), radius=6):
    """カメラ座標系の3D点を画像に投影描画"""
    try:
        # rvec, tvec が 0 の場合、入力 point3d はカメラ座標系とみなされる
        pts2d, _ = cv2.projectPoints(np.array([point3d_cam], dtype=np.float32), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        p = tuple(pts2d.ravel().astype(int))
        cv2.circle(img, p, radius, color, -1)
        return p
    except Exception:
        return None

def line_plane_intersection(plane_point, plane_normal, ray_origin, ray_dir):
    """レイと平面の交点を計算"""
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None # 平行
    d = np.dot(plane_point - ray_origin, plane_normal) / denom
    if d < 0: # 交点がレイの始点より後ろ
        return None
    return ray_origin + d * ray_dir

def camera_to_local(point_cam, T_inv):
    """カメラ座標系 -> ローカル座標系"""
    point_cam_h = np.hstack([point_cam.reshape(3,), 1.0])
    point_local_h = (T_inv @ point_cam_h)
    return point_local_h[:3]


# ======== 3. メイン処理 ========
def main():
    # --- 初期化 ---
    camera_matrix, dist_coeffs = load_camera_calibration(CALIBRATION_FILE)
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAM_ID}")
        return

    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # ボード（コテと基板）の定義
    cube_board = build_cube_board(CUBE_MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH, dictionary)
    plane_board = build_plane_board(dictionary)

    # MediaPipe Hand の初期化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=MP_MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=MP_DETECT_CONF,
        min_tracking_confidence=MP_TRACK_CONF
    )

    # ログヘッダ
    if LOG_CSV:
        header = ["time",
                  "tip_cam_x", "tip_cam_y", "tip_cam_z",
                  "tip_plane_x", "tip_plane_y", "tip_plane_z",
                  "solder_plane_x", "solder_plane_y", "solder_plane_z",
                  "roll_deg", "pitch_deg", "yaw_deg",
                  "touch_tip_board", "touch_solder_board", "touch_tip_solder", "soldering_3way"]
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # 平滑化 (EMA) 用の変数
    ema_tip_pos = None
    ema_tip_rot_mat = None
    ema_plane_center = None
    ema_plane_normal = None
    
    # 接触判定デバウンス用
    count_tip_board = 0
    count_solder_board = 0
    count_tip_solder = 0
    
    print("=== Soldering Tracking Start ===")
    print(f"Cube IDs: {CUBE_IDS}")
    print(f"Plane IDs: {PLANE_MARKER_IDS}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

        # 状態リセット
        plane_avail = False
        cube_avail = False
        solder_avail = False
        
        # 座標系
        T_plane = T_plane_inv = None
        T_cube = None
        
        # 最終的な座標 (基板座標系)
        P_tip = None
        P_solder = None
        
        # カメラ座標系 (計算用)
        plane_center_cam = None
        plane_normal_cam = None
        hit_solder_cam = None # 仮想ハンダのカメラ座標系でのヒット位置

        # --- (A) 基板 (Plane) のポーズ推定 ---
        if ids is not None:
            retval_p, rvec_p, tvec_p = aruco.estimatePoseBoard(corners, ids, plane_board, camera_matrix, dist_coeffs, None, None)
            if retval_p > 0:
                plane_avail = True
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_p, tvec_p, AXIS_LEN_PLANE)
                
                # 生の変換行列
                T_plane_raw = rvec_tvec_to_transform(rvec_p, tvec_p)
                center_raw = T_plane_raw[:3, 3]
                normal_raw = T_plane_raw[:3, 2] # 基板のZ軸（法線）

                if USE_SMOOTHING_PLANE:
                    if ema_plane_center is None:
                        ema_plane_center = center_raw.copy()
                        ema_plane_normal = normal_raw.copy()
                    else:
                        ema_plane_center = SMOOTH_ALPHA_PLANE * center_raw + (1 - SMOOTH_ALPHA_PLANE) * ema_plane_center
                        ema_plane_normal = SMOOTH_ALPHA_PLANE * normal_raw + (1 - SMOOTH_ALPHA_PLANE) * ema_plane_normal
                        ema_plane_normal /= np.linalg.norm(ema_plane_normal) # 正規化
                    
                    plane_center_cam = ema_plane_center
                    plane_normal_cam = ema_plane_normal
                    # EMA後の値からrvec, tvecを再構成するのは難しいため、T_planeは生の値を使い、
                    # 指の交点計算のみEMA後の値を使う
                    T_plane = T_plane_raw
                    
                else:
                    plane_center_cam = center_raw
                    plane_normal_cam = normal_raw
                    T_plane = T_plane_raw
                
                T_plane_inv = transform_inverse(T_plane)


        # --- (B) コテ (Cube) のポーズ推定 ---
        if ids is not None:
            retval_c, rvec_c, tvec_c = aruco.estimatePoseBoard(corners, ids, cube_board, camera_matrix, dist_coeffs, None, None)
            if retval_c > 0:
                cube_avail = True
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_c, tvec_c, AXIS_LEN_CUBE)
                T_cube = rvec_tvec_to_transform(rvec_c, tvec_c)
                
                # コテ先を画像に投影 (デバッグ用)
                project_point_and_draw(frame, TIP_OFFSET, rvec_c, tvec_c, camera_matrix, dist_coeffs, color=(0,0,255), radius=6)

        aruco.drawDetectedMarkers(frame, corners, ids)

        # --- (C) 手・仮想ハンダ (Solder) の認識 ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False # 処理速度向上のため
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks and plane_avail:
            # 簡易的に、最初に見つかった手の人差し指(8)を使う
            # TODO: 左右の手を識別し、ハンダを持つ手を特定するロジック
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 人差し指の先端(8)のピクセル座標 (u, v) を取得
            tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            u = tip_landmark.x * frame.shape[1]
            v = tip_landmark.y * frame.shape[0]

            # カメラ座標系でのレイを生成
            pt_cam = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0])
            ray_dir = pt_cam / np.linalg.norm(pt_cam)
            ray_origin = np.zeros(3) # カメラ原点

            # 基板平面との交点 (カメラ座標系) を計算
            hit_solder_cam = line_plane_intersection(plane_center_cam, plane_normal_cam, ray_origin, ray_dir)

            if hit_solder_cam is not None:
                solder_avail = True
                # 交点を画像に描画 (デバッグ用)
                project_point_cam_coord(frame, hit_solder_cam, camera_matrix, dist_coeffs, color=(0, 255, 255), radius=5)
                # 基板座標系に変換
                P_solder = camera_to_local(hit_solder_cam, T_plane_inv)
                

        # --- (D) 座標計算と接触判定 ---
        frame_touch_tip_board = False
        frame_touch_solder_board = False
        frame_touch_tip_solder = False
        euler = np.zeros(3)

        if plane_avail and cube_avail:
            # (D-1) コテ先 (P_tip) の基板座標系での位置を計算
            tip_local_h = np.hstack([TIP_OFFSET.reshape(3,), 1.0])
            cam_tip_h = (T_cube @ tip_local_h)
            cam_tip = cam_tip_h[:3]
            plane_tip_h = (T_plane_inv @ cam_tip_h)
            P_tip_raw = plane_tip_h[:3]

            # (D-2) コテの角度 (Euler) の計算
            T_plane_to_cube = T_plane_inv @ T_cube
            R_plane_to_cube_raw = T_plane_to_cube[:3,:3]

            # (D-3) コテ先と角度の平滑化 (EMA)
            if USE_SMOOTHING_TIP:
                # 位置
                if ema_tip_pos is None: ema_tip_pos = P_tip_raw.copy()
                else: ema_tip_pos = SMOOTH_ALPHA_TIP_POS * P_tip_raw + (1 - SMOOTH_ALPHA_TIP_POS) * ema_tip_pos
                P_tip = ema_tip_pos
                
                # 姿勢
                if ema_tip_rot_mat is None: ema_tip_rot_mat = R_plane_to_cube_raw.copy()
                else:
                    M = SMOOTH_ALPHA_TIP_ROT * R_plane_to_cube_raw + (1 - SMOOTH_ALPHA_TIP_ROT) * ema_tip_rot_mat
                    U, _, Vt = np.linalg.svd(M) # オルソノーマライズ
                    ema_tip_rot_mat = U @ Vt
                
                # EMA後の回転行列からオイラー角を計算
                T_temp = np.eye(4); T_temp[:3,:3] = ema_tip_rot_mat
                euler = np.degrees(transform_to_euler(T_temp))
            else:
                P_tip = P_tip_raw
                euler = np.degrees(transform_to_euler(T_plane_to_cube))

            
            # (D-4) 接触判定
            
            # (A) コテ先 vs 基板
            dist_z_tip = abs(P_tip[2])
            dist_xy_tip = np.linalg.norm(P_tip[:2])
            if (dist_z_tip < TIP_BOARD_Z_THRESH) and (dist_xy_tip < TIP_BOARD_XY_THRESH):
                frame_touch_tip_board = True
            
            if solder_avail:
                # (B) 仮想ハンダ vs 基板
                # P_solder は既に基板座標系
                dist_xy_solder = np.linalg.norm(P_solder[:2])
                # Z距離は計算方法からほぼ0のはずだが、念のため
                dist_z_solder = abs(P_solder[2]) 
                
                if (dist_xy_solder < SOLDER_BOARD_XY_THRESH) and (dist_z_solder < TIP_BOARD_Z_THRESH): # コテ先と同じZ閾値
                    frame_touch_solder_board = True

                # (C) コテ先 vs 仮想ハンダ (3D距離)
                dist_3d_tip_solder = np.linalg.norm(P_tip - P_solder)
                if dist_3d_tip_solder < TIP_SOLDER_DIST_THRESH:
                    frame_touch_tip_solder = True

        
        # (D-5) デバウンス処理
        touch_confirmed_tip_board = False
        touch_confirmed_solder_board = False
        touch_confirmed_tip_solder = False
        touch_confirmed_3way = False

        count_tip_board = (count_tip_board + 1) * frame_touch_tip_board
        if count_tip_board >= TOUCH_CONFIRM_FRAMES: touch_confirmed_tip_board = True

        count_solder_board = (count_solder_board + 1) * frame_touch_solder_board
        if count_solder_board >= TOUCH_CONFIRM_FRAMES: touch_confirmed_solder_board = True

        count_tip_solder = (count_tip_solder + 1) * frame_touch_tip_solder
        if count_tip_solder >= TOUCH_CONFIRM_FRAMES: touch_confirmed_tip_solder = True
        
        # 3者同時接触
        if touch_confirmed_tip_board and touch_confirmed_solder_board and touch_confirmed_tip_solder:
            touch_confirmed_3way = True


        # --- (E) 描画とロギング ---
        
        # 座標情報
        if P_tip is not None:
             cv2.putText(frame, f"Tip(plane): x={P_tip[0]:.3f} y={P_tip[1]:.3f} z={P_tip[2]:.3f}",
                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        if P_solder is not None:
             cv2.putText(frame, f"Solder(plane): x={P_solder[0]:.3f} y={P_solder[1]:.3f} z={P_solder[2]:.3f}",
                        (8,48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        # 角度
        cv2.putText(frame, f"Rot(plane->cube): R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f}",
                    (8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,80), 2)

        # 接触判定
        color_tb = (0,255,0) if touch_confirmed_tip_board else (50,50,50)
        cv2.putText(frame, f"Tip-Board", (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tb, 2)
        
        color_sb = (0,255,0) if touch_confirmed_solder_board else (50,50,50)
        cv2.putText(frame, f"Solder-Board", (120, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_sb, 2)

        color_ts = (0,255,0) if touch_confirmed_tip_solder else (50,50,50)
        cv2.putText(frame, f"Tip-Solder", (250, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ts, 2)
        
        if touch_confirmed_3way:
            cv2.putText(frame, "SOLDERING DETECTED!", (8, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if LOG_CSV:
            log_data = [time.time(),
                        cam_tip[0] if 'cam_tip' in locals() else 0,
                        cam_tip[1] if 'cam_tip' in locals() else 0,
                        cam_tip[2] if 'cam_tip' in locals() else 0,
                        P_tip[0] if P_tip is not None else 0,
                        P_tip[1] if P_tip is not None else 0,
                        P_tip[2] if P_tip is not None else 0,
                        P_solder[0] if P_solder is not None else 0,
                        P_solder[1] if P_solder is not None else 0,
                        P_solder[2] if P_solder is not None else 0,
                        euler[0], euler[1], euler[2],
                        touch_confirmed_tip_board,
                        touch_confirmed_solder_board,
                        touch_confirmed_tip_solder,
                        touch_confirmed_3way]
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(log_data)


        cv2.imshow("Soldering Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()

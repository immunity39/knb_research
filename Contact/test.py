import cv2
import numpy as np
from cv2 import aruco
import math
import time
import csv
import os
import mediapipe as mp

# ======== 1. 定数設定 ========

# --- カメラ ---
CALIBRATION_FILE = "calibration.yaml"
CAM_ID = 0
FLIP_FRAME = False

# --- ArUco基板 (Plane) ---
ARUCO_DICT = aruco.DICT_4X4_50
PLANE_MARKER_IDS = [10, 11, 12, 13] 
PLANE_MARKER_LENGTH = 0.04
PLANE_MARKER_GAP_X = 0.20
PLANE_MARKER_GAP_Y = 0.15
AXIS_LEN_PLANE = 0.03

# --- ハンダコテ (Cube / Hakko FX601 T19-65C) ---
CUBE_IDS = [0, 1, 2, 3] # ID0=Front(斜面側), ID1=Right, ID2=Back, ID3=Left
CUBE_MARKER_LENGTH = 0.0315
CUBE_WIDTH  = 0.047
CUBE_DEPTH  = 0.040
AXIS_LEN_CUBE = 0.03

# コテ先位置 (キューブ中心からのオフセット)
TIP_OFFSET = np.array([0.0, -0.20, 0.00], dtype=np.float32)

# ★★★ コテ先斜面（ベベル）の設定 ★★★
# ID0 (Front, Z+) の方向に斜面が向いていると仮定
# T19-65C は 45度 or 60度カット。ここでは45度と仮定。
# 軸(-Y) から Z+ 方向に 45度 傾いたベクトルが「斜面の法線」
BEVEL_ANGLE_DEG = 45.0
_rad = np.radians(BEVEL_ANGLE_DEG)
# Normal vector in Cube Local Coords
# Y成分: -cos(45) (下向き), Z成分: +sin(45) (ID0側)
TIP_BEVEL_NORMAL_LOCAL = np.array([0.0, -np.cos(_rad), np.sin(_rad)], dtype=np.float32)
TIP_BEVEL_NORMAL_LOCAL /= np.linalg.norm(TIP_BEVEL_NORMAL_LOCAL)

# --- MediaPipe ---
MP_MAX_HANDS = 2
MP_DETECT_CONF = 0.5
MP_TRACK_CONF = 0.5
HAND_FOR_SOLDER = "Right"
HAND_FOR_IRON = "Left"
SOLDER_TIP_OFFSET_M = 0.04 

# --- 平滑化 (EMA) ---
USE_SMOOTHING_PLANE = True
SMOOTH_ALPHA_PLANE = 0.3
USE_SMOOTHING_TIP = True
SMOOTH_ALPHA_TIP_POS = 0.4
SMOOTH_ALPHA_TIP_ROT = 0.4

# --- 接触判定 ---
TIP_BOARD_Z_THRESH = 0.015   # 1.5cm
TIP_BOARD_XY_THRESH = 0.05   # 5.0cm
SOLDER_BOARD_XY_THRESH = 0.05
TIP_SOLDER_DIST_THRESH = 0.02
TOUCH_CONFIRM_FRAMES = 3

LOG_CSV = True
LOG_PATH = "soldering_log_v4.csv"

# 面積最大値
MAX_CONTACT_AREA_MM2 = 12.0 # T19-65Cは面積広め

# ------------------------------------------

def load_camera_calibration(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {file_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

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
    if sy < 1e-6:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    else:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    return np.array([x, y, z])

def marker_corners_on_face(center, u_vec, v_vec, marker_length):
    half = marker_length / 2.0
    return np.vstack([
        center - u_vec*half + v_vec*half,
        center + u_vec*half + v_vec*half,
        center + u_vec*half - v_vec*half,
        center - u_vec*half - v_vec*half
    ]).astype(np.float32)

def build_cube_board(marker_length, width, depth, dictionary):
    half_w = width/2.0; half_d = depth/2.0
    X = np.array([1.0,0.0,0.0]); Y = np.array([0.0,1.0,0.0]); Z = np.array([0.0,0.0,1.0])
    # ID0 (Front) at +Z
    objPoints = [
        marker_corners_on_face(np.array([0,0,half_d]), X, Y, marker_length),
        marker_corners_on_face(np.array([half_w,0,0]), -Z, Y, marker_length),
        marker_corners_on_face(np.array([0,0,-half_d]), -X, Y, marker_length),
        marker_corners_on_face(np.array([-half_w,0,0]), Z, Y, marker_length),
    ]
    ids = np.array([[i] for i in CUBE_IDS], dtype=np.int32)
    return aruco.Board(objPoints=objPoints, ids=ids, dictionary=dictionary)

def build_plane_board(dictionary):
    half_x = PLANE_MARKER_GAP_X / 2
    half_y = PLANE_MARKER_GAP_Y / 2
    def mk_pts(x,y):
        h = PLANE_MARKER_LENGTH/2
        return np.array([[x-h,y+h,0],[x+h,y+h,0],[x+h,y-h,0],[x-h,y-h,0]], dtype=np.float32)
    obj_points = [
        mk_pts(-half_x, half_y), mk_pts(half_x, half_y),
        mk_pts(half_x, -half_y), mk_pts(-half_x, -half_y)
    ]
    ids = np.array([[i] for i in PLANE_MARKER_IDS], dtype=np.int32)
    return aruco.Board(objPoints=obj_points, ids=ids, dictionary=dictionary)

def project_point_and_draw(img, pt3d, rvec, tvec, K, D, color=(0,0,255), radius=6):
    try:
        pts2d, _ = cv2.projectPoints(np.array([pt3d]), rvec, tvec, K, D)
        p = tuple(pts2d.ravel().astype(int))
        cv2.circle(img, p, radius, color, -1)
        return p
    except: return None

def project_point_cam(img, pt3d_cam, K, D, color=(255,255,0), radius=6):
    try:
        pts2d, _ = cv2.projectPoints(np.array([pt3d_cam]), np.zeros(3), np.zeros(3), K, D)
        p = tuple(pts2d.ravel().astype(int))
        cv2.circle(img, p, radius, color, -1)
        return p
    except: return None

def line_plane_intersection(plane_p, plane_n, ray_o, ray_d):
    denom = np.dot(plane_n, ray_d)
    if abs(denom) < 1e-6: return None
    d = np.dot(plane_p - ray_o, plane_n) / denom
    if d < 0: return None
    return ray_o + d * ray_d

def camera_to_local(pt_cam, T_inv):
    h = np.hstack([pt_cam, 1.0])
    return (T_inv @ h)[:3]

# ★★★ 新しい面積推定ロジック ★★★
def estimate_contact_area_bevel(R_plane_to_cube):
    """
    R_plane_to_cube: 基板座標系からコテ座標系への回転行列ではない。
                     通常 T_plane_inv @ T_cube で求まるのは「基板基準で見たコテの姿勢」
                     すなわち、基板座標系におけるコテの基底ベクトル。
    """
    if R_plane_to_cube is None: return 0.0, 0.0

    # 1. コテ先の斜面法線ベクトル（ローカル）を基板座標系に変換
    # V_world = R * V_local
    V_bevel_on_board = R_plane_to_cube @ TIP_BEVEL_NORMAL_LOCAL
    
    # 2. 基板の法線ベクトル（基板座標系なので常に Z=[0,0,1]）
    V_board_normal = np.array([0.0, 0.0, 1.0])
    
    # 3. アライメントの計算
    # 理想：コテ先斜面が基板にピタリとつく = 斜面法線が基板法線と「逆向き」
    # つまり V_bevel が [0,0,-1] に近いほど良い
    # dot(V_bevel, V_board) が -1.0 に近いほど良い
    
    dot_val = np.dot(V_bevel_on_board, V_board_normal)
    
    # -1.0 (Best) ~ 1.0 (Worst)
    # これを 0.0 (Worst) ~ 1.0 (Best) に正規化
    # 完全に逆向き(-1)なら1.0, 直角(0)なら0.0
    
    alignment_score = 0.0
    if dot_val < 0:
        alignment_score = abs(dot_val) # 下を向いている成分の大きさ
    else:
        alignment_score = 0.0 # 上を向いている＝背中が当たっている

    # 面積計算 (alignment_score の2乗などで重み付けも可)
    # 斜めカットなので、角度が合うと急激に面積が増える
    est_area = (alignment_score ** 2) * MAX_CONTACT_AREA_MM2
    
    return alignment_score, est_area

def main():
    K, D = load_camera_calibration(CALIBRATION_FILE)
    cap = cv2.VideoCapture(CAM_ID)
    dic = aruco.getPredefinedDictionary(ARUCO_DICT)
    
    cube_board = build_cube_board(CUBE_MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH, dic)
    plane_board = build_plane_board(dic)
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=MP_MAX_HANDS, min_detection_confidence=MP_DETECT_CONF)

    if LOG_CSV:
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["time","tip_px","tip_py","tip_pz","align_score","area","contact_3way"])

    # Smooth vars
    ema_tip_pos = None
    ema_tip_R = None
    ema_pl_cen = None
    ema_pl_norm = None

    # Debounce counters
    cnt_tb = cnt_sb = cnt_ts = 0
    
    print("=== Integrated Soldering V4 (Bevel Logic) ===")
    print(f"Bevel Normal Local: {TIP_BEVEL_NORMAL_LOCAL}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP_FRAME: frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = aruco.detectMarkers(gray, dic)
        
        plane_ok = False; cube_ok = False; solder_ok = False
        
        # Coords
        T_plane = None; T_plane_inv = None; T_cube = None
        P_tip_board = None; P_solder_board = None
        
        pl_cen_cam = None; pl_norm_cam = None
        
        align_score = 0.0; est_area = 0.0
        
        # --- Plane Detect ---
        if ids is not None:
            ret_p, rvec_p, tvec_p = aruco.estimatePoseBoard(corners, ids, plane_board, K, D, None, None)
            if ret_p > 0:
                plane_ok = True
                cv2.drawFrameAxes(frame, K, D, rvec_p, tvec_p, AXIS_LEN_PLANE)
                
                # 基板中心点(紫)
                project_point_and_draw(frame, np.zeros(3), rvec_p, tvec_p, K, D, (255,0,255), 8)
                
                T_raw = rvec_tvec_to_transform(rvec_p, tvec_p)
                c_raw = T_raw[:3,3]; n_raw = T_raw[:3,2]
                
                if USE_SMOOTHING_PLANE:
                    if ema_pl_cen is None:
                        ema_pl_cen = c_raw; ema_pl_norm = n_raw
                    else:
                        ema_pl_cen = SMOOTH_ALPHA_PLANE*c_raw + (1-SMOOTH_ALPHA_PLANE)*ema_pl_cen
                        ema_pl_norm = SMOOTH_ALPHA_PLANE*n_raw + (1-SMOOTH_ALPHA_PLANE)*ema_pl_norm
                        ema_pl_norm /= np.linalg.norm(ema_pl_norm)
                    pl_cen_cam = ema_pl_cen; pl_norm_cam = ema_pl_norm
                    # Note: T_plane rotation matrix isn't fully smoothed here, utilizing raw for transform base
                    # but using smoothed center/normal for raycasting
                    T_plane = T_raw 
                else:
                    pl_cen_cam = c_raw; pl_norm_cam = n_raw
                    T_plane = T_raw
                
                T_plane_inv = transform_inverse(T_plane)

        # --- Cube Detect ---
        if ids is not None:
            ret_c, rvec_c, tvec_c = aruco.estimatePoseBoard(corners, ids, cube_board, K, D, None, None)
            if ret_c > 0:
                cube_ok = True
                cv2.drawFrameAxes(frame, K, D, rvec_c, tvec_c, AXIS_LEN_CUBE)
                T_cube = rvec_tvec_to_transform(rvec_c, tvec_c)
                project_point_and_draw(frame, TIP_OFFSET, rvec_c, tvec_c, K, D, (0,0,255), 6)

        aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Hand Detect ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        h_solder = None; h_iron = None
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, meta in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = meta.classification[0].label
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                if label == HAND_FOR_SOLDER: h_solder = lm
                elif label == HAND_FOR_IRON: h_iron = lm

        # --- Solder Raycast ---
        if h_solder and plane_ok:
            try:
                tip8 = h_solder.landmark[8]; dip7 = h_solder.landmark[7]
                h, w_ = frame.shape[:2]
                u8, v8 = tip8.x*w_, tip8.y*h
                u7, v7 = dip7.x*w_, dip7.y*h
                
                # Ray 8
                pt8_c = np.linalg.inv(K) @ np.array([u8, v8, 1.0])
                dir8 = pt8_c / np.linalg.norm(pt8_c)
                hit8 = line_plane_intersection(pl_cen_cam, pl_norm_cam, np.zeros(3), dir8)
                
                # Ray 7
                pt7_c = np.linalg.inv(K) @ np.array([u7, v7, 1.0])
                dir7 = pt7_c / np.linalg.norm(pt7_c)
                hit7 = line_plane_intersection(pl_cen_cam, pl_norm_cam, np.zeros(3), dir7)
                
                if hit8 is not None and hit7 is not None:
                    vec_fin = hit8 - hit7
                    nm = np.linalg.norm(vec_fin)
                    if nm > 1e-6:
                        vec_fin /= nm
                        hit_solder = hit8 + SOLDER_TIP_OFFSET_M * vec_fin
                        solder_ok = True
                        project_point_cam(frame, hit_solder, K, D, (0,255,255), 6)
                        P_solder_board = camera_to_local(hit_solder, T_plane_inv)
            except: pass

        # --- Calculations & Logic ---
        ft_tb = ft_sb = ft_ts = False
        
        if plane_ok and cube_ok:
            # 1. Tip Position
            tip_loc_h = np.hstack([TIP_OFFSET, 1.0])
            cam_tip = (T_cube @ tip_loc_h)[:3]
            P_tip_raw = (T_plane_inv @ np.hstack([cam_tip, 1.0]))[:3]
            
            # 2. Rotation Plane->Cube
            T_p2c = T_plane_inv @ T_cube
            R_p2c_raw = T_p2c[:3,:3]

            # 3. Smoothing
            if USE_SMOOTHING_TIP:
                if ema_tip_pos is None: ema_tip_pos = P_tip_raw
                else: ema_tip_pos = SMOOTH_ALPHA_TIP_POS*P_tip_raw + (1-SMOOTH_ALPHA_TIP_POS)*ema_tip_pos
                P_tip_board = ema_tip_pos
                
                if ema_tip_R is None: ema_tip_R = R_p2c_raw
                else:
                    m = SMOOTH_ALPHA_TIP_ROT*R_p2c_raw + (1-SMOOTH_ALPHA_TIP_ROT)*ema_tip_R
                    u,_,vt = np.linalg.svd(m); ema_tip_R = u@vt
                R_calc = ema_tip_R
            else:
                P_tip_board = P_tip_raw
                R_calc = R_p2c_raw

            # 4. Area & Alignment (New Logic)
            align_score, est_area = estimate_contact_area_bevel(R_calc)

            # 5. Contact Check
            dz = abs(P_tip_board[2])
            dxy = np.linalg.norm(P_tip_board[:2])
            
            if dz < TIP_BOARD_Z_THRESH and dxy < TIP_BOARD_XY_THRESH:
                ft_tb = True
            
            if solder_ok:
                sdxy = np.linalg.norm(P_solder_board[:2])
                if sdxy < SOLDER_BOARD_XY_THRESH: # Z assumed ~0
                    ft_sb = True
                
                d3d = np.linalg.norm(P_tip_board - P_solder_board)
                if d3d < TIP_SOLDER_DIST_THRESH:
                    ft_ts = True

        # --- Debounce ---
        cnt_tb = (cnt_tb+1)*ft_tb
        ok_tb = (cnt_tb >= TOUCH_CONFIRM_FRAMES)
        
        cnt_sb = (cnt_sb+1)*ft_sb
        ok_sb = (cnt_sb >= TOUCH_CONFIRM_FRAMES)
        
        cnt_ts = (cnt_ts+1)*ft_ts
        ok_ts = (cnt_ts >= TOUCH_CONFIRM_FRAMES)
        
        ok_3way = (ok_tb and ok_sb and ok_ts)

        # --- UI Draw ---
        y0, dy = 100, 25
        
        # 1. Tip-Board
        col = (0,255,0) if ok_tb else (100,100,100)
        cv2.putText(frame, f"1. Tip-Board: {'CNT' if ok_tb else '---'}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        # 2. Solder-Board
        col = (0,255,0) if ok_sb else (100,100,100)
        cv2.putText(frame, f"2. Solder-Board: {'CNT' if ok_sb else '---'}", (10,y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        # 3. Tip-Solder
        col = (0,255,0) if ok_ts else (100,100,100)
        cv2.putText(frame, f"3. Tip-Solder: {'CNT' if ok_ts else '---'}", (10,y0+dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        # Status
        col_st = (0,0,255) if ok_3way else (150,150,150)
        txt_st = "SOLDERING!!" if ok_3way else "Wait..."
        cv2.putText(frame, f"STATUS: {txt_st}", (10,y0+dy*3+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_st, 2)
        
        # Bevel Info
        col_align = (0,255,0) if align_score > 0.8 else (0,255,255)
        cv2.putText(frame, f"Align: {align_score:.2f}", (10,y0+dy*5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_align, 2)
        cv2.putText(frame, f"Area : {est_area:.1f} mm2", (10,y0+dy*6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        
        # Bar for alignment
        bw = int(align_score * 100)
        cv2.rectangle(frame, (130, y0+dy*5-10), (130+100, y0+dy*5), (50,50,50), -1)
        cv2.rectangle(frame, (130, y0+dy*5-10), (130+bw, y0+dy*5), col_align, -1)

        # Log
        if LOG_CSV and P_tip_board is not None:
             with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([time.time(), 
                                        f"{P_tip_board[0]:.3f}", f"{P_tip_board[1]:.3f}", f"{P_tip_board[2]:.3f}",
                                        f"{align_score:.2f}", f"{est_area:.2f}", int(ok_3way)])

        cv2.imshow("Soldering V4", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
import cv2
import numpy as np
from cv2 import aruco
import math

# --- 設定 ---
CALIBRATION_FILE = "calibration.yaml"
CAM_ID = 0
MARKER_LEN_CUBE = 0.0315
CUBE_W = 0.047
CUBE_D = 0.040
CUBE_IDS = [0, 1, 2, 3]

# T19-65Cの設定
# ID0 (Front) が向いている方向を基準に、45度カットされていると仮定
# 軸はY軸(shaft), ID0はZ軸プラス方向
# 法線ベクトルは -Y方向(先端)と +Z方向(ID0側) の間
BEVEL_ANGLE_DEG = 45.0 
rad = np.radians(BEVEL_ANGLE_DEG)
# コテの軸方向(-Y) と 正面方向(+Z) の合成ベクトル
TIP_BEVEL_NORMAL_LOCAL = np.array([0.0, -np.cos(rad), np.sin(rad)])
TIP_BEVEL_NORMAL_LOCAL /= np.linalg.norm(TIP_BEVEL_NORMAL_LOCAL)

def load_calibration():
    fs = cv2.FileStorage(CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): return None, None
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, D

def main():
    K, D = load_calibration()
    if K is None:
        print("Calib file not found. using default.")
        K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=float)
        D = np.zeros(5)

    cap = cv2.VideoCapture(CAM_ID)
    dic = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Cube Board定義
    half_w, half_d = CUBE_W/2, CUBE_D/2
    # ID0 (Front, Z+)
    c0 = np.array([0,0,half_d]); u0, v0 = np.array([1,0,0]), np.array([0,1,0])
    # ID1 (Right, X+) -> Outside is -Z? No, standard logic from prev code
    # Previous logic: Right(ID1) normal was -Z. Let's stick to ID0 as Anchor.
    # We only care about ID0 being detected or the whole board being detected.
    
    def get_corners(c, u, v):
        h = MARKER_LEN_CUBE/2
        return np.array([c-u*h+v*h, c+u*h+v*h, c+u*h-v*h, c-u*h-v*h], dtype=np.float32)
    
    X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    obj_pts = [
        get_corners(np.array([0,0,half_d]), X, Y),      # ID0 Front
        get_corners(np.array([half_w,0,0]), -Z, Y),     # ID1 Right
        get_corners(np.array([0,0,-half_d]), -X, Y),    # ID2 Back
        get_corners(np.array([-half_w,0,0]), Z, Y)      # ID3 Left
    ]
    ids = np.array(CUBE_IDS, dtype=np.int32).reshape(-1,1)
    board = aruco.Board(obj_pts, dictionary=dic, ids=ids)

    print("=== Test Angle Start ===")
    print(f"Bevel Normal Local: {TIP_BEVEL_NORMAL_LOCAL}")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        corners, detected_ids, _ = aruco.detectMarkers(frame, dic)
        aruco.drawDetectedMarkers(frame, corners, detected_ids)
        
        score = 0.0
        
        if detected_ids is not None:
            ret, rvec, tvec = aruco.estimatePoseBoard(corners, detected_ids, board, K, D, None, None)
            if ret > 0:
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)
                
                # 回転行列
                R, _ = cv2.Rodrigues(rvec)
                
                # ローカルの法線ベクトルをカメラ座標系へ変換
                # V_cam = R * V_local
                bevel_normal_cam = R @ TIP_BEVEL_NORMAL_LOCAL
                
                # カメラ座標系での「真下（基板方向）」を仮定
                # ここでは簡易的にカメラのY軸プラス方向を「下」とみなしてテストします
                # （本来は基板マーカから基板法線を得るべきですが、単体テストのため）
                camera_down_vector = np.array([0.0, 1.0, 0.0]) 
                
                # 内積 (1.0に近いほど一致)
                alignment = np.dot(bevel_normal_cam, camera_down_vector)
                score = max(0.0, alignment) # 0~1
                
                # 描画: 法線ベクトル
                p_start = tvec.reshape(3)
                p_end = p_start + bevel_normal_cam * 0.05
                
                pt1, _ = cv2.projectPoints(p_start.reshape(1,3), np.zeros(3), np.zeros(3), K, D)
                pt2, _ = cv2.projectPoints(p_end.reshape(1,3), np.zeros(3), np.zeros(3), K, D)
                p1 = tuple(pt1[0][0].astype(int))
                p2 = tuple(pt2[0][0].astype(int))
                
                col = (0, 0, 255)
                if score > 0.9: col = (0, 255, 0) # Good alignment
                
                cv2.arrowedLine(frame, p1, p2, col, 2)
                cv2.putText(frame, f"Align: {score:.2f}", (p1[0]+10, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # ゲージ表示
        h, w = frame.shape[:2]
        bar_w = int(score * 200)
        cv2.rectangle(frame, (50, h-50), (250, h-30), (100,100,100), -1)
        cv2.rectangle(frame, (50, h-50), (50+bar_w, h-30), (0,255,255), -1)
        cv2.putText(frame, "Face Alignment", (50, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Angle Test", frame)
        if cv2.waitKey(1) == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
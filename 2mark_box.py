import cv2
import numpy as np
from cv2 import aruco

# カメラキャリブレーション結果を読み込む関数
def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def detect_and_estimate_pose():
    camera_id = 0
    marker_length = 0.009  # 0.9cmのマーカ
    cap = cv2.VideoCapture(camera_id)

    # カメラキャリブレーション結果を読み込む
    camera_matrix, dist_coeffs = load_camera_calibration("calibration.yaml")

    # マーカ辞書
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # 4面に配置したマーカの3D座標 (単位: m)
    # 例: 正方形の箱の1辺が1.5cm
    half_box = 0.015 / 2 # 1.5cmの半分

    # 4面に配置（例: ±X, ±Y 面）
    board = aruco.Board(
        ids = np.array([[0],[1],[2],[3]], dtype=np.int32),
        objPoints = [
            np.array([[-marker_length/2, -marker_length/2, half_box],
                    [ marker_length/2, -marker_length/2, half_box],
                    [ marker_length/2,  marker_length/2, half_box],
                    [-marker_length/2,  marker_length/2, half_box]], dtype=np.float32),

            np.array([[-marker_length/2, -marker_length/2, -half_box],
                    [ marker_length/2, -marker_length/2, -half_box],
                    [ marker_length/2,  marker_length/2, -half_box],
                    [-marker_length/2,  marker_length/2, -half_box]], dtype=np.float32),

            np.array([[-marker_length/2, -marker_length/2,  half_box],
                    [-marker_length/2,  marker_length/2,  half_box],
                    [-marker_length/2,  marker_length/2, -half_box],
                    [-marker_length/2, -marker_length/2, -half_box]], dtype=np.float32),

            np.array([[ marker_length/2, -marker_length/2,  half_box],
                    [ marker_length/2,  marker_length/2,  half_box],
                    [ marker_length/2,  marker_length/2, -half_box],
                    [ marker_length/2, -marker_length/2, -half_box]], dtype=np.float32)
        ],
        dictionary=dictionary
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

        if ids is not None:
            # Boardとして推定
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)
            if retval > 0:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.015)
                cv2.putText(frame, f"tvec: {tvec.ravel()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("frame", frame)

        # q key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_estimate_pose()
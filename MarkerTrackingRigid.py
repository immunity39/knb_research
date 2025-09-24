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

def marker_corners_on_face(center, u_vec, v_vec, marker_length):
    """
    center: (3,) array - マーカ中心の3D座標（board座標系）
    u_vec: (3,) array - その面の「右方向」単位ベクトル（board系）
    v_vec: (3,) array - その面の「上方向」単位ベクトル（board系）
    marker_length: float - マーカ辺長（m）
    戻り値: (4,3) ndarray の順序 top-left, top-right, bottom-right, bottom-left
    """
    half = marker_length / 2.0
    tl = center - u_vec*half + v_vec*half
    tr = center + u_vec*half + v_vec*half
    br = center + u_vec*half - v_vec*half
    bl = center - u_vec*half - v_vec*half
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def detect_and_estimate_pose():
    camera_id = 0
    marker_length = 0.0315  # 31.5 mm
    cap = cv2.VideoCapture(camera_id)

    camera_matrix, dist_coeffs = load_camera_calibration("calibration.yaml")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # キューブサイズ
    width  = 0.047 
    depth  = 0.040
    hw = width / 2
    hd = depth / 2
    m = marker_length / 2

    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    Z = np.array([0, 0, 1])

    c0 = np.array([0, 0, hd], dtype=np.float32)  # front
    u0, v0 = X,Y

    c1 = np.array([hw, 0, 0], dtype=np.float32)  # right
    u1 = -Z
    v1 = Y

    c2 = np.array([0, 0, -hd], dtype=np.float32) # back
    u2 = -X
    v2 = Y

    c3 = np.array([-hw, 0, 0], dtype=np.float32) # left
    u3 = Z
    v3 = Y

    objectPoints = [
        marker_corners_on_face(c0, u0, v0, marker_length),
        marker_corners_on_face(c1, u1, v1, marker_length),
        marker_corners_on_face(c2, u2, v2, marker_length),
        marker_corners_on_face(c3, u3, v3, marker_length)
    ]

    ids = np.array([[0], [1], [2], [3]], dtype=np.int32)
    board = aruco.Board(objPoints=objectPoints, ids=ids, dictionary=dictionary)

    # board = aruco.Board(
    #     objPoints=[
    #         # # 前面 (0) false front
    #         # np.array([[-m, -m,  hd],
    #         #           [ m, -m,  hd],
    #         #           [ m,  m,  hd],
    #         #           [-m,  m,  hd]], dtype=np.float32),
    #         # # 右面 (1)
    #         # np.array([[ hw, -m, -m],
    #         #           [ hw, -m,  m],
    #         #           [ hw,  m,  m],
    #         #           [ hw,  m, -m]], dtype=np.float32),
    #         # # 後面 (2)
    #         # np.array([[-m, -m, -hd],
    #         #           [ m, -m, -hd],
    #         #           [ m,  m, -hd],
    #         #           [-m,  m, -hd]], dtype=np.float32),   
    #         # # 左面 (3) false front
    #         # np.array([[-hw, -m, -m],
    #         #           [-hw, -m,  m],
    #         #           [-hw,  m,  m],
    #         #           [-hw,  m, -m]], dtype=np.float32)
    #     ],
    #     ids=np.array([[0], [1], [2], [3]], dtype=np.int32),
    #     dictionary=dictionary
    # )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

        if ids is not None:
            retval, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, camera_matrix, dist_coeffs, None, None
            )
            # 0-3以外のIDが検出された場合は表示しない
            if ids is not None and np.all((ids >= 0) & (ids <= 3)) and retval > 0:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                cv2.putText(frame, f"tvec: {tvec.ravel()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"rvec: {rvec.ravel()}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("frame", frame)

        # q key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_estimate_pose()

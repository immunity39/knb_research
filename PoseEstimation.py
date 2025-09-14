import cv2
import cv2.aruco as aruco
import numpy as np

# カメラキャリブレーション結果を読み込む関数
def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def detect_and_estimate_pose(camera_id=0, marker_length=0.05):  # 5cmのマーカ
    cap = cv2.VideoCapture(camera_id)

    # ArUco辞書と検出パラメータ
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters()

    # カメラ内部パラメータをロード
    camera_matrix, dist_coeffs = load_camera_calibration("calibration.yaml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # マーカ検出
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # マーカを描画
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 姿勢推定
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # 座標軸を描画（長さ = 0.03m = 3cm）
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
                # aruco.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # 結果をコンソール出力
                print(f"ID: {ids[i][0]}")
                print(f"  移動ベクトル tvec: {tvecs[i].ravel()}")
                print(f"  回転ベクトル rvec: {rvecs[i].ravel()}")

                # rvec を回転行列に変換
                R, _ = cv2.Rodrigues(rvecs[i])
                # オイラー角に変換（Z-Y-X順）
                sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
                x_angle = np.degrees(np.arctan2(R[2,1], R[2,2]))
                y_angle = np.degrees(np.arctan2(-R[2,0], sy))
                z_angle = np.degrees(np.arctan2(R[1,0], R[0,0]))
                print(f"  角度 [deg]: X={x_angle:.1f}, Y={y_angle:.1f}, Z={z_angle:.1f}")

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_estimate_pose(camera_id=0, marker_length=0.05)

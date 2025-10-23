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

def detect_and_estimate_pose(camera_id=0, marker_length=0.009):  # 1cmのマーカ
    cap = cv2.VideoCapture(camera_id)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # カメラキャリブレーション結果を読み込む
    cameraMatrix, distCoeffs = load_camera_calibration("calibration.yaml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # マーカ検出
        corners, ids, rejected = aruco.detectMarkers(frame, dictionary, parameters=parameters)

        if ids is not None and len(ids) >= 2:
            # 姿勢推定
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, distCoeffs)

            # 各マーカを描画
            aruco.drawDetectedMarkers(frame, corners, ids)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.009)

                # コンソール出力用配列
                output = []

                # 2重forでマーカ間の距離・角度を計算
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        tvec1, rvec1 = tvecs[i][0], rvecs[i][0]
                        tvec2, rvec2 = tvecs[j][0], rvecs[j][0]

                        # 距離
                        dist = np.linalg.norm(tvec1 - tvec2)

                        # 角度差
                        R1, _ = cv2.Rodrigues(rvec1)
                        R2, _ = cv2.Rodrigues(rvec2)
                        R_rel = R2 @ R1.T
                        angle = np.arccos((np.trace(R_rel) - 1) / 2)

                        # 結果を配列に格納
                        output .append((ids[i][0], ids[j][0], dist, angle))

                # 結果をコンソール出力
                for id1, id2, dist, angle in output:
                    # print(f"ID {id1} - ID {id2}: 距離 = {dist*100:.2f} cm, 角度差 = {np.degrees(angle):.2f} deg")
                    cv2.putText(frame, f"ID {id1}-{id2}: {dist*100:.1f}cm, {np.degrees(angle):.1f}deg",
                                (10, 90 + 30 * (id1 + id2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        cv2.imshow("Aruco Multi-Marker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_estimate_pose(camera_id=0, marker_length=0.009)  # 1cmのマーカ
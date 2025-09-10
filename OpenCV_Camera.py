import cv2
import cv2.aruco as aruco

def detect_aruco_from_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)

    # 利用する辞書を指定
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # マーカ検出
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # マーカの枠とIDを描画
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids):
                print(f"検出マーカID: {marker_id[0]}")

        cv2.imshow("ArUco Detection", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_from_camera(camera_id=0)

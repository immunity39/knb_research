#!/usr/bin/env python3
# coding: utf-8

import cv2
from cv2 import aruco

camera_id = 0
input_file = ""
output_file = ""

def ReadArUco():

    # get dicionary and get parameters
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # read from image
    input_img = cv2.imread(input_file)

    # detect and draw marker's information
    corners, ids, rejectedCandidates = aruco.detectMarkers(input_img, dictionary, parameters=parameters)
    print(ids)
    ar_image = aruco.drawDetectedMarkers(input_img, corners, ids)

    cv2.imwrite(output_file, ar_image)

def CaptureArUco():
    cap = cv2.VideoCapture(camera_id)

    # 利用する辞書を指定
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # マーカ検出
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

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
    camera_id = 0
    CaptureArUco()
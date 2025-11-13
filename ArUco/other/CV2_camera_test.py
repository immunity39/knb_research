## Linux カメラデバイスid 確認方法
## v4l2-ctl --list-devices
## 結果 UVC Camera:
##         /dev/video0

import cv2

# ウェブカメラのキャプチャを開始
cap = cv2.VideoCapture(0)

# キャプチャがオープンしている間続ける
while(cap.isOpened()):
    # フレームを読み込む
    ret, frame = cap.read()
    if ret == True:
        # フレームを表示
        cv2.imshow('Webcam Live', frame)

        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# キャプチャをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()

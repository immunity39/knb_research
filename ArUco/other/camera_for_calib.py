# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time

# ウェブカメラのキャプチャを開始
device_id = 0
cap = cv2.VideoCapture(device_id)
pic_count = 0

# キャプチャがオープンしている間続ける
while(cap.isOpened()):
    # フレームを読み込む
    ret, frame = cap.read()
    if ret == True:
        # フレームを表示
        cv2.imshow('Webcam Live', frame)

        # enter キーが押されたら画像を保存
        if cv2.waitKey(1) & 0xFF == 13:  # 13はEnterキー
            filename = f'./image_{pic_count:02d}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')
            time.sleep(0.25)  # 連続撮影防止のため少し待つ
            pic_count += 1

        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# キャプチャをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()


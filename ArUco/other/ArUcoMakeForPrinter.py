#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from cv2 import aruco

def ArUcoGen():
    for aruco_num in range(0, 4):
        # プリンタの解像度
        dpi = 1200  # 例: 1200 DPI

        # マーカの物理サイズ（mm）
        marker_size_mm = 35.0  

        # mm → ピクセルに変換
        px_per_mm = dpi / 25.4
        size_px = int(round(marker_size_mm * px_per_mm))

        # 余白（オフセット）を設定。たとえばマーカサイズの 5% を余白とする
        margin_ratio = 0.05
        offset = int(round(size_px * margin_ratio))
        x_offset = y_offset = offset // 2

        # 辞書とマーカ画像を生成
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        ar_img = aruco.generateImageMarker(dictionary, aruco_num, size_px)

        # 白背景の画像（余白込み）
        img = np.ones((size_px + offset, size_px + offset), dtype=np.uint8) * 255

        # マーカを白背景の中央近くに配置
        img[y_offset : y_offset + ar_img.shape[0],
            x_offset : x_offset + ar_img.shape[1]] = ar_img

        # 保存
        fileName = f"./ArUcoMake/35mm{aruco_num}.png"
        cv2.imwrite(fileName, img)
        print(f"Saved {fileName} — {marker_size_mm} mm ≒ {size_px} px (DPI: {dpi})")

if __name__ == "__main__":
    ArUcoGen()

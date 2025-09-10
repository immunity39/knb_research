# openCV install (already python3 can use enviroment)
# pip install opencv-contrib-python

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from cv2 import aruco

def ArUcoGen():
    fileName = "./ArUcoMake/aruco2.png"
    # Size and offset value
    size = 150
    offset = 10
    x_offset = y_offset = int(offset) // 2

    # get dictionary and generate image
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    ar_img = aruco.generateImageMarker(dictionary, 1, size)

    # make white image
    img = np.zeros((size + offset, size + offset), dtype=np.uint8)
    img += 255

    # overlap image
    img[y_offset:y_offset + ar_img.shape[0], x_offset:x_offset + ar_img.shape[1]] = ar_img

    cv2.imwrite(fileName, img)

if __name__ == "__main__":
    ArUcoGen()
import cv2
import numpy as np
import glob

# チェッカーボードの交点数（マス目数 -1）
CHECKERBOARD = (9, 6)  # 9x6 の交点
square_size = 0.025    # マス目の1辺のサイズ [m] (例: 25mm)

# 3D点の座標系準備（原点から平面上に格子点を配置）
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 検出結果を保存
objpoints = []  # 3D点
imgpoints = []  # 2D点

images = glob.glob('calib_images/*.jpg')  # 画像フォルダ

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # コーナー検出
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # サブピクセル精度にコーナーを補正
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 検出結果を描画
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# キャリブレーション実行
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:\n", camera_matrix)
print("Distortion coeffs:\n", dist_coeffs)

# 保存
fs = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeff", dist_coeffs)
fs.release()
print("Calibration saved to calibration.yaml")

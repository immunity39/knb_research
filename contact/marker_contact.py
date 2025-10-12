import cv2, math, time, csv, os
import numpy as np
from collections import deque
from cv2 import aruco
import matplotlib.pyplot as plt

# ------------- user params -------------
PLANE_ID = 7
PLANE_MARKER_LENGTH = 0.064
CUBE_IDS = [0,1,2,3]
CUBE_MARKER_LENGTH = 0.0315
CUBE_WIDTH = 0.047
CUBE_DEPTH = 0.040

TIP_OFFSET = np.array([0.0, -0.020, 0.00], dtype=np.float32)  # m
CONTACT_THRESHOLD = 0.0015  # m (1.5 mm)
LOG_CSV = True
LOG_PATH = "marker_tip_log.csv"
MAX_HISTORY = 300  # plot points
CAM_ID = 0
# ---------------------------------------

# helper functions (same as you had)
def load_camera_calibration(file_path="calibration.yaml"):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("calibration.yaml not found")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeff").mat()
    fs.release()
    return K, dist

def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4, dtype=np.float64); T[:3,:3]=R; T[:3,3]=tvec.reshape(3,)
    return T

def transform_inverse(T):
    R = T[:3,:3]; t = T[:3,3]
    Tinv = np.eye(4, dtype=np.float64); Tinv[:3,:3]=R.T; Tinv[:3,3] = -R.T @ t
    return Tinv

# minimal board builder (same approach)
def marker_corners_on_face(center, u_vec, v_vec, marker_length):
    half = marker_length/2.0
    tl = center - u_vec*half + v_vec*half
    tr = center + u_vec*half + v_vec*half
    br = center + u_vec*half - v_vec*half
    bl = center - u_vec*half - v_vec*half
    return np.vstack([tl,tr,br,bl]).astype(np.float32)

def build_cube_board(marker_length, width, depth):
    half_w=width/2.0; half_d=depth/2.0
    X=np.array([1,0,0]); Y=np.array([0,1,0]); Z=np.array([0,0,1])
    c0=np.array([0,0,+half_d]); u0,v0 = X,Y
    c1=np.array([+half_w,0,0]); u1,v1 = -Z,Y
    c2=np.array([0,0,-half_d]); u2,v2 = -X,Y
    c3=np.array([-half_w,0,0]); u3,v3 = Z,Y
    objPoints = [
        marker_corners_on_face(c0,u0,v0,marker_length),
        marker_corners_on_face(c1,u1,v1,marker_length),
        marker_corners_on_face(c2,u2,v2,marker_length),
        marker_corners_on_face(c3,u3,v3,marker_length)
    ]
    ids = np.array([[CUBE_IDS[0]],[CUBE_IDS[1]],[CUBE_IDS[2]],[CUBE_IDS[3]]], dtype=np.int32)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    return aruco.Board(objPoints=objPoints, ids=ids, dictionary=dictionary)

# plotting helper
def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height (mm)")
    ax.set_ylim(-5, 50)  # adjust depending expected heights
    line_height, = ax.plot([], [], '-o', lw=1, ms=3, label="tip height (mm)")
    ax.legend(loc="upper right")
    return fig, ax, line_height

def main():
    K, dist = load_camera_calibration("calibration.yaml")
    cap = cv2.VideoCapture(CAM_ID)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    cube_board = build_cube_board(CUBE_MARKER_LENGTH, CUBE_WIDTH, CUBE_DEPTH)

    if LOG_CSV:
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["time","frame","tip_plane_x_m","tip_plane_y_m","tip_plane_z_m","contact"])

    # plotting buffers
    times = deque(maxlen=MAX_HISTORY)
    heights = deque(maxlen=MAX_HISTORY)
    contacts = deque(maxlen=MAX_HISTORY)

    fig, ax, line_height = init_plot()

    frame_idx = 0
    contact_active=False
    contact_start=None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)

        plane_avail=False; cube_avail=False
        plane_rvec=plane_tvec=None; cube_rvec=cube_tvec=None

        if ids is not None:
            ids_list = ids.flatten().tolist()
            if PLANE_ID in ids_list:
                idx = ids_list.index(PLANE_ID)
                rvecs_p, tvecs_p, _ = aruco.estimatePoseSingleMarkers([corners[idx]], PLANE_MARKER_LENGTH, K, dist)
                plane_rvec = rvecs_p[0].reshape(3); plane_tvec = tvecs_p[0].reshape(3)
                plane_avail=True
                cv2.drawFrameAxes(frame, K, dist, plane_rvec, plane_tvec, 0.02)
            retval, rvec_b, tvec_b = aruco.estimatePoseBoard(corners, ids, cube_board, K, dist, None, None)
            if retval>0:
                cube_rvec = rvec_b.reshape(3); cube_tvec = tvec_b.reshape(3)
                cube_avail=True
                cv2.drawFrameAxes(frame, K, dist, cube_rvec, cube_tvec, 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

        tip_plane = None
        if plane_avail and cube_avail:
            T_plane = rvec_tvec_to_transform(plane_rvec, plane_tvec)
            T_cube = rvec_tvec_to_transform(cube_rvec, cube_tvec)
            T_plane_inv = transform_inverse(T_plane)
            tip_local_h = np.hstack([TIP_OFFSET.reshape(3,), 1.0])
            plane_tip = (T_plane_inv @ (T_cube @ tip_local_h))[:3]
            tip_plane = plane_tip
            tip_height = plane_tip[2]
            contact_now = abs(tip_height) < CONTACT_THRESHOLD
        else:
            tip_height = np.nan
            contact_now = False

        # record + log
        tnow = time.time()
        times.append(tnow)
        heights.append(tip_height*1000 if not math.isnan(tip_height) else np.nan)
        contacts.append(1 if contact_now else 0)

        if LOG_CSV and tip_plane is not None:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([tnow, frame_idx, tip_plane[0], tip_plane[1], tip_plane[2], int(contact_now)])

        # contact state transition detection
        if contact_now and not contact_active:
            contact_active=True; contact_start=tnow; print(f"[marker] contact start @ {tnow}")
        if (not contact_now) and contact_active:
            contact_active=False; duration = tnow - (contact_start or tnow); print(f"[marker] contact end, dur={duration:.3f}s"); contact_start=None

        # update plot (simple)
        if len(times)>1:
            t0 = times[0]
            x = [tt - t0 for tt in times]
            y = list(heights)
            line_height.set_data(x, y)
            ax.set_xlim(max(0, x[-1]-10), x[-1]+0.1)  # show last ~10 sec
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            plt.pause(0.001)

        # draw text on frame
        text = f"TipZ(mm): {heights[-1]:.2f}" if not math.isnan(heights[-1]) else "TipZ(mm): N/A"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if contact_now else (200,200,200), 2)
        if contact_now:
            cv2.putText(frame, "CONTACT", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Marker Tip Contact", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    print("Finished. CSV:", LOG_PATH)

if __name__ == "__main__":
    main()

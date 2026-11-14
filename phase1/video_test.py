#!/usr/bin/env python3
"""
Vision + PID test on a recorded video (or webcam). NO drone, NO dronekit needed.

Install:  pip install opencv-python numpy
Run:      python video_test.py balloon.mp4        (or: python video_test.py 0   for webcam)

Keys:  SPACE = pause/resume    q = quit (prints the HSV ranges to paste into the main script)
Use the sliders in the "controls" window to tune the colour until the mask shows ONLY the balloon.
"""
import argparse
import math

import cv2
import numpy as np

W, H = 640, 480
HFOV_DEG = 62.2                                   # set to YOUR camera
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))   # ~530 px

KP_YAW, KI_YAW, KD_YAW = 1.5, 0.0, 0.10
KP_Z, KI_Z, KD_Z = 2.0, 0.0, 0.10
MAX_YAW_RATE, MAX_VZ = 1.0, 0.5
MIN_RADIUS_PX = 8


class PID:
    def __init__(self, kp, ki, kd, out_limit, i_limit=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit = out_limit, i_limit
        self.integral, self.prev_err = 0.0, None

    def reset(self):
        self.integral, self.prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral + err * dt))
        deriv = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        return max(-self.out_limit, min(self.out_limit, out))


def nothing(_):
    pass


def get_ranges():
    hl = cv2.getTrackbarPos("H_lo", "controls")
    hh = cv2.getTrackbarPos("H_hi", "controls")
    s = cv2.getTrackbarPos("S_min", "controls")
    v = cv2.getTrackbarPos("V_min", "controls")
    wrap = cv2.getTrackbarPos("red_wrap", "controls")
    if wrap:   # red: hue near 0 AND near 180
        return [((0, s, v), (hh, 255, 255)), ((hl, s, v), (180, 255, 255))]
    return [((hl, s, v), (hh, 255, 255))]


def detect(frame, ranges):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    if r < MIN_RADIUS_PX:
        return None, mask
    return (x, y, r), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video file path, or 0 for webcam")
    args = ap.parse_args()
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    cv2.namedWindow("controls")
    cv2.createTrackbar("H_lo", "controls", 170, 180, nothing)
    cv2.createTrackbar("H_hi", "controls", 10, 180, nothing)
    cv2.createTrackbar("S_min", "controls", 120, 255, nothing)
    cv2.createTrackbar("V_min", "controls", 70, 255, nothing)
    cv2.createTrackbar("red_wrap", "controls", 1, 1, nothing)

    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, MAX_YAW_RATE)
    pid_z = PID(KP_Z, KI_Z, KD_Z, MAX_VZ)
    paused, n, frame = False, 0, None

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:                                   # end of video -> loop
                if args.source.isdigit():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pid_yaw.reset()
                pid_z.reset()
                continue
            frame = cv2.resize(frame, (W, H))
            n += 1

        view = frame.copy()
        det, mask = detect(frame, get_ranges())
        if det is not None:
            cx, cy, r = det
            ex, ey = cx - W / 2, cy - H / 2                       # pixels
            ax, ay = math.atan2(ex, F_PX), math.atan2(ey, F_PX)   # radians
            yaw_rate = pid_yaw.update(ax, dt)
            vz = pid_z.update(ay, dt)
            cv2.circle(view, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            txt = (f"err {ex:+.0f},{ey:+.0f}px  ang {math.degrees(ax):+.1f},{math.degrees(ay):+.1f}deg  "
                   f"yaw {math.degrees(yaw_rate):+.1f}deg/s  vz {vz:+.2f}m/s")
            if n % 10 == 0:
                print(txt)
        else:
            pid_yaw.reset()
            pid_z.reset()
            txt = "NO TARGET -> hover (0, 0, 0)"
        cv2.drawMarker(view, (W // 2, H // 2), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(view, txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("tracking", view)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1 if paused else int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            paused = not paused

    print("\nPaste into HSV_RANGES in the main script:")
    print(get_ranges())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
balloon_tracker_mavsdk.py
Interceptor Drone Project — IIT Indore Aeromodelling Club
Mode A: HSV balloon tracking + PX4 via MAVSDK

Hardware: Pixhawk (PX4 firmware) + Raspberry Pi + RPi Camera
Run:  python balloon_tracker_mavsdk.py
SITL: python balloon_tracker_mavsdk.py --sitl   (uses webcam instead of picamera2)

Dependencies:
    pip install mavsdk opencv-python numpy
    (on RPi also: pip install picamera2)
"""

import asyncio
import math
import argparse
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ─────────────────────────────────────────────
#  CONFIG  — edit these values
# ─────────────────────────────────────────────

CONN          = "udp://:14540"   # PX4 SITL default port
                                 # real Pixhawk via USB: "serial:///dev/ttyUSB0:57600"

TAKEOFF_ALT   = 2.0              # metres

# Camera
W, H          = 640, 480
HFOV_DEG      = 62.2             # RPi camera v2 horizontal FOV
F_PX          = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))   # ≈ 530 px

# HSV ranges for RED balloon (two ranges because red wraps around hue axis)
HSV_RANGES = [
    ((0,   120, 70), (10,  255, 255)),   # low-hue red
    ((170, 120, 70), (180, 255, 255)),   # high-hue red
]

# PID gains  — tuned values from video_test.py go here
KP_YAW, KI_YAW, KD_YAW = 1.5, 0.0, 0.10
KP_Z,   KI_Z,   KD_Z   = 2.0, 0.0, 0.10

MAX_YAW_RATE  = 1.0    # rad/s   output clamp
MAX_VZ        = 0.5    # m/s     output clamp (positive = DOWN in PX4 FRD)
FWD_SPEED     = 0.3    # m/s     constant forward speed while tracking
I_LIMIT       = 0.3    # anti-windup: integral clamp

# ─────────────────────────────────────────────
#  PID CONTROLLER
# ─────────────────────────────────────────────

class PID:
    """
    Single-axis PID with anti-windup and output saturation.

    Convention: positive error → positive output.
    For yaw:    error = ang_x (radians right of centre)  → positive = turn right
    For vz:     error = ang_y (radians below centre)     → positive = descend
    """
    def __init__(self, kp, ki, kd, out_limit, i_limit=0.3, dt=0.033):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit   = i_limit
        self.dt        = dt
        self._integral  = 0.0
        self._prev_err  = 0.0

    def update(self, error):
        self._integral += error * self.dt
        self._integral  = max(-self.i_limit, min(self.i_limit, self._integral))

        derivative = (error - self._prev_err) / self.dt
        self._prev_err = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.out_limit, min(self.out_limit, output))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

# ─────────────────────────────────────────────
#  VISION
# ─────────────────────────────────────────────

def detect_balloon(frame):
    """
    Detect red balloon using HSV thresholding.

    Returns (cx, cy, radius) in pixels, or None if not found.
    cx, cy are measured from top-left (OpenCV convention).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # combine both red hue ranges
    mask = None
    for (lo, hi) in HSV_RANGES:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # morphological close: fills light-reflection holes, removes small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:   # ignore tiny blobs
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return (int(cx), int(cy), radius)


def pixel_to_angles(cx, cy):
    """
    Convert pixel (cx, cy) to angular error (ang_x, ang_y) in radians.

    ang_x > 0 → target is to the RIGHT  of centre → turn right
    ang_y > 0 → target is BELOW centre  → descend (vz positive = down in FRD)

    Uses pinhole model: ang = atan2(pixel_offset, focal_length_px)
    F_PX ≈ 530 px for 62.2° HFOV at 640 px wide.
    """
    ex = cx - W / 2
    ey = cy - H / 2
    ang_x = math.atan2(ex, F_PX)
    ang_y = math.atan2(ey, F_PX)
    return ang_x, ang_y

# ─────────────────────────────────────────────
#  CAMERA SETUP
# ─────────────────────────────────────────────

def open_camera(sitl_mode):
    """
    Returns a frame-grabbing function: grab() → numpy BGR array or None.

    SITL mode: uses OpenCV webcam (index 0) — for testing on laptop.
    Real RPi:  uses picamera2 for the CSI ribbon-cable camera.
    """
    if sitl_mode:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        def grab():
            ret, frame = cap.read()
            return frame if ret else None
        def release():
            cap.release()
    else:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"size": (W, H), "format": "BGR888"}
        ))
        cam.start()
        def grab():
            return cam.capture_array()
        def release():
            cam.stop()

    return grab, release

# ─────────────────────────────────────────────
#  FLIGHT
# ─────────────────────────────────────────────

async def wait_for_connection(drone):
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✓ Drone connected")
            break


async def arm_and_takeoff(drone):
    print("Arming...")
    await drone.action.arm()
    print(f"Taking off to {TAKEOFF_ALT} m...")
    await drone.action.takeoff()
    await asyncio.sleep(5)   # wait for takeoff to complete
    print("✓ Airborne")


async def start_offboard(drone):
    """
    PX4 requires at least one setpoint to be sent before offboard mode starts.
    Send a zero setpoint first, then switch to offboard.
    """
    print("Starting offboard mode...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
        print("✓ Offboard mode active")
    except OffboardError as e:
        print(f"✗ Offboard failed: {e._result.result}")
        raise


async def send_velocity(drone, vx, vy, vz, yaw_rate_rad):
    """
    Send body-frame velocity command.
    PX4 FRD frame: vx = forward, vy = right, vz = DOWN (positive = descend)
    MAVSDK VelocityBodyYawspeed: (forward, right, down, yawspeed_deg_s)
    """
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(vx, vy, vz, math.degrees(yaw_rate_rad))
    )

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

async def run(sitl_mode):
    # ── Connect ──
    drone = System()
    await drone.connect(system_address=CONN)
    await wait_for_connection(drone)

    # ── Arm + takeoff ──
    await arm_and_takeoff(drone)

    # ── Offboard ──
    await start_offboard(drone)

    # ── Camera ──
    grab, release_cam = open_camera(sitl_mode)

    # ── PID controllers ──
    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, MAX_YAW_RATE, I_LIMIT)
    pid_z   = PID(KP_Z,   KI_Z,   KD_Z,   MAX_VZ,       I_LIMIT)

    frame_n = 0
    print("\nTracking started. Press q to quit and land.\n")

    try:
        while True:
            frame = grab()
            if frame is None:
                print("Camera read failed — stopping")
                break

            det = detect_balloon(frame)

            if det is not None:
                cx, cy, r = det
                ang_x, ang_y = pixel_to_angles(cx, cy)

                yaw_rate = pid_yaw.update(ang_x)
                vz       = pid_z.update(ang_y)

                await send_velocity(drone, FWD_SPEED, 0.0, vz, yaw_rate)

                # draw overlay
                cv2.circle(frame, (cx, cy), int(r), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                if frame_n % 10 == 0:
                    print(
                        f"cx={cx:3d} cy={cy:3d}  r={r:5.1f}px  "
                        f"ang_x={math.degrees(ang_x):+5.1f}°  "
                        f"yaw={math.degrees(yaw_rate):+5.2f}°/s  "
                        f"vz={vz:+.2f}m/s"
                    )
            else:
                # Lost target → hover, reset integrators
                await send_velocity(drone, 0.0, 0.0, 0.0, 0.0)
                pid_yaw.reset()
                pid_z.reset()

                if frame_n % 30 == 0:
                    print("Target lost — hovering")

            cv2.imshow("Balloon Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested")
                break

            frame_n += 1
            await asyncio.sleep(0.033)   # ~30 Hz control loop

    finally:
        print("\nShutting down...")
        release_cam()
        cv2.destroyAllWindows()
        try:
            await drone.offboard.stop()
        except Exception:
            pass
        await drone.action.land()
        print("Landing.")

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balloon Tracker — PX4 MAVSDK")
    parser.add_argument(
        "--sitl", action="store_true",
        help="SITL mode: use webcam instead of RPi camera"
    )
    args = parser.parse_args()

    if args.sitl:
        print("SITL mode — using webcam (index 0)")
    else:
        print("Hardware mode — using RPi camera")

    asyncio.run(run(args.sitl))

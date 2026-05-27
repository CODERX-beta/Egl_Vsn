#!/usr/bin/env python3
import os
os.environ.setdefault('DISPLAY', ':0')

import math
import time
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

try:
    from pyzbar import pyzbar as _pyzbar
    PYZBAR_OK = True
except ImportError:
    PYZBAR_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

QR_PHYSICAL_SIZE = 0.50 #in meters

# The drone hovers ~0.2m higher than commanded due to motor/physics offset,
# so we subtract 0.2 from the target to compensate.
MINIMUM_HEIGHT        = 1.8
MINIMUM_HEIGHT_TARGET = MINIMUM_HEIGHT - 0.2   # what we actually command

# HOVER_THRUST: tune this until drone hovers in place with no QR visible
HOVER_THRUST   = 0.1

IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480
HFOV_DEG     = 62.1

SHOW_WINDOW  = True

_fx = (IMAGE_WIDTH  / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
_fy = _fx
_cx = IMAGE_WIDTH  / 2.0
_cy = IMAGE_HEIGHT / 2.0
CAMERA_MATRIX = np.array([[_fx, 0, _cx],
                           [0, _fy, _cy],
                           [0,   0,   1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

_h = QR_PHYSICAL_SIZE / 2.0
OBJ_POINTS = np.array([
    [-_h,  _h, 0], [ _h,  _h, 0],
    [ _h, -_h, 0], [-_h, -_h, 0],
], dtype=np.float32)

# ── Gains ─────────────────────────────────────────────────────────────────────
GAIN_HEIGHT   = 0.6    # linear.z correction for height
GAIN_CENTRE_X = 0.9    # lateral (linear.y)
GAIN_CENTRE_Y = 0.9    # fore/aft (linear.x)
GAIN_YAW      = 0.6    # angular.z

# ── Dead-bands ────────────────────────────────────────────────────────────────
DEAD_HEIGHT_M = 0.08
DEAD_X_PX     = 14
DEAD_Y_PX     = 14
DEAD_YAW_DEG  = 3.0

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_HEIGHT_CORR = 0.4
MAX_FORWARD     = 0.5
MAX_STRAFE      = 0.5
MAX_YAW         = 0.4

# ── Smoothing / lock ──────────────────────────────────────────────────────────
ALPHA_SMOOTH          = 0.35
LOCK_HOLD_FRAMES      = 20
ROI_PAD               = 0.45
LOST_TARGET_TIMEOUT_S = 1.2

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _qr_rotation_error_deg(pts):
    tl, tr = pts[0], pts[1]
    a = math.degrees(math.atan2(tr[1]-tl[1], tr[0]-tl[0]))
    while a >  45: a -= 90
    while a < -45: a += 90
    return a

def _find_qr_shapes(gray, min_area=400):
    results = []
    for preproc in [
        gray,
        cv2.GaussianBlur(gray, (3,3), 0),
        cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray),
    ]:
        binary = cv2.adaptiveThreshold(
            preproc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=11, C=4)
        for src in [binary, cv2.Canny(preproc, 30, 100)]:
            for cnt in cv2.findContours(
                    src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if area < min_area: continue
                peri = cv2.arcLength(cnt, True)
                for eps in [0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(cnt, eps*peri, True)
                    if len(approx)==4 and cv2.isContourConvex(approx):
                        pts   = approx.reshape(4,2).astype(np.float32)
                        sides = [np.linalg.norm(pts[(i+1)%4]-pts[i])
                                 for i in range(4)]
                        if max(sides)/(min(sides)+1e-6) <= 2.2:
                            results.append((area, pts))
                            break
    if not results: return []
    results.sort(key=lambda x: -x[0])
    deduped = []
    for area, pts in results:
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        if not any(math.hypot(cx-p[:,0].mean(), cy-p[:,1].mean()) < 30
                   for _, p in deduped):
            deduped.append((area, pts))
    return [p for _, p in deduped]

def _order_corners(pts):
    pts  = pts.reshape(4,2)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]],
                    dtype=np.float32)

def _try_decode(gray, detector):
    ok, info, _, _ = detector.detectAndDecodeMulti(gray)
    if ok and info:
        for s in info:
            if s: return s
    if PYZBAR_OK:
        for obj in _pyzbar.decode(gray):
            if obj.data: return obj.data.decode(errors='replace')
    return ''

def deadband(v, t): return 0.0 if abs(v) < t else float(v)
def smooth(p, c, a): return a*p + (1.0-a)*c

def _padded_roi(pts, pad, w, h):
    x1,y1 = pts.min(axis=0); x2,y2 = pts.max(axis=0)
    pw,ph  = (x2-x1)*pad, (y2-y1)*pad
    return (max(0,int(x1-pw)), max(0,int(y1-ph)),
            min(w,int(x2+pw)), min(h,int(y2+ph)))

def _draw_box(frame, pts, locked):
    col = (0,255,0) if locked else (0,200,255)
    pi  = pts.astype(np.int32).reshape(4,2)
    for i in range(4):
        cv2.line(frame, tuple(pi[i]), tuple(pi[(i+1)%4]), col, 2)
    cen = pi.mean(axis=0).astype(np.int32)
    for pt in pi:
        d = cen-pt; n = np.linalg.norm(d)
        if n < 1: continue
        cv2.line(frame, tuple(pt),
                 tuple((pt+d/n*18).astype(np.int32)), col, 3)
        cv2.circle(frame, tuple(pt), 5, col, -1)

# ─────────────────────────────────────────────────────────────────────────────
# NODE(ROS2)
# ─────────────────────────────────────────────────────────────────────────────

class DroneQRTrackerDownward(Node):

    def __init__(self):
        super().__init__('qr_tracker')

        self._svx = 0.0
        self._svy = 0.0
        self._svz = 0.0
        self._swz = 0.0

        self._last_detect_t = time.monotonic()
        self._locked_pts    = None
        self._locked_roi    = None
        self._miss_frames   = 0

        # Stabilisation detector
        self._stable_frames = 0
        self._is_stable     = False
        self._STABLE_FRAMES = 30      # consecutive frames needed (~1s at 30Hz)
        self._STABLE_H_M    = 0.15    # height error tolerance (metres)
        self._STABLE_PX     = 25      # centre error tolerance (pixels)
        self._STABLE_ROT    = 5.0     # rotation tolerance (degrees)

        self.bridge      = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()

        self.cmd_pub = self.create_publisher(
            Twist, '/model/parrot_bebop_2/cmd_vel', 10)
        self.debug_pub = self.create_publisher(
            Image, '/bebop/debug_image', 10)
        self.image_sub = self.create_subscription(
            Image,
            '/world/empty/model/parrot_bebop_2/link/front_camera_link'
            '/sensor/front_camera/image',
            self._cb, 10)

        self.get_logger().info('=' * 58)
        self.get_logger().info('  Bebop 2 QR Tracker — Downward Camera')
        self.get_logger().info(f'  HOVER_THRUST   : {HOVER_THRUST}')
        self.get_logger().info(f'  MINIMUM_HEIGHT : {MINIMUM_HEIGHT} m')
        self.get_logger().info('=' * 58)

    def _cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(str(e)); return

        fh, fw = frame.shape[:2]
        ccx, ccy = fw//2, fh//2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detection ──────────────────────────────────────────────────────
        candidates = _find_qr_shapes(gray)
        img_pts = None
        if candidates:
            if self._locked_pts is not None:
                lc = self._locked_pts.mean(axis=0)
                candidates.sort(
                    key=lambda p: np.linalg.norm(p.mean(axis=0)-lc))
            img_pts = _order_corners(candidates[0])

        if img_pts is not None:
            self._locked_pts  = img_pts
            self._locked_roi  = _padded_roi(img_pts, ROI_PAD, fw, fh)
            self._miss_frames = 0
            status = 'LOCKED'
        else:
            self._miss_frames += 1
            if self._miss_frames > LOCK_HOLD_FRAMES:
                self._locked_pts = None
                self._locked_roi = None
            status = 'SEARCHING' if self._locked_pts is None else 'COASTING'

        twist    = Twist()
        qr_found = False
        height   = 0.0

        if img_pts is not None:
            qr_cx = float(img_pts[:,0].mean())
            qr_cy = float(img_pts[:,1].mean())
            self._last_detect_t = time.monotonic()
            qr_found = True

            # PnP — height = tvec.z when camera faces straight down
            ok, _, tvec = cv2.solvePnP(
                OBJ_POINTS, img_pts, CAMERA_MATRIX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                height = abs(float(tvec[2][0]))

            # ── 1. Height — maintain minimum ───────────────────────────
            # Use MINIMUM_HEIGHT_TARGET (= MINIMUM_HEIGHT - 0.2) to
            # compensate for the ~0.2m hover offset in the physics model.
            height_err = MINIMUM_HEIGHT_TARGET - height  # + too low, - too high
            height_err = deadband(height_err, DEAD_HEIGHT_M)
            raw_vz = float(np.clip(
                height_err * GAIN_HEIGHT, -MAX_HEIGHT_CORR, MAX_HEIGHT_CORR))

            # ── 2. Centre fore/aft — linear.x ─────────────────────────
            # QR below centre in image (positive py error) → drone moves back
            # QR above centre in image (negative py error) → drone moves forward
            px_y   = deadband(qr_cy - ccy, DEAD_Y_PX)
            raw_vx = float(np.clip(
                -(px_y / ccy) * GAIN_CENTRE_Y, -MAX_FORWARD, MAX_FORWARD))

            # ── 3. Centre lateral — linear.y ──────────────────────────
            # QR right of centre (positive px error) → drone strafes left
            # QR left of centre  (negative px error) → drone strafes right
            px_x   = deadband(qr_cx - ccx, DEAD_X_PX)
            raw_vy = float(np.clip(
                -(px_x / ccx) * GAIN_CENTRE_X, -MAX_STRAFE, MAX_STRAFE))

            # ── 4. Yaw alignment ───────────────────────────────────────
            rot_err = deadband(_qr_rotation_error_deg(img_pts), DEAD_YAW_DEG)
            raw_wz  = float(np.clip(
                -math.radians(rot_err) * GAIN_YAW, -MAX_YAW, MAX_YAW))

            self._svx = smooth(self._svx, raw_vx, ALPHA_SMOOTH)
            self._svy = smooth(self._svy, raw_vy, ALPHA_SMOOTH)
            self._svz = smooth(self._svz, raw_vz, ALPHA_SMOOTH)
            self._swz = smooth(self._swz, raw_wz, ALPHA_SMOOTH)

            twist.linear.x  = self._svx
            twist.linear.y  = self._svy
            twist.linear.z  = HOVER_THRUST + self._svz
            twist.angular.z = self._swz

            # ── Stabilisation detector ─────────────────────────────────
            cx_err  = abs(qr_cx - ccx)
            cy_err  = abs(qr_cy - ccy)
            h_err   = abs(height - MINIMUM_HEIGHT)
            r_err   = abs(_qr_rotation_error_deg(img_pts))
            stable_now = (cx_err  < self._STABLE_PX  and
                          cy_err  < self._STABLE_PX  and
                          h_err   < self._STABLE_H_M and
                          r_err   < self._STABLE_ROT)

            if stable_now:
                self._stable_frames += 1
            else:
                self._stable_frames = 0
                if self._is_stable:
                    self._is_stable = False
                    self.get_logger().info('  ↕  Stability lost — reacquiring')

            if self._stable_frames >= self._STABLE_FRAMES and not self._is_stable:
                self._is_stable = True
                # ── Extract the data the moment stability is achieved ──

                self.get_logger().info('★ ═══════════════════════════════ ★')
                self.get_logger().info('★   DRONE STABILISED OVER QR CODE  ★')
                self.get_logger().info('★ ═══════════════════════════════ ★')
                
            if self._is_stable:
                self.get_logger().info(
                    f'★ STABLE  h={height:.2f}m  '
                    f'cx={qr_cx-ccx:+.0f}px  cy={qr_cy-ccy:+.0f}px  '
                    f'rot={r_err:.1f}°',
                    throttle_duration_sec=0.5)

            self.get_logger().info(
                f'[{status}] h={height:.2f}m '
                f'cx={qr_cx-ccx:+.0f}px cy={qr_cy-ccy:+.0f}px '
                f'vx={twist.linear.x:+.3f} vy={twist.linear.y:+.3f} '
                f'vz={twist.linear.z:+.3f} wz={twist.angular.z:+.3f}',
                throttle_duration_sec=0.25)

            _draw_box(frame, img_pts, locked=True)
            cv2.arrowedLine(frame, (ccx,ccy), (int(qr_cx),int(qr_cy)),
                            (0,80,255), 2, tipLength=0.15)
            label = _try_decode(gray, self.qr_detector) or 'QR Target'
            self._draw_hud(frame, qr_cx, qr_cy, ccx, ccy,
                           height, rot_err, twist, label)

        # ── No detection ───────────────────────────────────────────────────
        if not qr_found:
            elapsed = time.monotonic() - self._last_detect_t
            if not (elapsed <= LOST_TARGET_TIMEOUT_S and status == 'COASTING'):
                self._svx *= 0.85; self._svy *= 0.85
                self._svz *= 0.85; self._swz *= 0.85
            if self._is_stable:
                self._is_stable     = False
                self._stable_frames = 0
                self.get_logger().info('  ↕  QR lost — stability reset')
            twist.linear.x  = self._svx
            twist.linear.y  = self._svy
            twist.linear.z  = HOVER_THRUST + self._svz
            twist.angular.z = self._swz
            self.get_logger().warn(
                f'[{status}] lost {time.monotonic()-self._last_detect_t:.1f}s',
                throttle_duration_sec=1.0)

        self.cmd_pub.publish(twist)

        # ── Visuals ────────────────────────────────────────────────────────
        if self._locked_roi:
            x1,y1,x2,y2 = self._locked_roi
            cv2.rectangle(frame,(x1,y1),(x2,y2),(160,60,255),1)
        cv2.line(frame,(ccx-20,ccy),(ccx+20,ccy),(255,200,0),1)
        cv2.line(frame,(ccx,ccy-20),(ccx,ccy+20),(255,200,0),1)
        cv2.circle(frame,(ccx,ccy),4,(255,200,0),-1)
        s_col = ((0,255,80) if status=='LOCKED'
                 else (255,150,0) if status=='COASTING'
                 else (0,100,255))
        cv2.putText(frame, status, (8,fh-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 1, cv2.LINE_AA)

        if SHOW_WINDOW:
            cv2.imshow('Bebop 2 — Downward QR Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows(); rclpy.shutdown(); sys.exit(0)

        out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out.header = msg.header
        self.debug_pub.publish(out)

    def _draw_hud(self, frame, qr_cx, qr_cy, ccx, ccy,
                  height, rot_err, twist, label):
        cx_err = qr_cx - ccx
        cy_err = qr_cy - ccy
        h_col  = (0,255,80) if height >= MINIMUM_HEIGHT else (0,50,255)
        stable_str = '  ★ STABLE' if self._is_stable else ''
        lines  = [
            (f'{label[:40]}',                                            (0,255,80)),
            (f'Height {height:.2f}m  min {MINIMUM_HEIGHT}m{stable_str}',h_col),
            (f'Centre  X:{cx_err:+.0f}px  Y:{cy_err:+.0f}px',
             (0,255,80) if abs(cx_err)<20 and abs(cy_err)<20 else (0,150,255)),
            (f'Rot {rot_err:+.1f}deg',                               (200,200,255)),
            (f'vx={twist.linear.x:+.3f} vy={twist.linear.y:+.3f} '
             f'vz={twist.linear.z:+.3f} wz={twist.angular.z:+.3f}', (180,180,180)),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (8, 20+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


def main():
    rclpy.init()
    node = DroneQRTrackerDownward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutdown — zeroing motors')
        node.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

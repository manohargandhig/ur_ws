#!/usr/bin/env python3
"""
multi_object_perception_node.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detects all 6 bin objects using HSV colour + depth segmentation,
then publishes per-object TF frames consumed by pick_place_vision.cpp

Published TF frames (in base_link):
  • {object_name}_grasp_frame      ← exact grasp pose
  • {object_name}_pregrasp_frame   ← approach pose above grasp

  • object_grasp_frame             ← alias for CURRENT target object
  • object_pregrasp_frame          ← alias for CURRENT target object

ROS2 Parameters:
  config_file   : path to object_color_profiles.yaml
  debug_viz     : publish annotated image (default: true)
  pick_order    : list of object names in order to pick
                  default: all detected, left-to-right
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cv2
import numpy as np
import yaml
import os
import math

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber


class DetectedObject:
    """Holds detection result for one object."""
    def __init__(self, name: str, cx_px: int, cy_px: int,
                 x_m: float, y_m: float, z_m: float,
                 angle_rad: float, area: float):
        self.name = name
        self.cx_px = cx_px
        self.cy_px = cy_px
        self.x_m = x_m          # in base_link frame
        self.y_m = y_m
        self.z_m = z_m
        self.angle_rad = angle_rad   # yaw from PCA
        self.area = area


class MultiObjectPerceptionNode(Node):

    def __init__(self):
        super().__init__('multi_object_perception_node')

        # ── Parameters ────────────────────────────────────────────────
        default_cfg = os.path.join(
            os.path.dirname(__file__), '..', 'config',
            'object_color_profiles.yaml')
        self.declare_parameter('config_file', default_cfg)
        self.declare_parameter('debug_viz', True)
        self.declare_parameter('pick_order', '')   # comma-separated string, empty = auto

        cfg_path = self.get_parameter('config_file').value
        self.debug_viz = self.get_parameter('debug_viz').value

        # ── Load colour profiles ───────────────────────────────────────
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.object_profiles = cfg['objects']
        cam_cfg = cfg['camera']
        self.optical_frame = cam_cfg['optical_frame']   # camera_optical_link
        self.base_frame    = cam_cfg['base_frame']       # base_link

        self.get_logger().info(
            f'Loaded profiles for: {list(self.object_profiles.keys())}')

        # ── TF ────────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(
            self.tf_buffer, self)

        # ── Camera intrinsics (filled on first CameraInfo msg) ────────
        self.K  = None     # 3×3 intrinsic matrix
        self.fx = self.fy = self.cx = self.cy = None

        # ── Bridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── State ─────────────────────────────────────────────────────
        self.detected_objects: list[DetectedObject] = []
        self.current_target_idx = 0

        # ── Publishers ────────────────────────────────────────────────
        self.viz_pub = self.create_publisher(Image, '/perception/viz', 10)
        self.status_pub = self.create_publisher(
            String, '/perception/current_target', 10)

        # ── Subscribers ───────────────────────────────────────────────
        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            cam_cfg['camera_info_topic'],
            self._cam_info_cb, 1)

        color_sub = Subscriber(self, Image,
                               cam_cfg['color_topic'],
                               qos_profile=qos)
        depth_sub = Subscriber(self, Image,
                               cam_cfg['depth_topic'],
                               qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=5, slop=0.05)
        self.sync.registerCallback(self._image_pair_cb)

        # ── Timer: republish TF at 10 Hz even between detections ──────
        self.create_timer(0.1, self._republish_tf)

        self.get_logger().info('MultiObjectPerceptionNode ready.')

    # ──────────────────────────────────────────────────────────────────
    # Camera Intrinsics
    # ──────────────────────────────────────────────────────────────────
    def _cam_info_cb(self, msg: CameraInfo):
        if self.K is None:
            K = np.array(msg.k).reshape(3, 3)
            self.K  = K
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.get_logger().info(
                f'Camera intrinsics received: '
                f'fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    # ──────────────────────────────────────────────────────────────────
    # Main Callback: colour image + depth image
    # ──────────────────────────────────────────────────────────────────
    def _image_pair_cb(self, color_msg: Image, depth_msg: Image):
        if self.K is None:
            return  # wait for intrinsics

        # Convert to OpenCV
        color_bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_raw = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough')

        # Depth in metres (Gazebo depth camera publishes float32 in metres)
        if depth_raw.dtype == np.uint16:
            depth_m = depth_raw.astype(np.float32) / 1000.0
        else:
            depth_m = depth_raw.astype(np.float32)

        color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

        # ── Detect each object ────────────────────────────────────────
        detected = []
        viz_image = color_bgr.copy() if self.debug_viz else None

        for obj_name, profile in self.object_profiles.items():
            result = self._detect_single_object(
                obj_name, profile, color_hsv, depth_m,
                color_msg.header.stamp)
            if result is not None:
                detected.append(result)
                if viz_image is not None:
                    self._draw_detection(viz_image, result, profile)

        # Sort detections left-to-right (x pixel) for deterministic pick order
        detected.sort(key=lambda d: d.cx_px)
        self.detected_objects = detected

        # ── Publish TF for all detected objects ───────────────────────
        for obj in detected:
            self._publish_object_tf(obj, color_msg.header.stamp)

        # ── Publish alias frames for current target ───────────────────
        if detected:
            target = self._get_current_target(detected)
            if target:
                self._publish_alias_tf(target, color_msg.header.stamp)
                msg = String()
                msg.data = target.name
                self.status_pub.publish(msg)

        # ── Debug visualisation ───────────────────────────────────────
        if self.debug_viz and viz_image is not None:
            if detected:
                target = self._get_current_target(detected)
                if target:
                    cv2.putText(
                        viz_image,
                        f'TARGET: {target.name}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
            self.viz_pub.publish(
                self.bridge.cv2_to_imgmsg(viz_image, 'bgr8'))

    # ──────────────────────────────────────────────────────────────────
    # Single Object Detection
    # ──────────────────────────────────────────────────────────────────
    def _detect_single_object(
            self, name: str, profile: dict,
            hsv: np.ndarray, depth_m: np.ndarray,
            stamp) -> DetectedObject | None:

        lo = np.array(profile['hsv_lower'], dtype=np.uint8)
        hi = np.array(profile['hsv_upper'], dtype=np.uint8)

        # Handle hue wrap-around (e.g. red spans 170-10)
        if lo[0] > hi[0]:
            mask1 = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]]))
            mask2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, lo, hi)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < profile.get('min_area', 300):
            return None

        # Centroid
        M = cv2.moments(best)
        if M['m00'] == 0:
            return None
        cx_px = int(M['m10'] / M['m00'])
        cy_px = int(M['m01'] / M['m00'])

        # Orientation from PCA on contour points
        angle_rad = self._pca_angle(best)

        # Depth: median in a small ROI around centroid
        roi_half = 8
        h, w = depth_m.shape
        r0 = max(0, cy_px - roi_half)
        r1 = min(h, cy_px + roi_half)
        c0 = max(0, cx_px - roi_half)
        c1 = min(w, cx_px + roi_half)
        roi_depth = depth_m[r0:r1, c0:c1]
        valid = roi_depth[(roi_depth > 0.05) & (roi_depth < 4.0)]
        if len(valid) < 5:
            return None
        Z_cam = float(np.median(valid))

        # Back-project to 3D in camera_optical_link frame
        X_cam = (cx_px - self.cx) * Z_cam / self.fx
        Y_cam = (cy_px - self.cy) * Z_cam / self.fy

        # Transform to base_link
        pt_base = self._transform_point_to_base(X_cam, Y_cam, Z_cam, stamp)
        if pt_base is None:
            return None

        x_b, y_b, z_b = pt_base

        # Apply grasp z-offset (top surface → grasp point)
        z_b += profile.get('grasp_z_offset', 0.02)

        return DetectedObject(
            name=name,
            cx_px=cx_px, cy_px=cy_px,
            x_m=x_b, y_m=y_b, z_m=z_b,
            angle_rad=angle_rad,
            area=area)

    # ──────────────────────────────────────────────────────────────────
    # PCA-based orientation angle
    # ──────────────────────────────────────────────────────────────────
    def _pca_angle(self, contour: np.ndarray) -> float:
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return 0.0
        mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
        angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
        return angle

    # ──────────────────────────────────────────────────────────────────
    # Transform point from camera_optical_link → base_link
    # ──────────────────────────────────────────────────────────────────
    def _transform_point_to_base(
            self, X: float, Y: float, Z: float, stamp) -> tuple | None:
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,          # target: base_link
                self.optical_frame,       # source: camera_optical_link
                rclpy.time.Time())        # latest available
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return None

        # Apply transform manually (avoids PointStamped import chain)
        tx = t.transform.translation
        rq = t.transform.rotation

        # Quaternion → rotation matrix
        R = self._quat_to_matrix(rq.x, rq.y, rq.z, rq.w)
        p_cam = np.array([X, Y, Z])
        p_base = R @ p_cam + np.array([tx.x, tx.y, tx.z])
        return float(p_base[0]), float(p_base[1]), float(p_base[2])

    @staticmethod
    def _quat_to_matrix(x, y, z, w) -> np.ndarray:
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])

    # ──────────────────────────────────────────────────────────────────
    # Publish per-object TF frames
    # ──────────────────────────────────────────────────────────────────
    def _publish_object_tf(self, obj: DetectedObject, stamp):
        profile = self.object_profiles[obj.name]
        pregrasp_dz = profile.get('pregrasp_z_offset', 0.15)

        # Grasp orientation: top-down, gripper Z pointing down
        # In base_link: we want gripper Z (-Z of tool0) pointing -Z (downward)
        # Rotation: 180° around X axis → quat (1,0,0,0) rotated
        # For a top-down approach from above: qx=1, qy=0, qz=0, qw=0
        qx, qy, qz, qw = self._yaw_to_topdown_quat(obj.angle_rad)

        now = self.get_clock().now().to_msg()

        # ── Grasp frame ──────────────────────────────────────────────
        tf_g = TransformStamped()
        tf_g.header.stamp = now
        tf_g.header.frame_id = self.base_frame
        tf_g.child_frame_id  = f'{obj.name}_grasp_frame'
        tf_g.transform.translation.x = obj.x_m
        tf_g.transform.translation.y = obj.y_m
        tf_g.transform.translation.z = obj.z_m
        tf_g.transform.rotation.x = qx
        tf_g.transform.rotation.y = qy
        tf_g.transform.rotation.z = qz
        tf_g.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_g)

        # ── Pre-grasp frame (directly above grasp) ───────────────────
        tf_p = TransformStamped()
        tf_p.header.stamp = now
        tf_p.header.frame_id = self.base_frame
        tf_p.child_frame_id  = f'{obj.name}_pregrasp_frame'
        tf_p.transform.translation.x = obj.x_m
        tf_p.transform.translation.y = obj.y_m
        tf_p.transform.translation.z = obj.z_m + pregrasp_dz
        tf_p.transform.rotation.x = qx
        tf_p.transform.rotation.y = qy
        tf_p.transform.rotation.z = qz
        tf_p.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_p)

    def _publish_alias_tf(self, obj: DetectedObject, stamp):
        """Publish object_grasp_frame & object_pregrasp_frame for
        backward-compatibility with pick_place_vision.cpp"""
        profile = self.object_profiles[obj.name]
        pregrasp_dz = profile.get('pregrasp_z_offset', 0.15)
        qx, qy, qz, qw = self._yaw_to_topdown_quat(obj.angle_rad)
        now = self.get_clock().now().to_msg()

        for child, dz in [('object_grasp_frame', 0.0),
                           ('object_pregrasp_frame', pregrasp_dz)]:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.base_frame
            tf.child_frame_id  = child
            tf.transform.translation.x = obj.x_m
            tf.transform.translation.y = obj.y_m
            tf.transform.translation.z = obj.z_m + dz
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def _republish_tf(self):
        """Re-broadcast last known TFs at 10 Hz so move_group doesn't
        time out between image callbacks."""
        if not self.detected_objects:
            return
        fake_stamp = self.get_clock().now().to_msg()
        for obj in self.detected_objects:
            self._publish_object_tf(obj, fake_stamp)
        target = self._get_current_target(self.detected_objects)
        if target:
            self._publish_alias_tf(target, fake_stamp)

    # ──────────────────────────────────────────────────────────────────
    # Grasp orientation helper
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _yaw_to_topdown_quat(yaw: float) -> tuple:
        """
        Top-down grasp orientation in base_link:
          - Gripper Z pointing straight down (-Z of base_link)
          - Gripper yaw aligned with object major axis

        Base rotation for top-down: 180° around X → (qx=1, qy=0, qz=0, qw=0)
        Then apply yaw around Z.

        Combined: q = q_yaw * q_flip
        """
        # q_flip = (1, 0, 0, 0)  [180° around X]
        # q_yaw  = (0, 0, sin(yaw/2), cos(yaw/2))
        half_yaw = yaw / 2.0
        sy = math.sin(half_yaw)
        cy = math.cos(half_yaw)

        # Quaternion multiplication: q_yaw * q_flip
        # q_flip = (qx=1, qy=0, qz=0, qw=0)
        # q_yaw  = (qx=0, qy=0, qz=sy, qw=cy)
        # result = q_yaw ⊗ q_flip
        rx = cy * 1.0 + sy * 0.0  # qw_y*qx_f + qz_y*qy_f  ... full formula:
        ry = cy * 0.0 - sy * 1.0
        rz = cy * 0.0 + sy * 0.0
        rw = cy * 0.0 - sy * 0.0

        # Simplified (q_flip = pure-x unit quat):
        # result.x =  cy
        # result.y = -sy
        # result.z =  0
        # result.w =  0   → this gives zero norm, use proper formula
        qx = cy * 1.0 + 0.0
        qy = cy * 0.0 - sy * 1.0 * 0 + 0.0  # simplify step by step
        # Correct: q = q_yaw ⊗ q_flip
        # q_yaw = [0, 0, sy, cy]   (x,y,z,w)
        # q_flip= [1, 0, 0,  0]
        # product (x,y,z,w):
        rx = cy*1 + 0*0  + sy*0 - 0*0
        ry = cy*0 - 0*1  + sy*0 + 0*0
        rz = cy*0 + 0*1  + sy*0 - 0*0   # wait, let me just do this properly
        rw = cy*0 - 0*1  - sy*0 - 0*0

        # Proper quaternion multiplication:
        # (a1,b1,c1,d1) ⊗ (a2,b2,c2,d2)  where (a=x,b=y,c=z,d=w)
        # result.w = d1*d2 - a1*a2 - b1*b2 - c1*c2
        # result.x = d1*a2 + a1*d2 + b1*c2 - c1*b2
        # result.y = d1*b2 - a1*c2 + b1*d2 + c1*a2
        # result.z = d1*c2 + a1*b2 - b1*a2 + c1*d2

        # q_yaw (x,y,z,w) = (0, 0, sy, cy)
        # q_flip(x,y,z,w) = (1, 0, 0,  0 )
        a1, b1, c1, d1 = 0.0, 0.0, sy, cy     # q_yaw
        a2, b2, c2, d2 = 1.0, 0.0, 0.0, 0.0   # q_flip

        fw = d1*d2 - a1*a2 - b1*b2 - c1*c2
        fx = d1*a2 + a1*d2 + b1*c2 - c1*b2
        fy = d1*b2 - a1*c2 + b1*d2 + c1*a2
        fz = d1*c2 + a1*b2 - b1*a2 + c1*d2

        # Normalise
        n = math.sqrt(fx*fx + fy*fy + fz*fz + fw*fw)
        if n < 1e-6:
            return 1.0, 0.0, 0.0, 0.0
        return fx/n, fy/n, fz/n, fw/n

    # ──────────────────────────────────────────────────────────────────
    # Pick order helper
    # ──────────────────────────────────────────────────────────────────
    def _get_current_target(
            self, detected: list[DetectedObject]) -> DetectedObject | None:
        if not detected:
            return None
        # pick_order is a comma-separated string e.g. "green_cube,mouse,Stapler"
        pick_order_str = self.get_parameter('pick_order').value.strip()
        if pick_order_str:
            for name in [n.strip() for n in pick_order_str.split(',')]:
                for d in detected:
                    if d.name == name:
                        return d
        # Default: left-most detected object (already sorted by cx_px)
        return detected[0]

    # ──────────────────────────────────────────────────────────────────
    # Debug drawing
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_detection(img: np.ndarray, obj: DetectedObject, profile: dict):
        COLORS = {
            'green_cube':       (0,   255,   0),
            'calculator':       (128, 128, 128),
            'mini_stacking_box':(255,   0,   0),
            'mouse':            ( 50,  50,  50),
            'scissor':          (192, 192, 192),
            'Stapler':          (  0,   0, 255),
        }
        color = COLORS.get(obj.name, (255, 255, 0))

        cv2.circle(img, (obj.cx_px, obj.cy_px), 8, color, -1)

        # Draw orientation arrow
        length = 40
        ex = int(obj.cx_px + length * math.cos(obj.angle_rad))
        ey = int(obj.cy_px + length * math.sin(obj.angle_rad))
        cv2.arrowedLine(img, (obj.cx_px, obj.cy_px), (ex, ey), color, 2)

        label = (f'{obj.name} '
                 f'({obj.x_m:.2f},{obj.y_m:.2f},{obj.z_m:.2f})')
        cv2.putText(img, label,
                    (obj.cx_px + 10, obj.cy_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MultiObjectPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
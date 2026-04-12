#!/usr/bin/env python3
"""
hsv_tuner.py
━━━━━━━━━━━━
Interactive HSV tuner for calibrating object_color_profiles.yaml.
Subscribes to /camera/image_raw and opens an OpenCV trackbar window.

Usage:
  ros2 run opencv hsv_tuner --ros-args -p object_name:=green_cube

Controls:
  Trackbars → adjust H/S/V min/max live
  Press 's'  → save values to object_color_profiles.yaml
  Press 'q'  → quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os


class HSVTuner(Node):

    WINDOW_ORIGINAL = 'Original + Mask'
    WINDOW_CONTROLS = 'HSV Controls'

    def __init__(self):
        super().__init__('hsv_tuner')

        self.declare_parameter('object_name', 'green_cube')
        self.declare_parameter('color_topic', '/camera/image_raw')
        default_cfg = os.path.join(
            os.path.dirname(__file__), '..', 'config',
            'object_color_profiles.yaml')
        self.declare_parameter('config_file', default_cfg)

        self.object_name = self.get_parameter('object_name').value
        self.cfg_path    = self.get_parameter('config_file').value
        self.bridge      = CvBridge()

        # Load existing values as starting point
        with open(self.cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        profile = self.cfg['objects'].get(self.object_name, {})
        lo = profile.get('hsv_lower', [0,   0,   0])
        hi = profile.get('hsv_upper', [179, 255, 255])

        # Create windows
        cv2.namedWindow(self.WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_CONTROLS, 600, 250)

        cv2.createTrackbar('H min', self.WINDOW_CONTROLS, lo[0], 179, lambda x: None)
        cv2.createTrackbar('H max', self.WINDOW_CONTROLS, hi[0], 179, lambda x: None)
        cv2.createTrackbar('S min', self.WINDOW_CONTROLS, lo[1], 255, lambda x: None)
        cv2.createTrackbar('S max', self.WINDOW_CONTROLS, hi[1], 255, lambda x: None)
        cv2.createTrackbar('V min', self.WINDOW_CONTROLS, lo[2], 255, lambda x: None)
        cv2.createTrackbar('V max', self.WINDOW_CONTROLS, hi[2], 255, lambda x: None)

        self.sub = self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self._image_cb, 5)

        self.get_logger().info(
            f'HSV Tuner started for: {self.object_name}\n'
            f'  Press [s] to save, [q] to quit')

    def _image_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos('H min', self.WINDOW_CONTROLS)
        h_max = cv2.getTrackbarPos('H max', self.WINDOW_CONTROLS)
        s_min = cv2.getTrackbarPos('S min', self.WINDOW_CONTROLS)
        s_max = cv2.getTrackbarPos('S max', self.WINDOW_CONTROLS)
        v_min = cv2.getTrackbarPos('V min', self.WINDOW_CONTROLS)
        v_max = cv2.getTrackbarPos('V max', self.WINDOW_CONTROLS)

        lo = np.array([h_min, s_min, v_min], dtype=np.uint8)
        hi = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        # Overlay mask as green on original
        result = bgr.copy()
        result[mask > 0] = [0, 255, 0]

        label = (f'Object: {self.object_name}  |  '
                 f'H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]  |  '
                 f'[s]=save  [q]=quit')
        cv2.putText(result, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow(self.WINDOW_ORIGINAL, result)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            self._save_values(h_min, s_min, v_min, h_max, s_max, v_max)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit

    def _save_values(self, h_lo, s_lo, v_lo, h_hi, s_hi, v_hi):
        self.cfg['objects'][self.object_name]['hsv_lower'] = [h_lo, s_lo, v_lo]
        self.cfg['objects'][self.object_name]['hsv_upper'] = [h_hi, s_hi, v_hi]
        with open(self.cfg_path, 'w') as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        self.get_logger().info(
            f'Saved HSV [{h_lo},{s_lo},{v_lo}] - [{h_hi},{s_hi},{v_hi}] '
            f'for {self.object_name} → {self.cfg_path}')


def main(args=None):
    rclpy.init(args=args)
    node = HSVTuner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

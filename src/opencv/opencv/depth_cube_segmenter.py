#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import tf_transformations

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class DepthCubeSegmenter(Node):

    def __init__(self):
        super().__init__('depth_cube_segmenter')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10)

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.info_callback, 10)

        self.centroid_pub = self.create_publisher(
            PointStamped, '/cube/centroid', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.depth = None

        self.get_logger().info("Depth Cube Segmenter Started")

    # ------------------------------------------------
    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # ------------------------------------------------
    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='32FC1')

    # ------------------------------------------------
    def rgb_callback(self, msg):

        if self.depth is None or self.fx is None:
            return

        color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        depth = self.depth.copy()

        depth_valid = depth[np.isfinite(depth)]
        if len(depth_valid) == 0:
            return

        # ---------------------------------
        # Table depth estimation
        # ---------------------------------
        hist_vals, hist_bins = np.histogram(depth_valid, bins=60)
        table_depth = hist_bins[np.argmax(hist_vals)]

        raw_mask = (depth < table_depth - 0.015) & (depth > 0.1)
        binary = raw_mask.astype(np.uint8)

        h = binary.shape[0]
        binary[int(h * 0.75):, :] = 0

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 300 < area < 8000:
                mask[labels == i] = 1

        if np.sum(mask) < 100:
            cv2.imshow("Cube Segmentation", color)
            cv2.imshow("Depth Mask", binary * 255)
            cv2.waitKey(1)
            return

        # =============================================
        # 6D POSE ESTIMATION
        # =============================================
        ys, xs = np.where(mask)

        points_xy = []

        for u_pix, v_pix in zip(xs, ys):
            Zp = depth[v_pix, u_pix]
            Xp = (u_pix - self.cx) * Zp / self.fx
            Yp = (v_pix - self.cy) * Zp / self.fy
            points_xy.append([Xp, Yp])

        points_xy = np.array(points_xy)

        centroid_xy = np.mean(points_xy, axis=0)

        Z = float(np.median(depth[mask]))
        X = float(centroid_xy[0])
        Y = float(centroid_xy[1])

        # PCA orientation
        cov = np.cov(points_xy.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        yaw_raw = np.arctan2(major_axis[1], major_axis[0])
        yaw = np.round(yaw_raw / (np.pi / 2)) * (np.pi / 2)

        quat = tf_transformations.quaternion_from_euler(
            np.pi, 0.0, yaw)

        # -----------------------------
        # DEBUG PRINTS
        # -----------------------------
        self.get_logger().info(
            f"Cube Pose: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

        self.get_logger().info(
            f"Quaternion: {quat}")

        # Publish centroid
        point_msg = PointStamped()
        point_msg.header = msg.header
        point_msg.point.x = X
        point_msg.point.y = Y
        point_msg.point.z = Z
        self.centroid_pub.publish(point_msg)

        # -----------------------------
        # cube_frame
        # -----------------------------
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = "cube_frame"

        t.transform.translation.x = X
        t.transform.translation.y = Y
        t.transform.translation.z = Z

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

        # -----------------------------
        # cube_grasp_frame (correct offset)
        # -----------------------------
        grasp_offset = 0.055

        # Convert quaternion to rotation matrix
        R = tf_transformations.quaternion_matrix(quat)
        cube_z_axis = R[:3, 2]  # local Z axis

        grasp_translation = np.array([X, Y, Z]) + grasp_offset * cube_z_axis

        tg = TransformStamped()
        tg.header = msg.header
        tg.child_frame_id = "cube_grasp_frame"

        tg.transform.translation.x = float(grasp_translation[0])
        tg.transform.translation.y = float(grasp_translation[1])
        tg.transform.translation.z = float(grasp_translation[2])

        tg.transform.rotation.x = quat[0]
        tg.transform.rotation.y = quat[1]
        tg.transform.rotation.z = quat[2]
        tg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tg)

        self.get_logger().info(
            f"Pre-Grasp Pose: {grasp_translation}")

        # -----------------------------
        # Visualization
        # -----------------------------
        overlay = color.copy()
        overlay[mask.astype(bool)] = (
            overlay[mask.astype(bool)] * 0.4 +
            np.array([0, 0, 255]) * 0.6
        )

        cv2.imshow("Cube Segmentation", overlay)
        cv2.imshow("Depth Mask", mask * 255)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthCubeSegmenter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
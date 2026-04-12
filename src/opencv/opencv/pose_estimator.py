#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf_transformations


class CubePoseEstimator(Node):

    def __init__(self):
        super().__init__('cube_pose_estimator')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/cube/global_pose',
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fx = self.fy = self.cx = self.cy = None
        self.depth = None

        self.get_logger().info("Cube 6D Pose Estimator Started")

    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')

    def rgb_callback(self, msg):

        if self.depth is None or self.fx is None:
            return

        color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        depth = self.depth.copy()

        depth_valid = depth[np.isfinite(depth)]
        if len(depth_valid) == 0:
            return

        # Detect table depth
        hist_vals, hist_bins = np.histogram(depth_valid, bins=60)
        table_depth = hist_bins[np.argmax(hist_vals)]

        mask = (depth < table_depth - 0.015) & (depth > 0.1)

        if np.sum(mask) < 200:
            return

        # Convert mask pixels to 3D points
        ys, xs = np.where(mask)

        points = []

        for u, v in zip(xs, ys):
            Z = depth[v, u]
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            points.append([X, Y, Z])

        points = np.array(points)

        centroid = np.mean(points, axis=0)

        # PCA for orientation
        cov = np.cov(points[:, :2].T)
        eigvals, eigvecs = np.linalg.eig(cov)

        major_axis = eigvecs[:, np.argmax(eigvals)]

        yaw = np.arctan2(major_axis[1], major_axis[0])

        # Construct rotation
        quat = tf_transformations.quaternion_from_euler(0, np.pi, yaw)

        # Build Pose in camera frame
        pose_cam = PoseStamped()
        pose_cam.header = msg.header
        pose_cam.pose.position.x = float(centroid[0])
        pose_cam.pose.position.y = float(centroid[1])
        pose_cam.pose.position.z = float(centroid[2])
        pose_cam.pose.orientation.x = quat[0]
        pose_cam.pose.orientation.y = quat[1]
        pose_cam.pose.orientation.z = quat[2]
        pose_cam.pose.orientation.w = quat[3]

        # Transform to base_link
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time()
            )

            pose_base = tf_transformations.concatenate_matrices(
                tf_transformations.translation_matrix([
                    pose_cam.pose.position.x,
                    pose_cam.pose.position.y,
                    pose_cam.pose.position.z
                ]),
                tf_transformations.quaternion_matrix(quat)
            )

            trans = tf_transformations.translation_matrix([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            rot = tf_transformations.quaternion_matrix([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            global_mat = tf_transformations.concatenate_matrices(trans, rot, pose_base)

            global_trans = tf_transformations.translation_from_matrix(global_mat)
            global_quat = tf_transformations.quaternion_from_matrix(global_mat)

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "base_link"
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.pose.position.x = float(global_trans[0])
            pose_msg.pose.position.y = float(global_trans[1])
            pose_msg.pose.position.z = float(global_trans[2])
            pose_msg.pose.orientation.x = global_quat[0]
            pose_msg.pose.orientation.y = global_quat[1]
            pose_msg.pose.orientation.z = global_quat[2]
            pose_msg.pose.orientation.w = global_quat[3]

            self.pose_pub.publish(pose_msg)

            # Broadcast TF
            t = TransformStamped()
            t.header = pose_msg.header
            t.child_frame_id = "cube_global_frame"
            t.transform.translation.x = pose_msg.pose.position.x
            t.transform.translation.y = pose_msg.pose.position.y
            t.transform.translation.z = pose_msg.pose.position.z
            t.transform.rotation = pose_msg.pose.orientation

            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().warn(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = CubePoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
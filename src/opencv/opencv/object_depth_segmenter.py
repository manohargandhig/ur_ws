#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import tf_transformations

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class ObjectPoseEstimator(Node):

    def __init__(self):

        super().__init__('object_pose_estimator')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.depth = None

        self.get_logger().info("Object Pose Estimator Started")

    # ------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------

    def info_callback(self, msg):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # ------------------------------------------------
    # Depth
    # ------------------------------------------------

    def depth_callback(self, msg):

        self.depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='32FC1')

    # ------------------------------------------------
    # RGB callback
    # ------------------------------------------------

    def rgb_callback(self, msg):

        if self.depth is None or self.fx is None:
            return

        color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        depth = self.depth.copy()

        # ------------------------------------------------
        # TABLE REMOVAL
        # ------------------------------------------------

        valid = depth[np.isfinite(depth)]

        if len(valid) == 0:
            return

        hist, bins = np.histogram(valid, bins=60)

        table_depth = bins[np.argmax(hist)]

        mask = (depth < table_depth - 0.02) & (depth > 0.1)

        mask = mask.astype(np.uint8)

        # remove robot area
        h = mask.shape[0]
        mask[int(h*0.75):,:] = 0

        kernel = np.ones((5,5),np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if np.sum(mask) < 100:
            cv2.imshow("mask", mask*255)
            cv2.imshow("rgb", color)
            cv2.waitKey(1)
            return

        # ------------------------------------------------
        # BUILD POINT CLOUD
        # ------------------------------------------------

        ys, xs = np.where(mask)

        points = []

        for u,v in zip(xs,ys):

            Z = depth[v,u]

            if Z <= 0 or np.isnan(Z):
                continue

            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            points.append([X,Y,Z])

        points = np.array(points)

        if len(points) < 30:
            return

        # ------------------------------------------------
        # CENTROID
        # ------------------------------------------------

        centroid = np.mean(points, axis=0)

        # ------------------------------------------------
        # PCA ORIENTATION
        # ------------------------------------------------

        centered = points - centroid

        cov = np.cov(centered.T)

        eigvals, eigvecs = np.linalg.eig(cov)

        major_axis = eigvecs[:, np.argmax(eigvals)]

        yaw = np.arctan2(major_axis[1], major_axis[0])

        quat = tf_transformations.quaternion_from_euler(
            np.pi,
            0.0,
            yaw)

        X,Y,Z = centroid

        # ------------------------------------------------
        # cube_frame
        # ------------------------------------------------

        t = TransformStamped()

        t.header = msg.header
        t.child_frame_id = "cube_frame"

        t.transform.translation.x = float(X)
        t.transform.translation.y = float(Y)
        t.transform.translation.z = float(Z)

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

        # ------------------------------------------------
        # GRASP FRAME
        # ------------------------------------------------

        grasp_offset = 0.05

        R = tf_transformations.quaternion_matrix(quat)

        z_axis = R[:3,2]

        grasp_position = centroid + grasp_offset * z_axis

        tg = TransformStamped()

        tg.header = msg.header
        tg.child_frame_id = "cube_grasp_frame"

        tg.transform.translation.x = float(grasp_position[0])
        tg.transform.translation.y = float(grasp_position[1])
        tg.transform.translation.z = float(grasp_position[2])

        tg.transform.rotation.x = quat[0]
        tg.transform.rotation.y = quat[1]
        tg.transform.rotation.z = quat[2]
        tg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tg)

        # ------------------------------------------------
        # PRE-GRASP FRAME
        # ------------------------------------------------

        pre_offset = 0.10

        pre_position = centroid + pre_offset * z_axis

        tp = TransformStamped()

        tp.header = msg.header
        tp.child_frame_id = "cube_pregrasp_frame"

        tp.transform.translation.x = float(pre_position[0])
        tp.transform.translation.y = float(pre_position[1])
        tp.transform.translation.z = float(pre_position[2])

        tp.transform.rotation.x = quat[0]
        tp.transform.rotation.y = quat[1]
        tp.transform.rotation.z = quat[2]
        tp.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tp)

        # ------------------------------------------------
        # Visualization
        # ------------------------------------------------

        overlay = color.copy()

        overlay[mask.astype(bool)] = (
            overlay[mask.astype(bool)]*0.4 +
            np.array([0,0,255])*0.6
        )

        cv2.imshow("segmentation", overlay)
        cv2.imshow("mask", mask*255)

        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = ObjectPoseEstimator()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
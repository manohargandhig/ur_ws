#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ColorFilter(object):

    def __init__(self, node):
        self.node = node
        self.bridge_object = CvBridge()

        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

    def camera_callback(self, data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
        except Exception as e:
            self.node.get_logger().error(str(e))
            return

        # Convert image from BGR to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # ✅ GREEN HSV RANGE
        min_green = np.array([35, 100, 70])
        max_green = np.array([85, 255, 255])

        # Mask for green color
        mask_g = cv2.inRange(hsv, min_green, max_green)

        # Apply mask
        res_g = cv2.bitwise_and(cv_image, cv_image, mask=mask_g)

        # Display results
        cv2.imshow('Original', cv_image)
        cv2.imshow('Mask on color (Green)', mask_g)
        cv2.imshow('Green', res_g)

        cv2.waitKey(1)


def main():
    rclpy.init()
    node = Node('color_filter_node')
    color_filter_object = ColorFilter(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
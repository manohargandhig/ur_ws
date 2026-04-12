#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GetHSVClick(object):

    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.img = None

        # ROS 2 image subscription
        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/image_raw',   # ROS 2 camera topic
            self.image_callback,
            10
        )

        # OpenCV window and mouse callback
        cv.namedWindow('mouseRGB')
        cv.setMouseCallback('mouseRGB', self.mouseRGB)

    def image_callback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='bgr8'
            )
        except Exception as e:
            self.node.get_logger().error(str(e))
            return

        cv.imshow('mouseRGB', self.img)
        cv.waitKey(1)

    def mouseRGB(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and self.img is not None:

            colorsB = self.img[y, x, 0]
            colorsG = self.img[y, x, 1]
            colorsR = self.img[y, x, 2]
            colors = self.img[y, x]

            hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
            hsv = cv.cvtColor(hsv_value, cv.COLOR_BGR2HSV)

            print("HSV :", hsv)
            print("Red:", colorsR)
            print("Green:", colorsG)
            print("Blue:", colorsB)
            print("BGR Format:", colors)
            print("Coordinates of pixel: X:", x, "Y:", y)
            print("----------------------------")


def main():
    rclpy.init()

    node = Node('get_hsv_click_node')
    get_hsv_object = GetHSVClick(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down")

    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
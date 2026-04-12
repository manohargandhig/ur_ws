#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class LoadImage(object):
    def __init__(self, node):
        self.node = node
        self.bridge_object = CvBridge()

        self.cv_image = None
        self.mouse_x = 0
        self.mouse_y = 0

        # ROS 2 equivalent of rospy.Subscriber
        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        # OpenCV window + mouse tracking
        cv2.namedWindow('frame from camera')
        cv2.setMouseCallback('frame from camera', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and self.cv_image is not None:
            h, w, _ = self.cv_image.shape
            if 0 <= x < w and 0 <= y < h:
                self.mouse_x = x
                self.mouse_y = y

    def camera_callback(self, data):
        try:
            self.cv_image = self.bridge_object.imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
        except Exception as e:
            self.node.get_logger().error(str(e))
            return

        # Copy image for drawing
        display = self.cv_image.copy()

        h, w, _ = display.shape

        # Draw white status bar
        cv2.rectangle(display, (0, h - 35), (w, h), (255, 255, 255), -1)

        # Get pixel values
        b, g, r = self.cv_image[self.mouse_y, self.mouse_x]

        # Status text (MATCHES VIDEO STYLE)
        text = f"x={self.mouse_x}, y={self.mouse_y}   R:{r} G:{g} B:{b}"

        cv2.putText(
            display,
            text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        cv2.imshow('frame from camera', display)
        cv2.waitKey(1)


def main():
    rclpy.init()

    node = Node('load_image_node')
    load_image_object = LoadImage(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
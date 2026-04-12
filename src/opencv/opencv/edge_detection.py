#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ObjectDetection(object):
    def __init__(self, node):
        self.node = node

        # SAME as ROS 1
        self.bridge_object = CvBridge()

        # ROS 2 equivalent of rospy.Subscriber
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

        # ✅ UPDATED CROP:
        # from x=210,y=150 to x=420,y=300
        cropped_img = cv_image[150:300, 210:420]

        # convert the image to grayscale
        gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
        cv.imshow("gray", gray)

        # adaptive threshold
        mask = cv.adaptiveThreshold(
            gray, 255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY_INV,
            3, 3
        )
        cv.imshow("mask", mask)

        # find contours
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        print("contours:", contours)

        for cnt in contours:
            cv.polylines(cropped_img, [cnt], True, [255, 0, 0], 1)

        object_detected = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 20:
                cnt = cv.approxPolyDP(
                    cnt, 0.03 * cv.arcLength(cnt, True), True
                )
                object_detected.append(cnt)

        print("how many object I detect:", len(object_detected))
        print(object_detected)

        for cnt in object_detected:
            rect = cv.minAreaRect(cnt)
            (x_center, y_center), (w, h), orientation = rect

            box = cv.boxPoints(rect)
            box = np.int0(box)

            cv.polylines(cropped_img, [box], True, (255, 0, 0), 1)

            cv.putText(
                cropped_img,
                "x: {} y: {}".format(round(x_center, 1), round(y_center, 1)),
                (int(x_center), int(y_center)),
                cv.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
                1
            )

            cv.circle(
                cropped_img,
                (int(x_center), int(y_center)),
                1,
                (255, 0, 0),
                thickness=-1
            )

        # 🔹 Resize only for display (keeps logic same)
        display = cv.resize(
            cropped_img,
            None,
            fx=3,
            fy=3,
            interpolation=cv.INTER_LINEAR
        )

        cv.imshow("cropped", display)
        # cv.imshow("original", cv_image)
        cv.waitKey(1)


def main():
    rclpy.init()

    node = Node('object_detection')
    object_detection = ObjectDetection(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down")

    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
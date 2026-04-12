#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import zmq
import pickle
import numpy as np


class ZMQRGBDPublisher(Node):
    def __init__(self):
        super().__init__("zmq_rgbd_publisher")
        self.bridge = CvBridge()

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind("tcp://*:5555")

        self.rgb_sub   = Subscriber(self, Image,      "/camera/image_raw")
        self.depth_sub = Subscriber(self, Image,      "/camera/depth/image_raw")
        self.cinfo_sub = Subscriber(self, CameraInfo, "/camera/camera_info")
        self.dinfo_sub = Subscriber(self, CameraInfo, "/camera/depth/camera_info")

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.cinfo_sub, self.dinfo_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.callback)
        self.get_logger().info("ZMQ RGBD Publisher started on tcp://*:5555")

    @staticmethod
    def extract_intrinsics(info_msg):
        K = np.array(info_msg.k).reshape(3, 3)
        return {
            "fx":     float(K[0, 0]),
            "fy":     float(K[1, 1]),
            "cx":     float(K[0, 2]),
            "cy":     float(K[1, 2]),
            "width":  info_msg.width,
            "height": info_msg.height,
            "D":      list(info_msg.d),
            "K":      K.tolist(),
        }

    def callback(self, rgb_msg, depth_msg, cam_info_msg, depth_info_msg):
        bgr   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        bundle = {
            "rgb":        bgr,
            "depth":      depth,
            "cam_info":   self.extract_intrinsics(cam_info_msg),
            "depth_info": self.extract_intrinsics(depth_info_msg),
            "stamp":      rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9,
        }

        # protocol=4 → max supported by Python 3.7 on the subscriber side
        self.socket.send(pickle.dumps(bundle, protocol=4))
        self.get_logger().debug(f"Sent bundle @ {bundle['stamp']:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = ZMQRGBDPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

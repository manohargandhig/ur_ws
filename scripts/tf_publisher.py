#!/usr/bin/env python3
"""
Reads camera_link -> base_link TF from ROS2 and publishes it over ZMQ.
Run this in your ROS2 environment alongside zmq_rgbd_publisher.
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import zmq
import pickle
import numpy as np


def transform_to_matrix(t):
    """Convert geometry_msgs/TransformStamped to 4x4 numpy matrix."""
    from scipy.spatial.transform import Rotation
    tx = t.transform.translation.x
    ty = t.transform.translation.y
    tz = t.transform.translation.z
    qx = t.transform.rotation.x
    qy = t.transform.rotation.y
    qz = t.transform.rotation.z
    qw = t.transform.rotation.w

    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T


class TFPublisher(Node):
    def __init__(self):
        super().__init__("tf_zmq_publisher")

        # TF setup
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ZMQ PUB on port 5558
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind("tcp://*:5558")

        # Publish at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_tf)
        self.get_logger().info("TF ZMQ publisher started on tcp://*:5558")

    def publish_tf(self):
        try:
            # Change these frame names to match your URDF/TF tree
            t = self.tf_buffer.lookup_transform(
                "base_link",        # target frame  (robot base)
                "camera_optical_link",      # source frame  (camera optical frame)
                rclpy.time.Time()   # latest available
            )
            matrix = transform_to_matrix(t)
            msg = {
                "T_base_camera": matrix,           # 4x4 float64
                "stamp": self.get_clock().now().nanoseconds * 1e-9,
            }
            self.sock.send(pickle.dumps(msg, protocol=4))

        except Exception:
            pass   # TF not yet available — keep trying silently


def main(args=None):
    rclpy.init(args=args)
    node = TFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

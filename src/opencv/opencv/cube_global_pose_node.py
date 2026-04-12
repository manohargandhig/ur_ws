#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


class CubeGlobalPose(Node):

    def __init__(self):
        super().__init__('cube_global_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/cube/global_pose',
            10
        )

        self.get_logger().info("Cube Global Pose Node Started")

    def timer_callback(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "cube_frame",
                rclpy.time.Time()
            )

            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z

            pose.pose.orientation = transform.transform.rotation

            self.pose_pub.publish(pose)

        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CubeGlobalPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import zmq
import pickle
import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from ur_sim.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import TransformStamped
import tf2_ros


class ObjectToBaseTF(Node):

    def __init__(self):
        super().__init__('object_to_base_tf')

        self.pub = self.create_publisher(
            DetectedObjectArray,
            '/detected_objects',
            10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        context = zmq.Context()

        self.det_sock = context.socket(zmq.SUB)
        self.det_sock.connect("tcp://localhost:5556")
        self.det_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self.tf_sock = context.socket(zmq.SUB)
        self.tf_sock.connect("tcp://localhost:5558")
        self.tf_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self.T_base_camera = None
        self.timer = self.create_timer(0.1, self.process)
        self.get_logger().info("Pose Node Started - FINAL CORRECTED")

    # -------------------------------
    def backproject_mask(self, mask, depth, cam_info):
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0:
            return None

        Zs = depth[ys, xs]
        valid = (Zs > 0.05) & (Zs < 10.0)
        if valid.sum() == 0:
            return None

        xs = xs[valid]
        ys = ys[valid]
        Zs = Zs[valid]

        Xs = (xs - cam_info["cx"]) * Zs / cam_info["fx"]
        Ys = (ys - cam_info["cy"]) * Zs / cam_info["fy"]

        return np.array([np.median(Xs), np.median(Ys), np.median(Zs)])

    # -------------------------------
    def transform_point_to_base(self, point_cam_cv):
        """
        The received T_base_camera has det(R) = -1 (invalid rotation).
        This means the sender is actually sending T_camera_base.
        
        To get T_base_camera correctly:
          T_base_camera = inv(T_camera_base)
        
        For a proper 4x4 transform:
          R_base_cam = R_cam_base^T
          t_base_cam = -R_base_cam @ t_cam_base
        """
        T_cam_base = self.T_base_camera  # what we actually receive
        
        R = T_cam_base[:3, :3]
        t = T_cam_base[:3, 3]
        
        # Correct inversion
        R_inv = R.T
        t_inv = -R_inv @ t
        
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R_inv
        T_base_cam[:3, 3]  = t_inv

        x_cv, y_cv, z_cv = point_cam_cv
        p = np.array([x_cv, y_cv, z_cv, 1.0])
        p_base = (T_base_cam @ p)[:3]

        self.get_logger().info(
            f"  det(R)={np.linalg.det(R):.3f} "
            f"cam_cv=({x_cv:.3f},{y_cv:.3f},{z_cv:.3f}) "
            f"-> base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})"
        )

        return p_base

    # -------------------------------
    def publish_tf(self, point_base, label, idx):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base"
        t.child_frame_id = f"{label}_{idx}"
        t.transform.translation.x = float(point_base[0])
        t.transform.translation.y = float(point_base[1])
        t.transform.translation.z = float(point_base[2])
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    # -------------------------------
    def process(self):

        try:
            while True:
                tf_raw = self.tf_sock.recv(zmq.NOBLOCK)
                tf_msg = pickle.loads(tf_raw)
                self.T_base_camera = tf_msg["T_base_camera"]
        except zmq.Again:
            pass

        if self.T_base_camera is None:
            self.get_logger().warn("Waiting for TF...")
            return

        try:
            raw = self.det_sock.recv(zmq.NOBLOCK)
            msg = pickle.loads(raw)
        except zmq.Again:
            return

        detections = msg.get("detections", [])
        depth      = msg.get("depth", None)
        cam_info   = msg.get("cam_info", None)
        frame      = msg.get("frame", None)

        if cam_info is None or depth is None or frame is None:
            self.get_logger().warn("Missing data")
            return

        overlay = frame.copy()
        obj_array = DetectedObjectArray()

        for i, det in enumerate(detections):
            mask  = det.get("mask", None)
            label = det.get("label", "unknown")
            bbox  = det.get("bbox", None)

            if mask is None or bbox is None:
                continue

            point_cam_cv = self.backproject_mask(mask, depth, cam_info)
            if point_cam_cv is None:
                continue

            point_base = self.transform_point_to_base(point_cam_cv)

            self.publish_tf(point_base, label, i)

            obj = DetectedObject()
            obj.label = label
            obj.pose.position.x = float(point_base[0])
            obj.pose.position.y = float(point_base[1])
            obj.pose.position.z = float(point_base[2])
            obj.pose.orientation.w = 1.0
            obj_array.objects.append(obj)

            y1, x1, y2, x2 = bbox
            text = (f"X:{point_base[0]:.2f} "
                    f"Y:{point_base[1]:.2f} "
                    f"Z:{point_base[2]:.2f}")
            cv2.putText(overlay, text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(overlay, text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(obj_array.objects) > 0:
            self.pub.publish(obj_array)

        cv2.imshow("Mask R-CNN + XYZ Pose", overlay)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ObjectToBaseTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

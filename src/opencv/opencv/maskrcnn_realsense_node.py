#!/home/roboticslab/anaconda3/envs/mask/bin/python

import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
import skimage.draw

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from cv_bridge import CvBridge

# ---------------- MaskRCNN Setup ----------------

ROOT_DIR = "/home/roboticslab/Mask_RCNN-master/Mask_RCNN-master"

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

WEIGHTS_PATH = os.path.join(
    ROOT_DIR,
    "logs",
    "object20260305T1222",
    "mask_rcnn_object_0090.h5"
)


# ---------------- Config ----------------

class CustomConfig(Config):

    NAME = "object"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 5

    DETECTION_MIN_CONFIDENCE = 0.5

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024


# ---------------- Dataset (only for class names) ----------------

class InferenceDataset(utils.Dataset):

    def load_classes(self):

        self.add_class("object", 1, "scissor")
        self.add_class("object", 2, "stapler")
        self.add_class("object", 3, "calculator")
        self.add_class("object", 4, "cube")
        self.add_class("object", 5, "mouse")

        self.prepare()


# ---------------- ROS Node ----------------

class MaskRCNNNode(Node):

    def __init__(self):

        super().__init__("maskrcnn_realsense_node")

        self.bridge = CvBridge()

        # ---------------- Dataset ----------------

        self.dataset = InferenceDataset()
        self.dataset.load_classes()

        self.class_names = self.dataset.class_names

        # ---------------- Model ----------------

        config = CustomConfig()

        self.model = modellib.MaskRCNN(
            mode="inference",
            model_dir=MODEL_DIR,
            config=config
        )

        self.get_logger().info("Loading weights...")

        self.model.load_weights(WEIGHTS_PATH, by_name=True)

        self.get_logger().info("MaskRCNN loaded successfully")

        # ---------------- Camera params ----------------

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.depth_image = None

        # ---------------- Subscribers ----------------

        self.rgb_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.info_callback,
            10
        )

        # ---------------- Publishers ----------------

        self.image_pub = self.create_publisher(
            Image,
            "/maskrcnn/annotated_image",
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/recognized_object_array",
            10
        )

        self.get_logger().info("MaskRCNN ROS Node Started")

    # ---------------- Camera Info ----------------

    def info_callback(self, msg):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # ---------------- Depth ----------------

    def depth_callback(self, msg):

        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="32FC1"
        )

    # ---------------- RGB callback ----------------

    def rgb_callback(self, msg):

        if self.depth_image is None or self.fx is None:
            return

        color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        image = rgb.astype(np.uint8)

        # ---------------- Run MaskRCNN ----------------

        results = self.model.detect([image], verbose=0)

        r = results[0]

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        for i in range(len(r["scores"])):

            score = r["scores"][i]

            if score < 0.6:
                continue

            y1, x1, y2, x2 = r["rois"][i]

            mask = r["masks"][:, :, i]

            class_id = r["class_ids"][i]

            label = self.class_names[class_id]

            # ---------------- Depth extraction ----------------

            depth_values = self.depth_image[mask]

            depth_values = depth_values[
                (depth_values > 0.1) &
                (depth_values < 5.0)
            ]

            if len(depth_values) == 0:
                continue

            Z = np.median(depth_values)

            ys, xs = np.where(mask)

            u = int(np.mean(xs))
            v = int(np.mean(ys))

            # ---------------- Backproject ----------------

            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            # ---------------- ROS Detection ----------------

            detection = Detection2D()

            detection.header = msg.header

            detection.bbox.center.x = float((x1 + x2) / 2)
            detection.bbox.center.y = float((y1 + y2) / 2)

            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)

            hypothesis = ObjectHypothesisWithPose()

            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = float(score)

            hypothesis.pose.pose.position.x = float(X)
            hypothesis.pose.pose.position.y = float(Y)
            hypothesis.pose.pose.position.z = float(Z)

            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

            # ---------------- Visualization ----------------

            cv2.rectangle(
                color_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(color_image, (u, v), 5, (255, 0, 0), -1)

            cv2.putText(
                color_image,
                f"{label} {Z:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        self.detection_pub.publish(detections_msg)

        annotated_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")

        annotated_msg.header = msg.header

        self.image_pub.publish(annotated_msg)


# ---------------- Main ----------------

def main(args=None):

    rclpy.init(args=args)

    node = MaskRCNNNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
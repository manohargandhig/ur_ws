#!/home/roboticslab/anaconda3/envs/rtsdf/bin/python

import time
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from tf_transformations import quaternion_matrix

import open3d as o3d
import open3d.core as o3c


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    MIN_DEPTH = 0.25
    MAX_DEPTH = 1.5

    VOXEL_SIZE = 0.004
    BLOCK_RESOLUTION = 16
    BLOCK_COUNT = 80000

    BILATERAL_KERNEL = 3
    BILATERAL_SIGMA_DEPTH = 0.003
    BILATERAL_SIGMA_COLOR = 15

    PREVIEW_EVERY_N_FRAMES = 10
    CURVATURE_SAMPLES = 8000

    USE_ICP = False   # Gazebo TF is already accurate


def curvature_colormap(values):
    norm = (values - values.min()) / (values.max() - values.min() + 1e-6)
    cmap = plt.get_cmap("jet")
    return cmap(norm)[:, :3]


def transform_to_matrix(transform_msg):
    t = transform_msg.transform.translation
    q = transform_msg.transform.rotation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


class TSDFGazeboNode(Node):
    def __init__(self):
        super().__init__("tsdf_gazebo_node")

        self.declare_parameter("color_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "camera_optical_link")
        self.declare_parameter("visualize", True)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.visualize = self.get_parameter("visualize").value

        self.get_logger().info(f"Color topic      : {self.color_topic}")
        self.get_logger().info(f"Depth topic      : {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic : {self.camera_info_topic}")
        self.get_logger().info(f"World frame      : {self.world_frame}")
        self.get_logger().info(f"Camera frame     : {self.camera_frame}")

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_color = None
        self.latest_depth = None
        self.intrinsic_tensor = None
        self.intrinsic_o3d = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        try:
            self.cuda_device = o3c.Device("CUDA:0")
            self.get_logger().info("Using Open3D device: CUDA:0")
        except Exception:
            self.cuda_device = o3c.Device("CPU:0")
            self.get_logger().warn("CUDA not available, using CPU:0")

        self.cpu_device = o3c.Device("CPU:0")

        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=Config.VOXEL_SIZE,
            block_resolution=Config.BLOCK_RESOLUTION,
            block_count=Config.BLOCK_COUNT,
            device=self.cuda_device,
        )

        self.create_subscription(Image, self.color_topic, self.color_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.frame_count = 0
        self.depth_history = []

        self.vis = None
        self.mesh_vis = None
        if self.visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("TSDF Gazebo Reconstruction", 1280, 720)
            self.mesh_vis = o3d.geometry.TriangleMesh()
            self.vis.add_geometry(self.mesh_vis)

        self.timer = self.create_timer(0.05, self.process_frame)

    def color_callback(self, msg):
        try:
            if msg.encoding.lower() == "rgb8":
                color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.latest_color = color.astype(np.float32) / 255.0
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth = np.asarray(depth, dtype=np.float32)

            if msg.encoding == "16UC1":
                depth = depth / 1000.0
            elif msg.encoding == "32FC1":
                pass
            else:
                self.get_logger().warn(f"Unexpected depth encoding: {msg.encoding}")

            depth[(depth < Config.MIN_DEPTH) | (depth > Config.MAX_DEPTH)] = 0.0

            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def camera_info_callback(self, msg):
        if self.intrinsic_tensor is not None:
            return

        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        self.intrinsic_tensor = o3c.Tensor(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]],
            dtype=o3c.float64,
            device=self.cpu_device
        )

        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            msg.width, msg.height, fx, fy, cx, cy
        )

        self.get_logger().info(
            f"Intrinsics received: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
        )

    def get_camera_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
            return transform_to_matrix(tf_msg)
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed [{self.world_frame} -> {self.camera_frame}]: {e}"
            )
            return None

    def process_frame(self):
        with self.lock:
            if self.latest_color is None or self.latest_depth is None or self.intrinsic_tensor is None:
                return

            color_np = self.latest_color.copy()
            depth_np = self.latest_depth.copy()

        pose = self.get_camera_pose()
        if pose is None:
            return

        self.depth_history.append(np.var(depth_np))

        depth_gpu = o3d.t.geometry.Image(
            o3c.Tensor(depth_np, dtype=o3c.float32, device=self.cuda_device)
        )
        color_gpu = o3d.t.geometry.Image(
            o3c.Tensor(color_np, dtype=o3c.float32, device=self.cuda_device)
        )

        depth_gpu = depth_gpu.filter_bilateral(
            Config.BILATERAL_KERNEL,
            Config.BILATERAL_SIGMA_DEPTH,
            Config.BILATERAL_SIGMA_COLOR
        )

        extrinsic_tensor = o3c.Tensor(
            pose,
            dtype=o3c.float64,
            device=self.cpu_device
        )

        frustum = self.vbg.compute_unique_block_coordinates(
            depth_gpu,
            self.intrinsic_tensor,
            extrinsic_tensor,
            depth_scale=1.0,
            depth_max=Config.MAX_DEPTH
        )

        self.vbg.integrate(
            frustum,
            depth_gpu,
            color_gpu,
            self.intrinsic_tensor,
            self.intrinsic_tensor,
            extrinsic_tensor,
            depth_scale=1.0,
            depth_max=Config.MAX_DEPTH,
        )

        if self.visualize and self.frame_count % Config.PREVIEW_EVERY_N_FRAMES == 0:
            if self.vbg.hashmap().size() > 0:
                mesh_gpu = self.vbg.extract_triangle_mesh()
                mesh = mesh_gpu.to_legacy()
                mesh.compute_vertex_normals()

                self.mesh_vis.vertices = mesh.vertices
                self.mesh_vis.triangles = mesh.triangles
                self.mesh_vis.vertex_normals = mesh.vertex_normals
                self.mesh_vis.vertex_colors = mesh.vertex_colors

                self.vis.update_geometry(self.mesh_vis)
                self.vis.poll_events()
                self.vis.update_renderer()

                self.get_logger().info(
                    f"Integrated frames: {self.frame_count}, active blocks: {self.vbg.hashmap().size()}"
                )

        self.frame_count += 1

    def save_results(self):
        self.get_logger().info("Saving TSDF reconstruction...")

        if self.vbg.hashmap().size() == 0:
            self.get_logger().warn("No voxels integrated.")
            if self.vis is not None:
                self.vis.destroy_window()
            return

        mesh_gpu = self.vbg.extract_triangle_mesh()
        mesh = mesh_gpu.to_legacy()
        mesh.compute_vertex_normals()

        mesh_file = f"sharp_tsdf_{int(time.time())}.ply"
        o3d.io.write_triangle_mesh(mesh_file, mesh)

        self.get_logger().info(f"Saved mesh: {mesh_file}")
        self.get_logger().info(f"Vertices: {len(mesh.vertices)}")
        self.get_logger().info(f"Triangles: {len(mesh.triangles)}")
        self.get_logger().info(f"Surface Area: {mesh.get_surface_area():.6f}")
        self.get_logger().info(f"Active Blocks: {self.vbg.hashmap().size()}")

        try:
            pcd = mesh.sample_points_poisson_disk(Config.CURVATURE_SAMPLES)
            pcd.estimate_normals()

            pts = np.asarray(pcd.points)
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            curvatures = []

            for i in range(len(pts)):
                _, idx, _ = kdtree.search_knn_vector_3d(pts[i], 20)
                neighbors = pts[idx]
                cov = np.cov(neighbors.T)
                eigvals, _ = np.linalg.eigh(cov)
                curvature = eigvals[0] / (eigvals.sum() + 1e-6)
                curvatures.append(curvature)

            curvatures = np.asarray(curvatures)
            self.get_logger().info(f"Mean Curvature: {np.mean(curvatures):.6f}")

            colors = curvature_colormap(curvatures)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud("curvature_heatmap.ply", pcd)
            self.get_logger().info("Saved curvature_heatmap.ply")

        except Exception as e:
            self.get_logger().warn(f"Curvature computation failed: {e}")

        if self.vis is not None:
            self.vis.destroy_window()


def main(args=None):
    rclpy.init(args=args)
    node = TSDFGazeboNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
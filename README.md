# UR5 Robotic Perception + TSDF Reconstruction (ROS2 Humble)

## 📌 Overview

This project implements a complete robotic perception pipeline using:

* Semantic Segmentation (Mask R-CNN)
* Object Pose Estimation
* ZMQ-based communication
* TSDF-based volumetric reconstruction

Built on:

* ROS2 Humble (Ubuntu 22.04)
* OpenCV + ZMQ
* Mask R-CNN
* Open3D

---

## 🚀 Features

* Real-time RGB-D streaming
* Object detection + segmentation
* Depth-aware filtering
* Pose estimation
* 3D volumetric reconstruction

---

## 📂 Project Structure

```
ur_ws/
├── src/                     # ROS2 packages
├── scripts/                 # Helper scripts
│   ├── tf_publisher.py
│   ├── maskrcnn_zmq_inference.py
├── README.md
├── .gitignore
```
---

## ⚙️ Setup Instructions

## ⚙️ System Requirements & Prerequisites

### 🖥️ System Requirements

* Ubuntu 22.04
* ROS2 Humble Distribution
* Python 3.10

---

### 🤖 ROS2 Installation

```bash
sudo apt update
sudo apt install ros-humble-desktop -y
```

Initialize ROS2 environment:

```bash
source /opt/ros/humble/setup.bash
```

---

### 🦾 UR5 + MoveIt + Gazebo Dependencies

```bash
sudo apt install ros-humble-moveit -y
sudo apt install ros-humble-gazebo-ros-pkgs -y
sudo apt install ros-humble-ros2-control -y
sudo apt install ros-humble-ros2-controllers -y
```

---

### 📷 OpenCV & Python Dependencies

```bash
pip install numpy scipy opencv-python pyzmq scikit-image matplotlib Pillow
```

---


### Clone Repository

```bash
https://github.com/manohargandhig/ur_ws.git
cd ~/ur_ws
```

---

#  Semantic Segmentation + Pose Estimation

### Terminal 1

```bash
source /opt/ros/humble/setup.bash 
cd ~/ur_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_sim spawn_ur5_camera_gripper_moveit.launch.py with_octomap:=false
```

---

### Terminal 2

```bash
source /opt/ros/humble/setup.bash 
source ~/ur_ws/install/setup.bash
ros2 run opencv zmq_camera_pub
```

---

### Terminal 3

```bash
conda activate mask
python scripts/maskrcnn_zmq_inference.py
```

---

### Terminal 4

```bash
python3 scripts/tf_publisher.py
```

---

### Terminal 5

```bash
source /opt/ros/humble/setup.bash 
source ~/ur_ws/install/setup.bash
ros2 run opencv pose_estimation
```

---

# 🌍 TSDF Reconstruction

## Required Environment

Refer:
https://github.com/manohargandhig/tsdf-conda-env.git

---

### Terminal 1

```bash
source /opt/ros/humble/setup.bash
cd ~
cd ~/ur_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_sim spawn_ur5_camera_gripper_moveit.launch.py with_octomap:=false
```

---

### Terminal 2

```bash
conda activate rtsdf
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHON_EXECUTABLE=/home/roboticslab/anaconda3/envs/rtsdf/bin/python
export PYTHONPATH=/home/roboticslab/anaconda3/envs/rtsdf/lib/python3.10/site-packages

source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash

ros2 run opencv tsdf_gazebo_node \
  --ros-args \
  -p color_topic:=/camera/image_raw \
  -p depth_topic:=/camera/depth/image_raw \
  -p camera_info_topic:=/camera/depth/camera_info \
  -p world_frame:=world \
  -p camera_frame:=camera_optical_link
```

---
---

## 🎥 Demo Videos

### 1. Mask R-CNN Pose Estimation (Simulation)
[▶️ Watch Video](docs/demo_videos/segmentation_demo.mp4)

### 2. TSDF Volumetric Reconstruction
[▶️ Watch Video](docs/demo_videos/tsdf_reconstruction.mp4)

### 3. Real-Time Pre-Grasp using Mask R-CNN + AprilTag
[▶️ Watch Video](docs/demo_videos/pregrasp_demo.mp4)

---

## 👨‍💻 Authors

* G Manohar Gandhi
* Aswin Kumar M S
* S Sai Sankalp

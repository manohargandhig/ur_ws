from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'opencv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amrita',
    maintainer_email='amrita@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'load_image_from_camera = opencv.load_image_from_camera:main',
            'color_filtering = opencv.color_filtering:main',
            'get_hsv_click = opencv.get_hsv_click:main',
            'edge_detection = opencv.edge_detection:main',
            'maskrcnn_realsense_node = opencv.maskrcnn_realsense_node:main',
            'depth_cube_segmenter = opencv.depth_cube_segmenter:main',
            'pose_estimator = opencv.pose_estimator:main',
            'cube_global_pose = opencv.cube_global_pose_node:main',
            'object_depth_segmenter= opencv.object_depth_segmenter:main',
            'multi_object_perception_node = opencv.multi_object_perception_node:main',
            'hsv_tuner= opencv.hsv_tuner:main',
            'zmq_camera_pub = opencv.zmq_camera_pub:main',
            'tsdf_gazebo_node = opencv.tsdf_gazebo_node:main',
            'pose_estimation = opencv.pose_estimation:main'
        ],
    },
)

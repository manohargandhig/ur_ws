"""
perception.launch.py
━━━━━━━━━━━━━━━━━━━
Launches the full multi-object perception pipeline.

Usage:
  # Detection only (check TF frames in RViz):
  ros2 launch opencv perception.launch.py

  # With HSV tuner for a specific object:
  ros2 launch opencv perception.launch.py run_tuner:=true tuner_object:=green_cube

  # Custom pick order:
  ros2 launch opencv perception.launch.py pick_order:=green_cube,mouse,Stapler
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('opencv')
    cfg = os.path.join(pkg, 'config', 'object_color_profiles.yaml')

    # ── Launch arguments ──────────────────────────────────────────────
    return LaunchDescription([

        DeclareLaunchArgument(
            'config_file', default_value=cfg,
            description='Path to object_color_profiles.yaml'),

        DeclareLaunchArgument(
            'debug_viz', default_value='true',
            description='Publish annotated detection image'),

        DeclareLaunchArgument(
            'pick_order', default_value='',
            description='Comma-separated pick order, e.g. green_cube,mouse'),

        DeclareLaunchArgument(
            'run_tuner', default_value='false',
            description='Also launch HSV tuner node'),

        DeclareLaunchArgument(
            'tuner_object', default_value='green_cube',
            description='Object name to tune in HSV tuner'),

        # ── Multi-object perception node ──────────────────────────────
        Node(
            package='opencv',
            executable='multi_object_perception_node',
            name='multi_object_perception',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'debug_viz':   LaunchConfiguration('debug_viz'),
                'pick_order':  LaunchConfiguration('pick_order'),
            }],
            remappings=[
                # If your camera topics differ, remap here
                # ('/camera/image_raw', '/your/camera/color/image_raw'),
                # ('/camera/depth/image_raw', '/your/camera/depth/image_raw'),
            ]
        ),

        # ── HSV tuner (optional) ──────────────────────────────────────
        # Only launched when run_tuner:=true
        Node(
            package='opencv',
            executable='hsv_tuner',
            name='hsv_tuner',
            output='screen',
            parameters=[{
                'config_file':  LaunchConfiguration('config_file'),
                'object_name':  LaunchConfiguration('tuner_object'),
                'color_topic':  '/camera/image_raw',
            }],
            condition=__import__(
                'launch.conditions', fromlist=['IfCondition']
            ).IfCondition(LaunchConfiguration('run_tuner')),
        ),
    ])

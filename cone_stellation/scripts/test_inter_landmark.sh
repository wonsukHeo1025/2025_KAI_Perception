#!/bin/bash
# Test script for inter-landmark factors

echo "Testing ConeSTELLATION with inter-landmark factors enabled"
echo "========================================================="

# Source ROS2
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash

# Terminal 1: Launch dummy publisher
echo "Starting dummy publisher..."
gnome-terminal --tab --title="Dummy Publisher" -- bash -c "
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash
ros2 run cone_stellation dummy_publisher_node.py
"

# Wait for publisher to start
sleep 2

# Terminal 2: Launch SLAM node
echo "Starting SLAM node with inter-landmark factors..."
gnome-terminal --tab --title="SLAM Node" -- bash -c "
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash
ros2 run cone_stellation cone_slam_node --ros-args --params-file src/cone_stellation/config/slam_config.yaml
"

# Terminal 3: Monitor topics
echo "Starting topic monitor..."
gnome-terminal --tab --title="Topic Monitor" -- bash -c "
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash
watch -n 1 'echo \"=== ROS2 Topics ===\"; ros2 topic list | grep -E \"slam|cone|odom\"; echo; echo \"=== TF Tree ===\"; ros2 run tf2_ros tf2_echo map odom --timeout 1.0'
"

# Terminal 4: RViz
echo "Starting RViz..."
gnome-terminal --tab --title="RViz" -- bash -c "
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash
rviz2 -d src/cone_stellation/config/cone_slam.rviz
"

echo ""
echo "All nodes launched! Look for:"
echo "1. Inter-landmark factors in RViz (blue lines between cones)"
echo "2. Console output showing 'Creating inter-landmark factor'"
echo "3. Improved map consistency with fewer drift errors"
echo ""
echo "Press Ctrl+C to stop all nodes"
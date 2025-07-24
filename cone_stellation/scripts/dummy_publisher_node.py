#!/usr/bin/env python3
"""
Dummy publisher node for testing ConeSTELLATION
Publishes simulated cone observations, IMU, and GPS data

This is adapted from cc_slam_sym simulation with modular components:
- sensor_simulator: Handles IMU, GPS, and odometry simulation
- motion_controller: Manages vehicle motion along trajectories
- visualization utils: Provides reusable visualization functions
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, TwistWithCovarianceStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path, Odometry
from custom_interface.msg import TrackedConeArray, TrackedCone
import tf2_ros
from tf_transformations import quaternion_from_euler

import numpy as np
from typing import Dict, List, Tuple
import yaml

# Import modular components (assuming scripts are run from this directory)
try:
    from cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2
    from sensor_simulator import SensorSimulator, SensorNoiseConfig
    from motion_controller import MotionController, MotionScenario, VehicleState
except ImportError:
    # If running as a module
    from .cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2
    from .sensor_simulator import SensorSimulator, SensorNoiseConfig
    from .motion_controller import MotionController, MotionScenario, VehicleState

# We'll create a simple visualization module adapter
class VisualizationAdapter:
    """Simple adapter for visualization functions"""
    
    @staticmethod
    def publish_cone_array(publisher, cones, namespace="cones", frame_id="map", 
                          with_text=False, timestamp=None, ground_truth=False):
        """Publish cone markers - simplified version"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        if timestamp:
            clear_marker.header.stamp = timestamp.to_msg()
        clear_marker.ns = namespace
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Add cone markers
        for idx, cone in enumerate(cones):
            marker = Marker()
            marker.header.frame_id = frame_id
            if timestamp:
                marker.header.stamp = timestamp.to_msg()
            marker.ns = namespace
            marker.id = cone.get('id', idx)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(cone['pos'][0])
            marker.pose.position.y = float(cone['pos'][1])
            marker.pose.position.z = -0.15
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Color
            cone_type = cone.get('type', 'unknown').lower()
            if ground_truth:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.3
            else:
                if cone_type == 'yellow':
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif cone_type == 'blue':
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                elif cone_type == 'red':
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        publisher.publish(marker_array)
    
    @staticmethod
    def create_roi_marker(roi_type, max_range, fov_rad, frame_id="base_link", timestamp=None):
        """Create ROI visualization marker"""
        marker = Marker()
        
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "roi"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Red color for ROI
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.scale.x = 0.1  # Line width
        marker.pose.orientation.w = 1.0
        
        # Generate ROI boundary points
        points = []
        
        if roi_type == 'sector':
            # Origin
            from geometry_msgs.msg import Point
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
            
            # Arc points
            import numpy as np
            num_arc_points = 20
            for i in range(num_arc_points + 1):
                angle = -fov_rad/2 + i * fov_rad / num_arc_points
                p = Point()
                p.x = max_range * np.cos(angle)
                p.y = max_range * np.sin(angle)
                p.z = 0.0
                points.append(p)
            
            # Back to origin
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
        else:  # rectangle
            # Four corners
            from geometry_msgs.msg import Point
            corners = [
                (max_range, max_range/2),
                (max_range, -max_range/2),
                (0, -max_range/2),
                (0, max_range/2),
                (max_range, max_range/2)  # Close the rectangle
            ]
            
            for x, y in corners:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0
                points.append(p)
        
        marker.points = points
        import rclpy.duration
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    @staticmethod
    def create_path_marker(path_points, color="green", namespace="path", frame_id="map", timestamp=None):
        """Create a path visualization marker"""
        marker = Marker()
        
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Color
        if color == "green":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
        marker.color.a = 0.8
        
        marker.scale.x = 0.05  # Line width
        marker.pose.orientation.w = 1.0
        
        # Add points
        from geometry_msgs.msg import Point
        for x, y in path_points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)
        
        import rclpy.duration
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    @staticmethod
    def publish_detected_cones(publisher, detected_cones, namespace="detected_cones", 
                              frame_id="base_link", timestamp=None):
        """Publish detected cone markers"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        if timestamp:
            clear_marker.header.stamp = timestamp.to_msg()
        clear_marker.ns = namespace
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Add detected cone markers
        for idx, (track_id, position, color) in enumerate(detected_cones):
            marker = Marker()
            marker.header.frame_id = frame_id
            if timestamp:
                marker.header.stamp = timestamp.to_msg()
            marker.ns = namespace
            marker.id = track_id
            marker.type = Marker.SPHERE  # Semi-transparent sphere for detected cones
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = -0.15
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.35  # Slightly larger for detected
            marker.scale.y = 0.35
            marker.scale.z = 0.35
            
            # Color based on detection
            color_lower = color.lower()
            if 'yellow' in color_lower:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif 'blue' in color_lower:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif 'red' in color_lower:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 0.4  # Semi-transparent
            
            marker_array.markers.append(marker)
        
        publisher.publish(marker_array)

# Use the adapter functions
publish_cone_array = VisualizationAdapter.publish_cone_array
publish_detected_cones = VisualizationAdapter.publish_detected_cones


class DummyPublisher(Node):
    def __init__(self):
        super().__init__('dummy_publisher')
        
        # Declare operation mode
        self.declare_parameter('operation_mode', 'simulation')
        self.declare_parameter('simulation.use_gps', True)
        self.declare_parameter('simulation.gps_topic', '/gps/odom')
        self.declare_parameter('publish_map_to_odom', True)  # For standalone testing without SLAM
        
        # Declare parameters with hierarchical structure matching YAML
        self.declare_parameter('scenario.id', 1)
        self.declare_parameter('publish_rates.cones', 20.0)
        self.declare_parameter('publish_rates.imu', 100.0)
        self.declare_parameter('publish_rates.gps', 8.0)
        self.declare_parameter('publish_rates.odometry', 100.0)
        self.declare_parameter('vehicle.speed', 5.0)
        self.declare_parameter('odometry_simulation.enable', True)
        self.declare_parameter('sensors.cone_detection.max_range', 15.0)
        self.declare_parameter('sensors.cone_detection.fov_deg', 120.0)
        self.declare_parameter('sensors.cone_detection.roi_type', 'sector')
        self.declare_parameter('sensors.noise.position_stddev', 0.05)
        
        # IMU Allan variance parameters
        self.declare_parameter('sensors.noise.imu_gyro_noise_density', 0.005)
        self.declare_parameter('sensors.noise.imu_gyro_bias_stability', 0.1)
        self.declare_parameter('sensors.noise.imu_gyro_random_walk', 0.00001)
        self.declare_parameter('sensors.noise.imu_accel_noise_density', 0.01)
        self.declare_parameter('sensors.noise.imu_accel_bias_stability', 0.01)
        self.declare_parameter('sensors.noise.imu_accel_random_walk', 0.0001)
        # Per-axis drift parameters
        self.declare_parameter('sensors.noise.odom_drift_x.systematic', 0.0)
        self.declare_parameter('sensors.noise.odom_drift_x.random_stddev', 0.0)
        self.declare_parameter('sensors.noise.odom_drift_y.systematic', 0.0)
        self.declare_parameter('sensors.noise.odom_drift_y.random_stddev', 0.0)
        self.declare_parameter('sensors.noise.odom_drift_theta.systematic', 0.0)
        self.declare_parameter('sensors.noise.odom_drift_theta.random_stddev', 0.0)
        
        # GPS RTK mode parameters
        self.declare_parameter('sensors.noise.gps_mode', 'rtk')
        self.declare_parameter('sensors.noise.gps_rtk_fix_noise_h', 0.02)
        self.declare_parameter('sensors.noise.gps_rtk_fix_noise_v', 0.04)
        self.declare_parameter('sensors.noise.gps_rtk_float_noise_h', 0.3)
        self.declare_parameter('sensors.noise.gps_rtk_float_noise_v', 0.5)
        self.declare_parameter('sensors.noise.gps_single_noise_h', 2.0)
        self.declare_parameter('sensors.noise.gps_single_noise_v', 5.0)
        
        # Detection error parameters
        self.declare_parameter('sensors.detection_errors.enable', True)
        self.declare_parameter('sensors.detection_errors.false_negative_rate', 0.07)
        self.declare_parameter('sensors.detection_errors.false_positive_rate', 0.002)
        self.declare_parameter('sensors.detection_errors.wrong_color_rate', 0.002)
        self.declare_parameter('sensors.detection_errors.unknown_color_rate', 0.08)
        
        # Get operation mode
        self.operation_mode = self.get_parameter('operation_mode').value
        self.publish_map_to_odom = self.get_parameter('publish_map_to_odom').value
        
        # Get parameters
        self.scenario = self.get_parameter('scenario.id').value
        self.publish_rate = self.get_parameter('publish_rates.cones').value
        self.imu_rate = self.get_parameter('publish_rates.imu').value
        self.gps_rate = self.get_parameter('publish_rates.gps').value
        self.odom_rate = self.get_parameter('publish_rates.odometry').value
        self.vehicle_speed = self.get_parameter('vehicle.speed').value
        self.odom_sim_enabled = self.get_parameter('odometry_simulation.enable').value
        self.detection_range = self.get_parameter('sensors.cone_detection.max_range').value
        self.fov_rad = np.radians(self.get_parameter('sensors.cone_detection.fov_deg').value)
        self.roi_type = self.get_parameter('sensors.cone_detection.roi_type').value
        
        # Get noise parameters
        self.cone_position_noise = self.get_parameter('sensors.noise.position_stddev').value
        
        # Create sensor noise configuration
        noise_config = SensorNoiseConfig(
            # IMU Allan variance parameters
            imu_gyro_noise_density=self.get_parameter('sensors.noise.imu_gyro_noise_density').value,
            imu_gyro_bias_stability=self.get_parameter('sensors.noise.imu_gyro_bias_stability').value,
            imu_gyro_random_walk=self.get_parameter('sensors.noise.imu_gyro_random_walk').value,
            imu_accel_noise_density=self.get_parameter('sensors.noise.imu_accel_noise_density').value,
            imu_accel_bias_stability=self.get_parameter('sensors.noise.imu_accel_bias_stability').value,
            imu_accel_random_walk=self.get_parameter('sensors.noise.imu_accel_random_walk').value,
            # GPS RTK mode parameters
            gps_mode=self.get_parameter('sensors.noise.gps_mode').value,
            gps_rtk_fix_noise_h=self.get_parameter('sensors.noise.gps_rtk_fix_noise_h').value,
            gps_rtk_fix_noise_v=self.get_parameter('sensors.noise.gps_rtk_fix_noise_v').value,
            gps_rtk_float_noise_h=self.get_parameter('sensors.noise.gps_rtk_float_noise_h').value,
            gps_rtk_float_noise_v=self.get_parameter('sensors.noise.gps_rtk_float_noise_v').value,
            gps_single_noise_h=self.get_parameter('sensors.noise.gps_single_noise_h').value,
            gps_single_noise_v=self.get_parameter('sensors.noise.gps_single_noise_v').value,
            # Per-axis drift parameters
            odom_drift_x_systematic=self.get_parameter('sensors.noise.odom_drift_x.systematic').value,
            odom_drift_x_random=self.get_parameter('sensors.noise.odom_drift_x.random_stddev').value,
            odom_drift_y_systematic=self.get_parameter('sensors.noise.odom_drift_y.systematic').value,
            odom_drift_y_random=self.get_parameter('sensors.noise.odom_drift_y.random_stddev').value,
            odom_drift_theta_systematic=self.get_parameter('sensors.noise.odom_drift_theta.systematic').value,
            odom_drift_theta_random=self.get_parameter('sensors.noise.odom_drift_theta.random_stddev').value
        )
        
        # Get detection error parameters
        self.detection_errors_enabled = self.get_parameter('sensors.detection_errors.enable').value
        self.false_negative_rate = self.get_parameter('sensors.detection_errors.false_negative_rate').value
        self.false_positive_rate = self.get_parameter('sensors.detection_errors.false_positive_rate').value
        self.wrong_color_rate = self.get_parameter('sensors.detection_errors.wrong_color_rate').value
        self.unknown_color_rate = self.get_parameter('sensors.detection_errors.unknown_color_rate').value
        
        # Load ground truth cones
        if self.scenario == 1:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_1
            self.get_logger().info("Using Scenario 1: Straight track with AEB zone")
            motion_scenario = MotionScenario.STRAIGHT_TRACK
        else:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_2
            self.get_logger().info("Using Scenario 2: Formula Student track")
            motion_scenario = MotionScenario.FORMULA_STUDENT
        
        # Initialize modular components
        self.sensor_sim = SensorSimulator(noise_config)
        self.motion_controller = MotionController(motion_scenario, self.vehicle_speed)
        
        # QoS for ground truth (TRANSIENT_LOCAL)
        gt_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers (with PRD-compliant topic names)
        self.odom_pub = self.create_publisher(Odometry, '/odom_sim', 10)
        self.cone_pub = self.create_publisher(TrackedConeArray, '/fused_sorted_cones_ukf_sim', 10)
        self.imu_pub = self.create_publisher(Imu, '/ouster/imu_sim', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/ublox_gps_node/fix_sim', 10)
        self.gps_vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/ublox_gps_node/fix_velocity_sim', 10)
        # Direct GPS odometry for robot_localization (bypasses navsat_transform)
        self.gps_odom_pub = self.create_publisher(Odometry, '/gps/odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot/pose', 10)
        self.path_pub = self.create_publisher(Path, '/robot/path', 10)
        self.gt_cones_pub = self.create_publisher(MarkerArray, '/ground_truth_map_cones', gt_qos)
        self.roi_pub = self.create_publisher(Marker, '/roi_visualization', 10)
        self.centerline_pub = self.create_publisher(Marker, '/centerline_visualization', 10)
        
        # Detected cones visualization with latched QoS for initial message
        vis_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.detected_cones_vis_pub = self.create_publisher(MarkerArray, '/detected_cones_visualization', vis_qos)
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Path tracking
        self.path = Path()
        self.path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"
        
        # Cone tracking state - simple simulation matching CALICO
        self.cone_track_mapping = {}  # cone_id -> track_id (only for currently visible)
        self.next_track_id = 0  # Start from 0 like CALICO
        
        # Last detected cones for visualization
        self.last_detected_cones = []  # List of (track_id, local_pos, cone_type)
        
        # Track start time for elapsed time
        self.start_time = self.get_clock().now()
        
        # Initialize vehicle state
        self.vehicle_state = self.motion_controller.state
        
        # Timers
        self.cone_timer = self.create_timer(1.0 / self.publish_rate, self.publish_cones)
        self.imu_timer = self.create_timer(1.0 / self.imu_rate, self.publish_imu)
        
        # GPS timer only in simulation mode with GPS enabled
        if self.operation_mode == 'simulation' and self.get_parameter('simulation.use_gps').value:
            self.gps_timer = self.create_timer(1.0 / self.gps_rate, self.publish_gps)
            
        if self.odom_sim_enabled:
            self.odom_timer = self.create_timer(1.0 / self.odom_rate, self.publish_odometry)
        self.motion_timer = self.create_timer(0.01, self.update_motion)  # 100Hz motion update
        self.gt_timer = self.create_timer(1.0, self.publish_ground_truth_cones)  # 1Hz for ground truth
        self.roi_timer = self.create_timer(0.1, self.publish_roi)  # 10Hz for ROI
        self.centerline_timer = self.create_timer(1.0, self.publish_centerline)  # 1Hz for centerline
        
        # Publish initial visualizations immediately
        self.publish_ground_truth_cones()
        self.publish_centerline()
        self.publish_roi()  # Publish ROI immediately
        
        # Publish empty detected cones initially to establish the topic
        empty_markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = "base_link"
        delete_all.ns = "detected_cones"
        delete_all.action = Marker.DELETEALL
        empty_markers.markers.append(delete_all)
        self.detected_cones_vis_pub.publish(empty_markers)
        
        # Log configuration
        self.get_logger().info("Dummy publisher initialized (refactored version)")
        self.get_logger().info(f"Odometry simulation: {'ENABLED' if self.odom_sim_enabled else 'DISABLED (use external odometry)'}")
        self.get_logger().info(f"[DUMMY_MODE] Running in mode: {self.operation_mode}")
        if self.operation_mode == 'simulation' and self.get_parameter('simulation.use_gps').value:
            self.get_logger().info("[DUMMY_GPS] GPS simulation enabled")
        
        # Debug logging
        self.get_logger().info(f"Publishing rates - Cones: {self.publish_rate}Hz, IMU: {self.imu_rate}Hz, Odom: {self.odom_rate}Hz")
        self.get_logger().info(f"Initial vehicle state - X: {self.vehicle_state.position[0]:.2f}, Y: {self.vehicle_state.position[1]:.2f}, Theta: {self.vehicle_state.orientation[2]:.2f}")
        self.get_logger().info(f"publish_map_to_odom: {self.publish_map_to_odom}")
        self.get_logger().info("Timer callbacks created successfully")
    
    
    def update_motion(self):
        """Update robot motion using motion controller"""
        dt = 0.01
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Update vehicle state using motion controller
        self.vehicle_state = self.motion_controller.update_motion(dt, elapsed_time)
        
        # Publish transforms based on ground truth state
        self.publish_transforms()
        
        # Update and publish ground truth path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = self.vehicle_state.position[0]
        pose_stamped.pose.position.y = self.vehicle_state.position[1]
        pose_stamped.pose.position.z = self.vehicle_state.position[2]
        
        q = quaternion_from_euler(
            self.vehicle_state.orientation[0],
            self.vehicle_state.orientation[1], 
            self.vehicle_state.orientation[2]
        )
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        
        self.gt_path.poses.append(pose_stamped)
        if len(self.gt_path.poses) > 1000:  # Limit path length
            self.gt_path.poses.pop(0)
        
        # Publish ground truth pose
        self.pose_pub.publish(pose_stamped)
    
    def publish_transforms(self):
        """Publish TF transforms including ground truth and noisy odometry"""
        now = self.get_clock().now().to_msg()
        transforms = []
        
        # Debug logging every 100 calls (10Hz)
        if not hasattr(self, '_transform_count'):
            self._transform_count = 0
        self._transform_count += 1
        if self._transform_count % 100 == 0:
            self.get_logger().info(f"Publishing transforms (call #{self._transform_count})")
        
        # Map -> Ground_truth_odom (identity - ground truth has no drift)
        map_to_gt_odom = TransformStamped()
        map_to_gt_odom.header.stamp = now
        map_to_gt_odom.header.frame_id = "map"
        map_to_gt_odom.child_frame_id = "ground_truth_odom"
        map_to_gt_odom.transform.rotation.w = 1.0
        transforms.append(map_to_gt_odom)
        
        # Ground_truth_odom -> Ground_truth_base_link
        gt_odom_to_gt_base = TransformStamped()
        gt_odom_to_gt_base.header.stamp = now
        gt_odom_to_gt_base.header.frame_id = "ground_truth_odom"
        gt_odom_to_gt_base.child_frame_id = "ground_truth_base_link"
        gt_odom_to_gt_base.transform.translation.x = self.vehicle_state.position[0]
        gt_odom_to_gt_base.transform.translation.y = self.vehicle_state.position[1]
        gt_odom_to_gt_base.transform.translation.z = self.vehicle_state.position[2]
        
        q_gt = quaternion_from_euler(
            self.vehicle_state.orientation[0],
            self.vehicle_state.orientation[1],
            self.vehicle_state.orientation[2]
        )
        gt_odom_to_gt_base.transform.rotation.x = q_gt[0]
        gt_odom_to_gt_base.transform.rotation.y = q_gt[1]
        gt_odom_to_gt_base.transform.rotation.z = q_gt[2]
        gt_odom_to_gt_base.transform.rotation.w = q_gt[3]
        transforms.append(gt_odom_to_gt_base)
        
        # Publish map->odom ONLY if configured (for standalone testing without SLAM)
        # When SLAM is running, set publish_map_to_odom=false to avoid TF conflicts
        if self.publish_map_to_odom:
            map_to_odom = TransformStamped()
            map_to_odom.header.stamp = now
            map_to_odom.header.frame_id = "map"
            map_to_odom.child_frame_id = "odom"
            # Identity transform - no drift correction
            map_to_odom.transform.rotation.w = 1.0
            transforms.append(map_to_odom)
            
            # Log warning once
            if self._transform_count == 1:
                self.get_logger().warn("Publishing map->odom transform. Disable this when running SLAM!")
        
        # Odom -> Base_link transform (only if odom simulation is enabled)
        if self.odom_sim_enabled:
            # Use the odometry simulator's internal state (already includes drift)
            noisy_x = self.sensor_sim.odom_sim.odom_x
            noisy_y = self.sensor_sim.odom_sim.odom_y
            noisy_theta = self.sensor_sim.odom_sim.odom_theta
            
            # Debug logging first time and every 1000 calls
            if self._transform_count == 1 or self._transform_count % 1000 == 0:
                self.get_logger().info(f"Odom->base_link transform - X: {noisy_x:.2f}, Y: {noisy_y:.2f}, Theta: {noisy_theta:.2f}")
            
            # Odom -> Base_link transform (noisy)
            odom_to_base = TransformStamped()
            odom_to_base.header.stamp = now
            odom_to_base.header.frame_id = "odom"
            odom_to_base.child_frame_id = "base_link"
            odom_to_base.transform.translation.x = noisy_x
            odom_to_base.transform.translation.y = noisy_y
            odom_to_base.transform.translation.z = 0.0
            
            q = quaternion_from_euler(0, 0, noisy_theta)
            odom_to_base.transform.rotation.x = q[0]
            odom_to_base.transform.rotation.y = q[1]
            odom_to_base.transform.rotation.z = q[2]
            odom_to_base.transform.rotation.w = q[3]
            transforms.append(odom_to_base)
        # If odom simulation is disabled, robot_localization will provide odom->base_link
        
        # Base_link -> IMU_link
        base_to_imu = TransformStamped()
        base_to_imu.header.stamp = now
        base_to_imu.header.frame_id = "base_link"
        base_to_imu.child_frame_id = "imu_link"
        base_to_imu.transform.translation.z = 0.3
        base_to_imu.transform.rotation.w = 1.0
        transforms.append(base_to_imu)
        
        # Base_link -> GPS_link
        base_to_gps = TransformStamped()
        base_to_gps.header.stamp = now
        base_to_gps.header.frame_id = "base_link"
        base_to_gps.child_frame_id = "gps_link"
        base_to_gps.transform.translation.x = -0.5
        base_to_gps.transform.translation.z = 0.5
        base_to_gps.transform.rotation.w = 1.0
        transforms.append(base_to_gps)
        
        # Broadcast all transforms
        self.tf_broadcaster.sendTransform(transforms)
    
    def publish_ground_truth_cones(self):
        """Publish ground truth cone positions for visualization"""
        # Convert dict to list format expected by publish_cone_array
        cones_list = []
        for cone_id, cone_data in self.ground_truth_cones.items():
            cones_list.append({
                'pos': cone_data['pos'],
                'type': cone_data['type'],
                'id': cone_id
            })
        
        publish_cone_array(
            self.gt_cones_pub,
            cones_list,
            namespace="ground_truth_cones",
            frame_id="map",
            with_text=False,  # GT cone에는 텍스트 제거
            timestamp=self.get_clock().now(),
            ground_truth=True
        )
    
    def publish_cones(self):
        """Publish cone observations with realistic errors"""
        msg = TrackedConeArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # Get visible cones from ground truth position
        visible_cones = self.get_visible_cones()
        
        # Track which cones are newly visible
        currently_visible_ids = set()
        self.last_detected_cones = []
        
        for cone_id, local_pos, cone_type in visible_cones:
            currently_visible_ids.add(cone_id)
            
            # Apply false negative (miss detection)
            if self.detection_errors_enabled and np.random.random() < self.false_negative_rate:
                continue  # Skip this cone
            
            # Get or assign track ID
            if cone_id not in self.cone_track_mapping:
                self.cone_track_mapping[cone_id] = self.next_track_id
                self.next_track_id += 1
            
            track_id = self.cone_track_mapping[cone_id]
            
            # Create tracked cone message
            tracked_cone = TrackedCone()
            tracked_cone.track_id = track_id
            
            # Add position noise
            noisy_pos = local_pos + np.random.normal(0, self.cone_position_noise, 2)
            tracked_cone.position.x = noisy_pos[0]
            tracked_cone.position.y = noisy_pos[1]
            tracked_cone.position.z = -0.15
            
            # Handle color errors
            detected_color = cone_type
            
            # Wrong color classification
            if self.detection_errors_enabled and np.random.random() < self.wrong_color_rate:
                # Randomly assign wrong color
                color_options = ['yellow', 'blue', 'red']
                color_options.remove(cone_type)  # Remove correct color
                detected_color = np.random.choice(color_options)
            
            # Unknown color (sensor fusion failure)
            elif self.detection_errors_enabled and np.random.random() < self.unknown_color_rate:
                detected_color = 'unknown'
            
            # Set color string
            if detected_color == 'yellow':
                tracked_cone.color = "Yellow cone"
            elif detected_color == 'blue':
                tracked_cone.color = "Blue cone"
            elif detected_color == 'red':
                tracked_cone.color = "Red cone"
            else:
                tracked_cone.color = "Unknown"
            
            msg.cones.append(tracked_cone)
            self.last_detected_cones.append((track_id, noisy_pos, detected_color))
        
        # Remove track IDs for cones no longer visible
        lost_cone_ids = set(self.cone_track_mapping.keys()) - currently_visible_ids
        for cone_id in lost_cone_ids:
            del self.cone_track_mapping[cone_id]
        
        # False positive detections
        if self.detection_errors_enabled:
            num_false_positives = np.random.poisson(self.false_positive_rate * len(self.ground_truth_cones))
            
            for _ in range(num_false_positives):
                # Generate random false detection within ROI
                if self.roi_type == 'sector':
                    # Random point in sector
                    r = np.random.uniform(2.0, self.detection_range)
                    theta = np.random.uniform(-self.fov_rad/2, self.fov_rad/2)
                    fake_local_x = r * np.cos(theta)
                    fake_local_y = r * np.sin(theta)
                else:
                    # Random point in rectangle
                    fake_local_x = np.random.uniform(2.0, self.detection_range)
                    max_y = self.detection_range * np.tan(self.fov_rad / 2)
                    fake_local_y = np.random.uniform(-max_y, max_y)
                
                # Create fake cone
                fake_cone = TrackedCone()
                fake_cone.track_id = self.next_track_id
                self.next_track_id += 1
                
                fake_cone.position.x = fake_local_x
                fake_cone.position.y = fake_local_y
                fake_cone.position.z = -0.15
                
                # Random color for fake cone
                fake_color = np.random.choice(['yellow', 'blue', 'red', 'unknown'])
                if fake_color == 'yellow':
                    fake_cone.color = "Yellow cone"
                elif fake_color == 'blue':
                    fake_cone.color = "Blue cone"
                elif fake_color == 'red':
                    fake_cone.color = "Red cone"
                else:
                    fake_cone.color = "Unknown"
                
                msg.cones.append(fake_cone)
                self.last_detected_cones.append((fake_cone.track_id, np.array([fake_local_x, fake_local_y]), fake_color))
        
        self.cone_pub.publish(msg)
        
        # Also publish visualization of detected cones
        self.publish_detected_cones_visualization()
    
    def get_visible_cones(self) -> List[Tuple[int, np.ndarray, str]]:
        """Get cones visible from ground truth robot pose"""
        visible = []
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            # Transform to ground truth robot frame
            dx = cone_data['pos'][0] - self.vehicle_state.position[0]
            dy = cone_data['pos'][1] - self.vehicle_state.position[1]
            
            # Rotate to robot frame
            theta = self.vehicle_state.orientation[2]
            local_x = dx * np.cos(-theta) - dy * np.sin(-theta)
            local_y = dx * np.sin(-theta) + dy * np.cos(-theta)
            
            # Check if in ROI
            if self.is_in_roi(local_x, local_y):
                visible.append((cone_id, np.array([local_x, local_y]), cone_data['type']))
        
        return visible
    
    def is_in_roi(self, x: float, y: float) -> bool:
        """Check if point is in Region of Interest"""
        if self.roi_type == 'sector':
            # Sector (fan) shape
            dist = np.sqrt(x*x + y*y)
            if dist > self.detection_range:
                return False
            
            angle = np.arctan2(y, x)
            if abs(angle) > self.fov_rad / 2:
                return False
            
            return True
        else:
            # Rectangle shape
            if x < 0 or x > self.detection_range:
                return False
            
            max_y = self.detection_range * np.tan(self.fov_rad / 2)
            if abs(y) > max_y:
                return False
            
            return True
    
    def publish_roi(self):
        """Publish ROI visualization"""
        roi_marker = VisualizationAdapter.create_roi_marker(
            self.roi_type,
            self.detection_range,
            self.fov_rad,
            frame_id="ground_truth_base_link",
            timestamp=self.get_clock().now()
        )
        self.roi_pub.publish(roi_marker)
    
    def publish_centerline(self):
        """Publish centerline visualization"""
        centerline_points = self.motion_controller.get_centerline_points()
        centerline_marker = VisualizationAdapter.create_path_marker(
            centerline_points,
            color="green",
            namespace="centerline",
            frame_id="map",
            timestamp=self.get_clock().now()
        )
        self.centerline_pub.publish(centerline_marker)
    
    def publish_detected_cones_visualization(self):
        """Publish visualization of detected cones"""
        if self.last_detected_cones:
            self.get_logger().debug(f"Publishing {len(self.last_detected_cones)} detected cones")
        publish_detected_cones(
            self.detected_cones_vis_pub,
            self.last_detected_cones,
            namespace="detected_cones",
            frame_id="base_link",
            timestamp=self.get_clock().now()
        )
    
    def publish_imu(self):
        """Publish IMU data using sensor simulator"""
        dt = 1.0 / self.imu_rate
        
        # Generate IMU message using sensor simulator
        imu_msg = self.sensor_sim.imu_sim.generate_imu_data(
            self.vehicle_state.linear_acceleration,
            self.vehicle_state.angular_velocity,
            quaternion_from_euler(
                self.vehicle_state.orientation[0],
                self.vehicle_state.orientation[1],
                self.vehicle_state.orientation[2]
            ),
            dt,
            self.get_clock().now()
        )
        
        self.imu_pub.publish(imu_msg)
    
    def publish_gps(self):
        """Publish GPS data using sensor simulator"""
        # Generate GPS message (for compatibility)
        gps_msg = self.sensor_sim.gps_sim.generate_gps_data(
            self.vehicle_state.position[0],
            self.vehicle_state.position[1],
            self.vehicle_state.position[2],
            self.get_clock().now()
        )
        
        self.gps_pub.publish(gps_msg)
        
        # Direct GPS odometry for robot_localization
        gps_odom_msg = Odometry()
        gps_odom_msg.header.stamp = self.get_clock().now().to_msg()
        gps_odom_msg.header.frame_id = "odom"  # GPS in odom frame for EKF
        gps_odom_msg.child_frame_id = "gps_link"
        
        # Get noise based on GPS mode
        noise_config = self.sensor_sim.gps_sim.config
        if noise_config.gps_mode == "rtk" or noise_config.gps_mode == "rtk_fix":
            h_noise = noise_config.gps_rtk_fix_noise_h
            v_noise = noise_config.gps_rtk_fix_noise_v
        elif noise_config.gps_mode == "rtk_float":
            h_noise = noise_config.gps_rtk_float_noise_h
            v_noise = noise_config.gps_rtk_float_noise_v
        elif noise_config.gps_mode == "dgps":
            h_noise = (noise_config.gps_single_noise_h + noise_config.gps_rtk_float_noise_h) / 2
            v_noise = (noise_config.gps_single_noise_v + noise_config.gps_rtk_float_noise_v) / 2
        else:  # single
            h_noise = noise_config.gps_single_noise_h
            v_noise = noise_config.gps_single_noise_v
        
        # Apply noise to position
        gps_odom_msg.pose.pose.position.x = self.vehicle_state.position[0] + np.random.normal(0, h_noise)
        gps_odom_msg.pose.pose.position.y = self.vehicle_state.position[1] + np.random.normal(0, h_noise)
        gps_odom_msg.pose.pose.position.z = self.vehicle_state.position[2] + np.random.normal(0, v_noise)
        
        # GPS doesn't provide orientation
        gps_odom_msg.pose.pose.orientation.w = 1.0
        
        # Set covariances
        pose_cov = np.zeros(36)
        pose_cov[0] = h_noise**2  # x
        pose_cov[7] = h_noise**2  # y
        pose_cov[14] = v_noise**2  # z
        # Large uncertainty for orientation (GPS doesn't measure it)
        pose_cov[21] = 1e6  # roll
        pose_cov[28] = 1e6  # pitch  
        pose_cov[35] = 1e6  # yaw
        gps_odom_msg.pose.covariance = pose_cov.tolist()
        
        # GPS velocity (optional)
        gps_vel_noise = 0.1  # m/s
        gps_odom_msg.twist.twist.linear.x = self.vehicle_state.linear_velocity[0] + np.random.normal(0, gps_vel_noise)
        gps_odom_msg.twist.twist.linear.y = self.vehicle_state.linear_velocity[1] + np.random.normal(0, gps_vel_noise)
        gps_odom_msg.twist.twist.linear.z = 0.0
        
        # Velocity covariance
        twist_cov = np.zeros(36)
        twist_cov[0] = gps_vel_noise**2  # vx
        twist_cov[7] = gps_vel_noise**2  # vy
        twist_cov[14] = 0.5**2  # vz (larger uncertainty)
        # No angular velocity from GPS
        twist_cov[21] = 1e6  # vroll
        twist_cov[28] = 1e6  # vpitch
        twist_cov[35] = 1e6  # vyaw
        gps_odom_msg.twist.covariance = twist_cov.tolist()
        
        self.gps_odom_pub.publish(gps_odom_msg)
        
        # Also publish GPS velocity as separate message (for compatibility)
        gps_vel_msg = TwistWithCovarianceStamped()
        gps_vel_msg.header.stamp = self.get_clock().now().to_msg()
        gps_vel_msg.header.frame_id = "gps_link"
        
        gps_vel_msg.twist.twist.linear.x = gps_odom_msg.twist.twist.linear.x
        gps_vel_msg.twist.twist.linear.y = gps_odom_msg.twist.twist.linear.y
        gps_vel_msg.twist.twist.linear.z = 0.0
        
        # Covariance
        gps_vel_msg.twist.covariance[0] = gps_vel_noise**2  # vx
        gps_vel_msg.twist.covariance[7] = gps_vel_noise**2  # vy
        gps_vel_msg.twist.covariance[14] = 0.5**2  # vz
        
        self.gps_vel_pub.publish(gps_vel_msg)
    
    def publish_odometry(self):
        """Publish odometry data at specified rate"""
        if not self.odom_sim_enabled:
            return
            
        dt = 1.0 / self.odom_rate
        odom_msg = self.sensor_sim.odom_sim.generate_odometry_data(
            self.vehicle_state.position[0],
            self.vehicle_state.position[1],
            self.vehicle_state.orientation[2],
            self.vehicle_state.linear_velocity[0],
            self.vehicle_state.linear_velocity[1],
            self.vehicle_state.angular_velocity[2],
            dt,
            self.get_clock().now()
        )
        
        self.odom_pub.publish(odom_msg)
        
        # Update path with noisy odometry
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose
        
        self.path.poses.append(pose_stamped)
        if len(self.path.poses) > 1000:  # Limit path length
            self.path.poses.pop(0)
        
        self.path_pub.publish(self.path)


def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
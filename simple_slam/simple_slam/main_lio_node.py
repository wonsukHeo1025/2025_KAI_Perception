#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import message_filters
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from custom_interface.msg import TrackedCone, TrackedConeArray
import numpy as np
import traceback

# Import modules from the package
from .ukf.ukf_slam import UKFSLAM # Assuming UKFSLAM class exists here
# from .ukf.ukf_base import UKF # If needed directly
from .ros_utils.visualization import VisualizationHandler # Assuming a class for visualization
from .ros_utils.tf_publisher import TFPublisher # Assuming a class for TF
from .ros_utils.parameters import ParameterHandler # Import the new handler
from .geometry.transforms import Transformations # Assuming a class/functions for transforms
from .geometry.svd import SvdSolver # Assuming a class/functions for SVD
# from .utils.common import some_utility_function # Example import
from .core.initialization_handler import InitializationHandler

# Additional imports needed for initialization logic
from scipy.spatial.transform import Rotation
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply, quaternion_conjugate, euler_matrix
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion

# Define state vector indices (adjust based on ukf_params['dim_x_robot'] and actual UKF implementation)
POS_IDX = 0
ORI_IDX = 3 # Assuming 3D position (x, y, z) -> index 3 for quaternion start (qx, qy, qz, qw)
VEL_IDX = 7 # Assuming quat is 4 elements -> index 7 for velocity start (vx, vy, vz)
ACC_BIAS_IDX = 10 # Assuming 3D velocity -> index 10 for accel bias (bx, by, bz)
GYRO_BIAS_IDX = 13 # Assuming 3D accel bias -> index 13 for gyro bias (bx, by, bz)
STATE_DIM = 16 # Example total dimension: 3(pos) + 4(quat) + 3(vel) + 3(acc_bias) + 3(gyro_bias) = 16

class MainLIONode(Node):
    def __init__(self):
        super().__init__('main_lio_node')
        self.get_logger().info("Initializing Main LIO Node...")

        # --- Parameter Handling (using ParameterHandler module) ---
        self.parameter_handler = ParameterHandler(self)

        # Get parameters using the handler
        core_params = self.parameter_handler.get_core_params()
        tf_static_params = self.parameter_handler.get_tf_static_params()
        ukf_params = self.parameter_handler.get_ukf_params()
        init_params = self.parameter_handler.get_init_params()

        # --- Use parameters obtained from handler ---
        self.odom_frame_id = core_params['odom_frame_id']
        self.base_link_frame_id = core_params['base_link_frame_id']
        self.map_frame_id = core_params['map_frame_id']
        imu_topic = core_params['imu_topic']
        cone_topic = core_params['cone_topic']
        sync_slop = core_params['sync_slop']
        self.min_known_landmarks_for_update = core_params['min_known_landmarks_for_update']

        # Store necessary UKF/Init params from the dictionaries
        self.robot_dim = ukf_params['dim_x_robot']
        if self.robot_dim != STATE_DIM:
             self.get_logger().warn(f"Parameter 'dim_x_robot' ({self.robot_dim}) does not match hardcoded STATE_DIM ({STATE_DIM}). Using parameter value, but ensure indices (POS_IDX, etc.) are correct.")
             # TODO: Consider dynamically setting indices based on parameter if structure is variable
        self.landmark_measurement_dim = ukf_params['dim_z_landmark']
        self.landmark_dim = ukf_params['landmark_dim']
        self.p0_params = ukf_params['P0']
        self.q_params = ukf_params['Q']
        self.r_params = ukf_params['R']
        # self.init_params = init_params # init_params will be passed directly to handler

        self.get_logger().info(f"Parameters loaded via ParameterHandler.")
        self.get_logger().info(f"Subscribing to IMU: {imu_topic}")
        self.get_logger().info(f"Subscribing to Cones: {cone_topic}")
        self.get_logger().info(f"Frame IDs: map='{self.map_frame_id}', odom='{self.odom_frame_id}', base='{self.base_link_frame_id}'")

        # --- QoS ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Consider RELIABLE if data loss is critical
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # =========================================================================
        # === Instantiate Modules ===
        # =========================================================================
        # TODO: Refine imports and initialization parameters based on actual module implementations
        # self.parameter_handler = ParameterHandler(self) # Already instantiated above

        # Get parameters using the handler - Already done above
        # core_params = self.parameter_handler.get_core_params()
        # tf_static_params = self.parameter_handler.get_tf_static_params()
        # ukf_params = self.parameter_handler.get_ukf_params()
        # init_params = self.parameter_handler.get_init_params()

        # Override parameters fetched earlier if using handler - No longer needed
        # self.odom_frame_id = core_params['odom_frame_id']
        # self.base_link_frame_id = core_params['base_link_frame_id']
        # self.map_frame_id = core_params['map_frame_id']
        # imu_topic = core_params['imu_topic']
        # cone_topic = core_params['cone_topic']
        # sync_slop = core_params['sync_slop']
        # self.min_known_landmarks_for_update = core_params['min_known_landmarks_for_update']

        self.tf_publisher = TFPublisher(self,
                                        self.map_frame_id,
                                        self.odom_frame_id,
                                        self.base_link_frame_id)
        # Use tf_static_params dictionary obtained earlier
        self.tf_publisher.publish_static_transforms(
            tf_static_params['sensor_to_imu']['trans'],
            tf_static_params['sensor_to_imu']['quat'],
            tf_static_params['sensor_to_lidar']['trans'],
            tf_static_params['sensor_to_lidar']['quat']
        )

        self.visualizer = VisualizationHandler(self, self.map_frame_id)

        # Define UKF state transition (fx) and measurement (hx) functions
        # These might need to be defined as methods of this class or imported
        def placeholder_fx(x, dt, **kwargs):
            # TODO: Implement actual state transition logic based on self.robot_dim
            # This should match the state definition [pos, ori, vel, acc_bias, gyro_bias]
            # Should use IMU measurements passed via kwargs if needed
            print(f"Warning: Placeholder fx called with dt={dt}, state_dim={len(x)}") # Add logging
            return x.copy()

        def placeholder_hx(x, landmark_idx, **kwargs):
            # TODO: Implement actual measurement prediction logic
            # Predicts measurement for a given landmark_idx
            # Needs robot state from x[:self.robot_dim]
            # Needs landmark state from x[self.robot_dim + landmark_idx * self.landmark_dim : ...]
            print(f"Warning: Placeholder hx called for landmark_idx={landmark_idx}") # Add logging
            # Return placeholder based on self.landmark_measurement_dim
            return np.zeros(self.landmark_measurement_dim)

        # TODO: Replace placeholders with actual functions
        self._fx = placeholder_fx
        self._hx = placeholder_hx

        # Initialize UKFSLAM
        # Use parameters stored in self attributes
        self.ukf_slam = UKFSLAM(
            dim_x=self.robot_dim, # Use stored robot dim
            dim_z=self.landmark_measurement_dim, # Use stored landmark measurement dim
            fx=self._fx,
            hx=self._hx,
            dt=0.1, # Initial dt, will be updated in predict step
            points=None, # Use default MerweScaledSigmaPoints if None
            landmark_dim=self.landmark_dim,
        )
        # P, Q, R are set later during initialization by the handler

        self.transformer = Transformations(self.tf_publisher.tf_buffer, self.get_logger()) # Pass buffer and logger
        self.svd_solver = SvdSolver(self.get_logger()) # Pass logger

        # =========================================================================
        # === Publisher Module ===
        # =========================================================================
        self.odom_pub = self.create_publisher(Odometry, '/odom_lio', 10)
        # self.marker_pub = self.visualizer.get_marker_publisher() # Or create directly if visualizer doesn't handle it

        # =========================================================================
        # === Input Synchronization Module ===
        # =========================================================================
        self.imu_sub = message_filters.Subscriber(self, Imu, imu_topic, qos_profile=qos_profile)
        self.cone_sub = message_filters.Subscriber(self, TrackedConeArray, cone_topic, qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.imu_sub, self.cone_sub],
            queue_size=15,
            slop=sync_slop
        )
        self.ts.registerCallback(self.synchronized_callback)

        # =========================================================================
        # === Node State Management Module ===
        # =========================================================================
        self.state = "IDLE" # Initial state before handler starts
        self.initialized = False
        self.last_imu_msg = None # Store the last IMU msg after initialization for dt calculation
        # self.initialization_data = {'imu': []} # Moved to handler
        self.get_logger().info(f"Node starting...")

        # =========================================================================
        # === State Estimation Variables ===
        # =========================================================================
        # TODO: These should be managed within the UKFSLAM instance
        # self.x = None # Managed by self.ukf_slam.x
        # self.P = None # Managed by self.ukf_slam.P
        self.map_landmarks = {} # Store landmark info {track_id: {'position': np.array, 'covariance': np.array, ...}}
        self.map_to_odom_transform = None # Will be calculated and set after initialization

        # Start initialization process using the handler
        self.initialization_handler = InitializationHandler(self, self.ukf_slam, init_params)
        self.initialization_handler.start()
        self.state = "INITIALIZING" # Update node state
        self.get_logger().info(f"Node state transitioned to: {self.state}")

        # Gravity vector (assuming standard gravity, will be refined in initialization)
        # Should be represented in the world frame (e.g., 'map' or 'odom' depending on context)
        # For prediction, we need it in the frame used for state propagation (likely odom)
        self.gravity = np.array([0.0, 0.0, -9.80665]) # Standard gravity


    # =========================================================================
    # === Callback Functions ===
    # =========================================================================
    def synchronized_callback(self, imu_msg, cones_msg):
        """Callback for synchronized IMU and Cone messages."""
        timestamp = self.get_clock().now().to_msg() # Or use message timestamp

        # --- Handle Initialization Phase --- #
        if not self.initialized:
            if self.state == "INITIALIZING":
                init_complete = self.initialization_handler.process_imu(imu_msg)
                if init_complete:
                    if self.initialization_handler.is_initialized():
                        self.initialized = True
                        self.state = "RUNNING"
                        self.last_imu_msg = self.initialization_handler.get_last_imu_msg()
                        # Initialize map->odom transform (usually identity at start)
                        self.map_to_odom_transform = TransformStamped()
                        self.map_to_odom_transform.header.stamp = self.get_clock().now().to_msg()
                        self.map_to_odom_transform.header.frame_id = self.map_frame_id
                        self.map_to_odom_transform.child_frame_id = self.odom_frame_id
                        self.map_to_odom_transform.transform.translation.x = 0.0
                        self.map_to_odom_transform.transform.translation.y = 0.0
                        self.map_to_odom_transform.transform.translation.z = 0.0
                        self.map_to_odom_transform.transform.rotation.w = 1.0

                        # Get initial state info if needed for gravity compensation refinement
                        self.gravity = self.initialization_handler.get_gravity_vector() # Get estimated gravity
                        self.get_logger().info(f"Initialization: Using gravity vector: {self.gravity}")

                        # Log success and initial state/biases from handler if needed
                        accel_bias, gyro_bias = self.initialization_handler.get_initial_biases()
                        init_orient_q = self.initialization_handler.get_initial_orientation()
                        try:
                            roll, pitch, yaw = euler_from_quaternion(init_orient_q)
                            self.get_logger().info(f"Initialization Complete. Initial RPY=[{np.degrees(roll):.2f}, {np.degrees(pitch):.2f}, {np.degrees(yaw):.2f}] deg")
                        except Exception:
                             self.get_logger().info("Initialization Complete. Could not log initial RPY.")
                        self.get_logger().info(f"Initial Biases: Accel={accel_bias}, Gyro={gyro_bias}")
                        self.get_logger().info(f"== Node successfully transitioned to RUNNING state ==")

                    elif self.initialization_handler.has_failed():
                        self.state = "FAILED"
                        self.get_logger().fatal("Initialization Failed! Node is in FAILED state. Stopping execution or specific handling needed.")
                        # rclpy.shutdown() # Option: shutdown on failure
                        # Or just stop processing callbacks
                    else:
                        # Should not happen if init_complete is True
                         self.get_logger().warn("Initialization handler reported completion but status is ambiguous.")
            # If state is not INITIALIZING but not initialized, something is wrong (e.g. FAILED)
            # In FAILED state, we just return and do nothing further.
            return

        # --- Handle Running Phase --- #
        if self.state != "RUNNING":
             self.get_logger().warn(f"Callback invoked but node is not in RUNNING state (current state: {self.state}). Skipping.")
             return

        self.get_logger().debug(f"Received synchronized messages: IMU t={imu_msg.header.stamp}, Cones t={cones_msg.header.stamp}")

        try:
            # 1. Prediction Step (using IMU)
            self.predict_step(imu_msg)

            # 2. Update Step (using Cones)
            self.update_step(cones_msg)

            # 3. Publish Results
            self.publish_results(timestamp) # Use consistent timestamp

        except Exception as e:
            self.get_logger().error(f"Error in synchronized_callback: {e}")
            self.get_logger().error(traceback.format_exc())
            # Optionally change state to FAILED
            # self.state = "FAILED"

    # =========================================================================
    # === Core Logic Methods (to be implemented using modules) ===
    # =========================================================================
    def predict_step(self, imu_msg):
        """Performs the UKF prediction step using IMU data and publishes predicted TF."""
        if self.state != "RUNNING": return

        if self.last_imu_msg is None:
            self.get_logger().warn("last_imu_msg is None in predict_step. Initializing.")
            self.last_imu_msg = imu_msg
            return

        # Calculate dt
        t_now = rclpy.time.Time.from_msg(imu_msg.header.stamp)
        t_last = rclpy.time.Time.from_msg(self.last_imu_msg.header.stamp)
        dt_duration = t_now - t_last
        dt = dt_duration.nanoseconds / 1e9

        if dt <= 0:
            self.get_logger().warn(f"Non-positive dt calculated ({dt:.4f}), skipping prediction. Current: {t_now.nanoseconds}, Last: {t_last.nanoseconds}")
            self.last_imu_msg = imu_msg # Update last_imu_msg anyway to prevent cycle
            return
        elif dt > 1.0: # Warn for large dt
            self.get_logger().warn(f"Large dt calculated ({dt:.4f}s), prediction might be inaccurate.")

        self.get_logger().debug(f"Performing prediction step with dt={dt:.4f}s...")

        # Extract IMU measurements
        accel = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])
        gyro = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        u = np.concatenate([accel, gyro])

        # Call UKF predict
        try:
            # Pass dt and u to predict. UKF handles calling fx with sigma points.
            # Ensure UKFSLAM.predict interface matches this call.
            self.ukf_slam.predict(dt=dt, u=u)
            self.get_logger().debug("UKF prediction successful.")

            # Publish the *predicted* TF after prediction
            self.publish_predicted_tf(imu_msg.header.stamp)

        except Exception as e:
             self.get_logger().error(f"Error during UKF prediction: {e}")
             self.get_logger().error(traceback.format_exc())
             # Optionally change state or handle error

        # Update last IMU message
        self.last_imu_msg = imu_msg


    def update_step(self, cones_msg):
        """Performs the UKF update step using observed cone landmarks."""
        self.get_logger().debug("Performing update step...")
        observed_cones = cones_msg.cones

        if not observed_cones:
            self.get_logger().debug("No cones observed, skipping update step.")
            return

        # TODO: Implement the update logic using UKFSLAM, SVD, Data Association
        # 1. Data Association: Match observed cones to existing landmarks in self.ukf_slam
        #    - Use self.ukf_slam.data_association(...)
        # 2. Estimate Pose with SVD (if enough matches):
        #    - Get associated landmark positions from self.ukf_slam.get_landmark_states()
        #    - Get observed cone positions (need to transform to map frame?)
        #    - Use self.svd_solver.calculate_pose(...)
        # 3. Update UKF State:
        #    - If SVD pose is reliable, use it as a measurement (or calculate innovation)
        #    - Call self.ukf_slam.update(...) with the SVD measurement or individual landmark updates
        # 4. Update Map / Add New Landmarks:
        #    - Update positions of matched landmarks in self.ukf_slam
        #    - Add new landmarks for unassociated cones using self.ukf_slam.add_landmark(...)
        #    - Prune landmarks if necessary using self.ukf_slam.prune_landmarks(...)

        pass # Placeholder


    def publish_predicted_tf(self, timestamp):
        """Publishes the predicted odom -> base_link TF based on UKF state."""
        if self.ukf_slam.x is None:
            self.get_logger().warn("Cannot publish predicted TF, UKF state (x) is None.")
            return

        # Extract predicted pose from the state vector
        pos = self.ukf_slam.x[POS_IDX : POS_IDX+3]
        quat = self.ukf_slam.x[ORI_IDX : ORI_IDX+4] # Assuming [qx, qy, qz, qw]

        # Ensure quaternion is valid before publishing
        if not np.all(np.isfinite(quat)) or np.isclose(np.linalg.norm(quat), 0.0):
            self.get_logger().warn(f"Invalid predicted quaternion: {quat}. Skipping TF publish.")
            return
        if not np.isclose(np.linalg.norm(quat), 1.0):
             self.get_logger().debug(f"Normalizing quaternion before publishing TF (norm={np.linalg.norm(quat):.4f})")
             quat = quat / np.linalg.norm(quat)

        # Create TransformStamped message
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.base_link_frame_id # Use base_link_frame_id from params

        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # Publish the transform
        self.tf_publisher.publish_dynamic_transform(t)
        self.get_logger().debug(f"Published predicted TF: {self.odom_frame_id} -> {self.base_link_frame_id}")


    def publish_results(self, timestamp):
        """Publishes final odometry, TF, and visualization markers (after correction)."""
        self.get_logger().debug("Publishing results...")

        # TODO: Get current state and covariance from self.ukf_slam
        # current_state = self.ukf_slam.x
        # current_covariance = self.ukf_slam.P

        # TODO: Publish Odometry using self.odom_pub
        # self.publish_odometry(timestamp, current_state, current_covariance)

        # TODO: Publish *Corrected* TF (map->odom and odom->base_link) using self.tf_publisher
        # The map->odom TF is handled separately (updated in Step 8)
        # The odom->base_link TF should reflect the corrected state here
        # For now, republishing based on current x (might be redundant with publish_predicted_tf if no update happened)
        if self.initialized and self.state == "RUNNING":
             # Example: Publish final odom->base TF (might overwrite predicted one)
             self.publish_predicted_tf(timestamp) # Re-using this for now, should be dedicated logic in Step 9

             # TODO: Publish map->odom TF (self.map_to_odom_transform)
             if self.map_to_odom_transform:
                 # Ensure timestamp is current
                 self.map_to_odom_transform.header.stamp = timestamp
                 self.tf_publisher.publish_dynamic_transform(self.map_to_odom_transform)

        # TODO: Publish Landmark Markers using self.visualizer
        # landmark_states = self.ukf_slam.get_landmark_states()
        # self.visualizer.publish_landmarks(timestamp, landmark_states)

        pass # Placeholder

    # =========================================================================
    # === Helper/Utility Methods (potentially move to utils module) ===
    # =========================================================================
    # Add helper functions as needed


def main(args=None):
    rclpy.init(args=args)
    main_lio_node = MainLIONode()
    try:
        rclpy.spin(main_lio_node)
    except KeyboardInterrupt:
        main_lio_node.get_logger().info("Shutting down node...")
    finally:
        main_lio_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
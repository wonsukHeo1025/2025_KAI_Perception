#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from sensor_msgs.msg import Imu
import traceback
from typing import Tuple, Optional

# Assuming UKFSLAM is importable relative to this file's location
# Adjust the import path if core is not directly under simple_slam
from ..ukf.ukf_slam import UKFSLAM

class InitializationHandler:
    """Handles the explicit initialization phase for the LIO node."""

    STATE_IDLE = 0
    STATE_INITIALIZING = 1
    STATE_SUCCEEDED = 2
    STATE_FAILED = 3

    def __init__(self, node: Node, ukf_slam: UKFSLAM, init_params: dict):
        """
        Initialize the InitializationHandler.

        Args:
            node (Node): The main LIO node instance.
            ukf_slam (UKFSLAM): The UKFSLAM instance to initialize.
            init_params (dict): Dictionary containing initialization parameters
                                (num_imu_samples, gravity_magnitude, etc.).
        """
        self.node = node
        self.ukf_slam = ukf_slam
        self.init_params = init_params
        self.logger = node.get_logger().get_child('InitializationHandler')

        # Initialization state variables
        self.state = self.STATE_IDLE
        self.imu_data = {'accel': [], 'gyro': [], 'timestamps': []}
        self.initial_gravity_vec_world = None
        self.initial_orientation_q = None # R_ws (world to sensor)
        self.accel_bias_init = np.zeros(3)
        self.gyro_bias_init = np.zeros(3)
        self.last_imu_msg_for_dt = None # Store the last IMU message used for transition

    def start(self):
        """Begins the explicit initialization phase."""
        if self.state != self.STATE_IDLE:
            self.logger.warn("Initialization already started or completed.")
            return

        self.state = self.STATE_INITIALIZING
        num_samples = self.init_params.get('num_imu_samples', 100) # Default if missing
        self.logger.info(f"Starting explicit initialization: Waiting for {num_samples} IMU samples...")
        # Reset internal state
        self.imu_data = {'accel': [], 'gyro': [], 'timestamps': []}
        self.initial_gravity_vec_world = None
        self.initial_orientation_q = None
        self.accel_bias_init = np.zeros(3)
        self.gyro_bias_init = np.zeros(3)
        self.last_imu_msg_for_dt = None

    def process_imu(self, imu_msg: Imu) -> bool:
        """
        Processes an IMU message during the initialization phase.

        Args:
            imu_msg (Imu): The incoming IMU message.

        Returns:
            bool: True if initialization is complete (succeeded or failed), False otherwise.
        """
        if self.state != self.STATE_INITIALIZING:
            # Either not started, or already finished/failed
            return self.state == self.STATE_SUCCEEDED or self.state == self.STATE_FAILED

        accel = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
        gyro = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
        timestamp = rclpy.time.Time.from_msg(imu_msg.header.stamp)

        # Check for NaN or Inf values
        if np.any(np.isnan(accel)) or np.any(np.isinf(accel)) or \
           np.any(np.isnan(gyro)) or np.any(np.isinf(gyro)):
            self.logger.warn("Invalid IMU data (NaN or Inf) received during initialization. Skipping sample.")
            return False # Still initializing

        self.imu_data['accel'].append(accel)
        self.imu_data['gyro'].append(gyro)
        self.imu_data['timestamps'].append(timestamp)

        num_samples = len(self.imu_data['accel'])
        required_samples = self.init_params.get('num_imu_samples', 100)

        if num_samples % 20 == 0:
             self.logger.info(f"Collected {num_samples}/{required_samples} IMU samples for initialization.")

        if num_samples >= required_samples:
            self.logger.info("Sufficient IMU samples collected. Performing initialization checks.")
            if self._perform_initialization_checks():
                # Checks passed, initialize UKF
                if self._initialize_ukf():
                    self.state = self.STATE_SUCCEEDED
                    # Store the last message for the main node to calculate the first dt
                    self.last_imu_msg_for_dt = imu_msg
                    self.logger.info("Initialization successful.")
                    return True # Initialization complete (Success)
                else:
                    # UKF initialization failed
                    self.state = self.STATE_FAILED
                    self.logger.error("UKF Initialization step failed.")
                    return True # Initialization complete (Failure)
            else:
                # Initial checks failed
                self.state = self.STATE_FAILED
                self.logger.error("Initial sensor checks failed.")
                return True # Initialization complete (Failure)

        return False # Still collecting samples

    def _perform_initialization_checks(self) -> bool:
        """Performs gravity and bias checks on accumulated IMU data."""
        gravity_magnitude = self.init_params.get('gravity_magnitude', 9.81)
        gravity_tolerance = self.init_params.get('gravity_tolerance', 0.5)
        bias_stability_threshold = self.init_params.get('bias_stability_threshold', 0.01)

        # 1. Estimate Gravity and Initial Orientation
        try:
            accel_data = np.array(self.imu_data['accel'])
            mean_accel_sensor = np.mean(accel_data, axis=0)
            gravity_norm_measured = np.linalg.norm(mean_accel_sensor)
        except Exception as e:
            self.logger.error(f"Error processing accumulated IMU accel data: {e}")
            return False

        self.logger.info(f"Mean acceleration norm: {gravity_norm_measured:.3f} (Target: {gravity_magnitude})")

        # Check gravity magnitude
        if abs(gravity_norm_measured - gravity_magnitude) > gravity_tolerance:
            self.logger.error(f"Gravity magnitude check failed ({gravity_norm_measured:.2f} vs {gravity_magnitude:.2f}).")
            return False

        if gravity_norm_measured < 1e-6:
             self.logger.error("Mean acceleration norm is near zero. Cannot determine gravity vector.")
             return False

        measured_gravity_dir_sensor = -mean_accel_sensor / gravity_norm_measured
        self.initial_gravity_vec_world = np.array([0.0, 0.0, -gravity_magnitude])
        world_gravity_dir = self.initial_gravity_vec_world / gravity_magnitude # Normalized [0, 0, -1]

        self.logger.info(f"Normalized gravity vector (sensor frame): {measured_gravity_dir_sensor}")

        # Estimate initial orientation (Roll, Pitch)
        try:
            rotation_object, _ = Rotation.align_vectors([measured_gravity_dir_sensor], [world_gravity_dir])
            initial_orientation_raw_q = rotation_object.as_quat() # R_ws

            roll, pitch, _ = rotation_object.as_euler('xyz', degrees=False) # Yaw is undetermined
            self.logger.info(f"Estimated initial Roll: {np.degrees(roll):.2f}, Pitch: {np.degrees(pitch):.2f} degrees.")

            self.initial_orientation_q = quaternion_from_euler(roll, pitch, 0.0) # Zero Yaw R_ws
            self.logger.info(f"Initial orientation quaternion R_ws (zero yaw): {self.initial_orientation_q}")

        except Exception as e:
             self.logger.error(f"Failed to estimate initial orientation: {e}\n{traceback.format_exc()}")
             return False

        # 2. Estimate Initial Gyro Bias
        try:
            gyro_data = np.array(self.imu_data['gyro'])
            self.gyro_bias_init = np.mean(gyro_data, axis=0)
            gyro_var = np.var(gyro_data, axis=0)
            if np.any(gyro_var > bias_stability_threshold):
                self.logger.warn(f"Initial gyro measurements variance ({gyro_var}) exceeds stability threshold ({bias_stability_threshold}).")
            self.logger.info(f"Estimated initial Gyro bias: {self.gyro_bias_init} (Variance: {gyro_var})")
        except Exception as e:
             self.logger.error(f"Error processing accumulated IMU gyro data: {e}")
             return False

        # Accel bias is estimated as zero for now
        self.accel_bias_init = np.zeros(3)
        self.logger.info(f"Initial Accel bias assumed: {self.accel_bias_init}")

        # All checks passed
        return True

    def _initialize_ukf(self) -> bool:
         """Initializes the UKF state vector and covariance matrix."""
         try:
             dim_x_robot = self.ukf_slam.robot_dim
             # Access parameter handler via the node reference
             p0_params = self.node.parameter_handler.get_ukf_params()['P0'] # Get P0 from handler
             q_params = self.node.parameter_handler.get_ukf_params()['Q'] # Get Q from handler
             r_params = self.node.parameter_handler.get_ukf_params()['R'] # Get R from handler
             landmark_measurement_dim = self.node.parameter_handler.get_ukf_params()['dim_z_landmark']

         except AttributeError as e:
             self.logger.fatal(f"Node is missing parameter_handler or UKF parameter getter: {e}")
             return False
         except KeyError as e:
             self.logger.fatal(f"Missing key in UKF parameters (P0/Q/R): {e}. Check parameter handler.")
             return False
         except Exception as e:
             self.logger.fatal(f"Failed to get UKF parameters for initialization: {e}")
             return False

         # Initialize state vector x
         initial_state = np.zeros(dim_x_robot)
         try:
             initial_roll, initial_pitch, initial_yaw = euler_from_quaternion(self.initial_orientation_q)
         except Exception as e:
             self.logger.error(f"Failed to get Euler angles from initial quaternion {self.initial_orientation_q}: {e}. Resetting orientation.")
             initial_roll, initial_pitch, initial_yaw = 0.0, 0.0, 0.0

         # Fill initial state based on estimated values and state dimension
         if dim_x_robot >= 6:
             initial_state[3:6] = [initial_roll, initial_pitch, initial_yaw]
         if dim_x_robot >= 12:
             initial_state[9:12] = self.accel_bias_init
         if dim_x_robot >= 15:
             initial_state[12:15] = self.gyro_bias_init

         # Initialize covariance matrix P
         try:
             initial_P = np.diag([
                 p0_params['pos'], p0_params['pos'], p0_params['pos'],
                 p0_params['ori'], p0_params['ori'], p0_params['ori'],
                 p0_params['vel'], p0_params['vel'], p0_params['vel'],
                 p0_params['acc_bias'], p0_params['acc_bias'], p0_params['acc_bias'],
                 p0_params['gyro_bias'], p0_params['gyro_bias'], p0_params['gyro_bias']
             ])
             if initial_P.shape[0] != dim_x_robot:
                  self.logger.error(f"Initial P dimension ({initial_P.shape[0]}) != robot state dimension ({dim_x_robot}).")
                  return False
         except KeyError as e:
             self.logger.fatal(f"Missing key in P0 parameters: {e}.")
             return False

         # Initialize Q matrix
         try:
             q_diag = np.array([
                 q_params['pos'], q_params['pos'], q_params['pos'],
                 q_params['ori'], q_params['ori'], q_params['ori'],
                 q_params['vel'], q_params['vel'], q_params['vel'],
                 q_params['acc_bias'], q_params['acc_bias'], q_params['acc_bias'],
                 q_params['gyro_bias'], q_params['gyro_bias'], q_params['gyro_bias']
             ])
             if len(q_diag) != dim_x_robot:
                 self.logger.error(f"Q diagonal length ({len(q_diag)}) != robot state dimension ({dim_x_robot}).")
                 return False
             Q_matrix = np.diag(q_diag)
         except KeyError as e:
             self.logger.fatal(f"Missing key in Q parameters: {e}.")
             return False

         # Initialize R matrix
         try:
             # Need to confirm R dimension based on actual landmark measurement
             # Assuming pose [x,y,z, r,p,y] -> 6 dim
             if landmark_measurement_dim == 6:
                 r_diag = np.array([
                     r_params['pos'], r_params['pos'], r_params['pos'],
                     r_params['ori'], r_params['ori'], r_params['ori']
                 ])
             elif landmark_measurement_dim == 3: # Assuming position only [x,y,z]
                  r_diag = np.array([
                     r_params['pos'], r_params['pos'], r_params['pos']
                 ])
             else:
                 self.logger.error(f"Unsupported landmark measurement dimension for R matrix: {landmark_measurement_dim}")
                 return False

             if len(r_diag) != landmark_measurement_dim:
                 self.logger.error(f"R diagonal length ({len(r_diag)}) != landmark measurement dimension ({landmark_measurement_dim}).")
                 return False
             R_matrix = np.diag(r_diag)
         except KeyError as e:
             self.logger.fatal(f"Missing key in R parameters: {e}.")
             return False


         # Call UKFSLAM's initialize method and set Q, R
         try:
             self.ukf_slam.initialize(initial_state, initial_P)
             self.ukf_slam.Q = Q_matrix
             self.ukf_slam.R = R_matrix
             self.logger.info(f"UKFSLAM initialized successfully. State: {self.ukf_slam.x}, State dim: {self.ukf_slam.dim_x}")
             self.logger.info(f"UKF Q set. Shape: {self.ukf_slam.Q.shape}")
             self.logger.info(f"UKF R set. Shape: {self.ukf_slam.R.shape}")
             return True
         except Exception as e:
             self.logger.error(f"Failed to initialize UKFSLAM instance: {e}\n{traceback.format_exc()}")
             return False

    def is_initialized(self) -> bool:
        """Check if initialization has succeeded."""
        return self.state == self.STATE_SUCCEEDED

    def has_failed(self) -> bool:
         """Check if initialization has failed."""
         return self.state == self.STATE_FAILED

    def get_last_imu_msg(self) -> Optional[Imu]:
        """Return the last IMU message used for completing initialization."""
        return self.last_imu_msg_for_dt

    def get_initial_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the estimated initial biases."""
        return self.accel_bias_init, self.gyro_bias_init

    def get_initial_orientation(self) -> Optional[np.ndarray]:
         """Return the estimated initial orientation quaternion (R_ws)."""
         return self.initial_orientation_q

    # transition_to_running is no longer needed here, the main node will handle the state change based on is_initialized() 
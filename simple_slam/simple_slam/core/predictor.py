#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import numpy as np
import traceback

class Predictor:
    """Handles the UKF prediction step logic."""

    def __init__(self, logger, ukf_slam):
        """
        Initialize the Predictor.

        Args:
            logger: ROS 2 Node logger instance.
            ukf_slam: The UKFSLAM instance to perform predictions on.
        """
        self._logger = logger
        self._ukf_slam = ukf_slam

    def predict(self, imu_msg, last_imu_msg):
        """
        Performs the UKF prediction step using IMU data.

        Args:
            imu_msg: The current sensor_msgs/Imu message.
            last_imu_msg: The previous sensor_msgs/Imu message.

        Returns:
            float: The calculated time difference (dt), or None if prediction skipped.
                   Returns -1.0 if dt is non-positive.
        """
        if last_imu_msg is None:
            self._logger.warn("last_imu_msg is None in predictor.predict(). Skipping initial prediction.")
            return None # Indicate skip

        # Calculate dt
        try:
            t_now = rclpy.time.Time.from_msg(imu_msg.header.stamp)
            t_last = rclpy.time.Time.from_msg(last_imu_msg.header.stamp)
            dt_duration = t_now - t_last
            dt = dt_duration.nanoseconds / 1e9
        except Exception as e:
             self._logger.error(f"Error converting timestamps: {e}")
             return None # Indicate skip due to error


        if dt <= 0:
            self._logger.warn(f"Predictor: Non-positive dt calculated ({dt:.4f}), skipping prediction. Current: {t_now.nanoseconds}, Last: {t_last.nanoseconds}")
            return -1.0 # Indicate non-positive dt
        elif dt > 1.0: # Warn for large dt
            self._logger.warn(f"Predictor: Large dt calculated ({dt:.4f}s), prediction might be inaccurate.")

        self._logger.debug(f"Predictor: Performing prediction step with dt={dt:.4f}s...")

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
            self._ukf_slam.predict(dt=dt, u=u)
            self._logger.debug("Predictor: UKF prediction successful.")
            return dt # Return the calculated dt
        except Exception as e:
             self._logger.error(f"Predictor: Error during UKF prediction: {e}")
             self._logger.error(traceback.format_exc())
             return None # Indicate skip due to error 
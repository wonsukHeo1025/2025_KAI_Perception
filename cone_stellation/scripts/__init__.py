#!/usr/bin/env python3
"""
ConeSTELLATION Simulation Scripts

Contains simulation components adapted from cc_slam_sym for testing:
- Dummy publisher: Simulates sensor data with realistic noise models
- Cone definitions: Ground truth track layouts
- Sensor simulators: IMU, GPS, odometry with noise
- Motion controller: Vehicle dynamics simulation
"""

# Make scripts available as a module when imported
try:
    from .dummy_publisher_node import DummyPublisher
    from .cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2
    from .sensor_simulator import SensorSimulator, SensorNoiseConfig, ImuSimulator, GpsSimulator, OdometrySimulator
    from .motion_controller import MotionController, VehicleState, MotionScenario, TrajectoryGenerator
except ImportError:
    # When running scripts directly
    pass
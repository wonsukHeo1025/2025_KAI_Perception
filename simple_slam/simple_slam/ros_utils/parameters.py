#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 parameter management utilities for the Simple SLAM package.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple, Union
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult


class ParameterManager:
    """
    A class to help manage ROS2 parameters for a node.
    
    This class provides utilities for declaring, getting, and validating parameters.
    It also handles parameter change callbacks.
    """
    
    def __init__(self, node: Node):
        """
        Initialize the ParameterManager.
        
        Args:
            node (Node): The ROS2 node to manage parameters for
        """
        self.node = node
        self.param_validators: Dict[str, Callable[[Any], Tuple[bool, str]]] = {}
        self.param_updaters: Dict[str, Callable[[str, Any], None]] = {}
        self.param_descriptions: Dict[str, str] = {}
        self.param_value_descriptions: Dict[str, Dict[Any, str]] = {}
        
        # Set up the parameter callback
        self.node.add_on_set_parameters_callback(self._parameter_callback)
    
    def declare_parameter(self, name: str, default_value: Any, 
                          description: str = "", validator: Optional[Callable[[Any], Tuple[bool, str]]] = None,
                          updater: Optional[Callable[[str, Any], None]] = None,
                          value_descriptions: Optional[Dict[Any, str]] = None):
        """
        Declare a parameter with validation and update handling.
        
        Args:
            name (str): Parameter name
            default_value (Any): Default value for the parameter
            description (str): Human-readable description of the parameter
            validator (Optional[Callable]): Function to validate parameter values
                                           Should return (is_valid, reason)
            updater (Optional[Callable]): Function to call when parameter is updated
                                         Will be called with (name, value)
            value_descriptions (Optional[Dict]): Dictionary mapping values to descriptions
        """
        descriptor = ParameterDescriptor(description=description)
        self.node.declare_parameter(name, default_value, descriptor)
        
        if validator is not None:
            self.param_validators[name] = validator
        if updater is not None:
            self.param_updaters[name] = updater
        if description:
            self.param_descriptions[name] = description
        if value_descriptions:
            self.param_value_descriptions[name] = value_descriptions
    
    def declare_parameters(self, namespace: str, parameters: Dict[str, Any],
                           descriptions: Optional[Dict[str, str]] = None,
                           validators: Optional[Dict[str, Callable]] = None,
                           updaters: Optional[Dict[str, Callable]] = None,
                           value_descriptions: Optional[Dict[str, Dict[Any, str]]] = None):
        """
        Declare multiple parameters with the same namespace.
        
        Args:
            namespace (str): Parameter namespace
            parameters (Dict[str, Any]): Dictionary of parameter names to default values
            descriptions (Optional[Dict[str, str]]): Dictionary of parameter names to descriptions
            validators (Optional[Dict[str, Callable]]): Dictionary of parameter names to validator functions
            updaters (Optional[Dict[str, Callable]]): Dictionary of parameter names to updater functions
            value_descriptions (Optional[Dict[str, Dict[Any, str]]]): Dictionary of parameter names to value description dictionaries
        """
        for name, default_value in parameters.items():
            full_name = f"{namespace}.{name}" if namespace else name
            description = descriptions.get(name, "") if descriptions else ""
            validator = validators.get(name) if validators else None
            updater = updaters.get(name) if updaters else None
            value_desc = value_descriptions.get(name) if value_descriptions else None
            self.declare_parameter(full_name, default_value, description, validator, updater, value_desc)
    
    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter value.
        
        Args:
            name (str): Parameter name
            
        Returns:
            Any: Parameter value
        """
        return self.node.get_parameter(name).value
    
    def get_parameters(self, names: List[str]) -> Dict[str, Any]:
        """
        Get multiple parameter values.
        
        Args:
            names (List[str]): List of parameter names
            
        Returns:
            Dict[str, Any]: Dictionary of parameter names to values
        """
        return {name: self.get_parameter(name) for name in names}
    
    def get_parameters_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        Get all parameters with a given prefix.
        
        Args:
            prefix (str): Parameter name prefix
            
        Returns:
            Dict[str, Any]: Dictionary of parameter names to values
        """
        param_names = [
            param.name for param in self.node.get_parameters_by_prefix(prefix)
        ]
        return {name: self.node.get_parameter(name).value for name in param_names}
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """
        Set a parameter value.
        
        Args:
            name (str): Parameter name
            value (Any): New parameter value
            
        Returns:
            bool: True if the parameter was set successfully, False otherwise
        """
        result = self.node.set_parameters([Parameter(name=name, value=value)])[0]
        return result.successful
    
    def _parameter_callback(self, parameters):
        """
        Callback for parameter changes.
        
        Validates parameters and calls updaters if they exist.
        
        Args:
            parameters (List[Parameter]): List of parameters that changed
            
        Returns:
            SetParametersResult: Result of parameter setting
        """
        result = SetParametersResult(successful=True)
        
        for param in parameters:
            name = param.name
            value = param.value
            
            # Validate if a validator exists
            if name in self.param_validators:
                is_valid, reason = self.param_validators[name](value)
                if not is_valid:
                    self.node.get_logger().error(f"Invalid parameter value for '{name}': {reason}")
                    result.successful = False
                    result.reason = f"Invalid parameter value for '{name}': {reason}"
                    return result
            
            # Call updater if it exists and parameter is valid
            if name in self.param_updaters:
                try:
                    self.param_updaters[name](name, value)
                except Exception as e:
                    self.node.get_logger().error(f"Error updating parameter '{name}': {e}")
                    result.successful = False
                    result.reason = f"Error updating parameter '{name}': {e}"
                    return result
        
        return result
    
    def list_parameters(self) -> List[str]:
        """
        Get a list of all parameter names.
        
        Returns:
            List[str]: List of parameter names
        """
        return [p.name for p in self.node.get_parameters([])]
    
    def get_parameter_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a parameter.
        
        Args:
            name (str): Parameter name
            
        Returns:
            Dict[str, Any]: Dictionary with parameter information
        """
        param = self.node.get_parameter(name)
        info = {
            "name": param.name,
            "type": param.type_,
            "value": param.value,
        }
        
        if name in self.param_descriptions:
            info["description"] = self.param_descriptions[name]
        
        if name in self.param_value_descriptions and param.value in self.param_value_descriptions[name]:
            info["value_description"] = self.param_value_descriptions[name][param.value]
        
        return info


# Common validator functions

def range_validator(min_val: Union[int, float], max_val: Union[int, float]) -> Callable[[Any], Tuple[bool, str]]:
    """
    Create a validator function that checks if a value is within a range.
    
    Args:
        min_val (Union[int, float]): Minimum allowed value
        max_val (Union[int, float]): Maximum allowed value
        
    Returns:
        Callable[[Any], Tuple[bool, str]]: Validator function
    """
    def validator(value: Any) -> Tuple[bool, str]:
        if not isinstance(value, (int, float)):
            return False, f"Expected int or float, got {type(value).__name__}"
        if value < min_val or value > max_val:
            return False, f"Value {value} is outside range [{min_val}, {max_val}]"
        return True, ""
    return validator


def enum_validator(allowed_values: List[Any]) -> Callable[[Any], Tuple[bool, str]]:
    """
    Create a validator function that checks if a value is one of a set of allowed values.
    
    Args:
        allowed_values (List[Any]): List of allowed values
        
    Returns:
        Callable[[Any], Tuple[bool, str]]: Validator function
    """
    def validator(value: Any) -> Tuple[bool, str]:
        if value not in allowed_values:
            return False, f"Value {value} is not one of {allowed_values}"
        return True, ""
    return validator


def type_validator(allowed_types: List[type]) -> Callable[[Any], Tuple[bool, str]]:
    """
    Create a validator function that checks if a value is of one of a set of allowed types.
    
    Args:
        allowed_types (List[type]): List of allowed types
        
    Returns:
        Callable[[Any], Tuple[bool, str]]: Validator function
    """
    def validator(value: Any) -> Tuple[bool, str]:
        if not any(isinstance(value, t) for t in allowed_types):
            type_names = [t.__name__ for t in allowed_types]
            return False, f"Value {value} is not of any allowed type: {type_names}"
        return True, ""
    return validator


def list_length_validator(min_length: int, max_length: Optional[int] = None) -> Callable[[Any], Tuple[bool, str]]:
    """
    Create a validator function that checks if a list has an allowed length.
    
    Args:
        min_length (int): Minimum allowed length
        max_length (Optional[int]): Maximum allowed length, or None for no maximum
        
    Returns:
        Callable[[Any], Tuple[bool, str]]: Validator function
    """
    def validator(value: Any) -> Tuple[bool, str]:
        if not isinstance(value, list):
            return False, f"Expected list, got {type(value).__name__}"
        
        if len(value) < min_length:
            return False, f"List length {len(value)} is less than minimum {min_length}"
        
        if max_length is not None and len(value) > max_length:
            return False, f"List length {len(value)} is greater than maximum {max_length}"
        
        return True, ""
    return validator


class ParameterHandler:
    """Handles parameter declaration and retrieval for the LIO node."""
    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()
        self._declare_parameters()

    def _declare_parameters(self):
        self.logger.info("Declaring parameters...")
        # Core Parameters
        self.node.declare_parameter('odom_frame_id', 'odom')
        self.node.declare_parameter('base_link_frame_id', 'os_sensor')
        self.node.declare_parameter('map_frame_id', 'map')
        self.node.declare_parameter('imu_topic', '/ouster/imu')
        self.node.declare_parameter('cone_topic', '/fused_sorted_cones_ukf')
        self.node.declare_parameter('sync_slop', 0.1)
        self.node.declare_parameter('min_known_landmarks_for_update', 3)

        # Static TF Parameters
        self.node.declare_parameter('tf_static.sensor_to_imu.trans', [0.006253, -0.011775, 0.007645])
        self.node.declare_parameter('tf_static.sensor_to_imu.quat', [0.0, 0.0, 0.0, 1.0])
        self.node.declare_parameter('tf_static.sensor_to_lidar.trans', [0.0, 0.0, 0.038195])
        self.node.declare_parameter('tf_static.sensor_to_lidar.quat', [0.0, 0.0, 1.0, 0.0])

        # UKF Parameters
        self.node.declare_parameter('ukf.dim_x_robot', 15)
        self.node.declare_parameter('ukf.dim_z_landmark', 6)
        self.node.declare_parameter('ukf.landmark_dim', 3)
        # UKF P0
        self.node.declare_parameter('ukf.P0.pos', 0.1)
        self.node.declare_parameter('ukf.P0.ori', 0.05)
        self.node.declare_parameter('ukf.P0.vel', 0.5)
        self.node.declare_parameter('ukf.P0.acc_bias', 0.01)
        self.node.declare_parameter('ukf.P0.gyro_bias', 0.005)
        # UKF Q
        self.node.declare_parameter('ukf.Q.pos', 0.05)
        self.node.declare_parameter('ukf.Q.ori', 0.01)
        self.node.declare_parameter('ukf.Q.vel', 0.1)
        self.node.declare_parameter('ukf.Q.acc_bias', 0.0001)
        self.node.declare_parameter('ukf.Q.gyro_bias', 0.00005)
        # UKF R
        self.node.declare_parameter('ukf.R.pos', 0.1)
        self.node.declare_parameter('ukf.R.ori', 0.05)
        # Add other UKF params like alpha, beta, kappa if needed

        # Initialization Parameters
        self.node.declare_parameter('init.num_imu_samples', 100)
        self.node.declare_parameter('init.gravity_magnitude', 9.81)
        self.node.declare_parameter('init.gravity_tolerance', 0.5)
        self.node.declare_parameter('init.bias_stability_threshold', 0.01)

        self.logger.info("Parameters declared.")

    def _get_param(self, name: str):
        try:
            return self.node.get_parameter(name).value
        except Exception as e:
            self.logger.error(f"Failed to get parameter '{name}': {e}")
            # Return a default or raise an exception depending on criticality
            return None # Or raise e

    def get_core_params(self) -> dict:
        """Returns core node parameters as a dictionary."""
        return {
            'odom_frame_id': self._get_param('odom_frame_id'),
            'base_link_frame_id': self._get_param('base_link_frame_id'),
            'map_frame_id': self._get_param('map_frame_id'),
            'imu_topic': self._get_param('imu_topic'),
            'cone_topic': self._get_param('cone_topic'),
            'sync_slop': self._get_param('sync_slop'),
            'min_known_landmarks_for_update': self._get_param('min_known_landmarks_for_update'),
        }

    def get_tf_static_params(self) -> dict:
        """Returns static TF parameters as a nested dictionary."""
        return {
            'sensor_to_imu': {
                'trans': self._get_param('tf_static.sensor_to_imu.trans'),
                'quat': self._get_param('tf_static.sensor_to_imu.quat')
            },
            'sensor_to_lidar': {
                'trans': self._get_param('tf_static.sensor_to_lidar.trans'),
                'quat': self._get_param('tf_static.sensor_to_lidar.quat')
            }
        }

    def get_ukf_params(self) -> dict:
        """Returns UKF parameters as a nested dictionary."""
        return {
            'dim_x_robot': self._get_param('ukf.dim_x_robot'),
            'dim_z_landmark': self._get_param('ukf.dim_z_landmark'),
            'landmark_dim': self._get_param('ukf.landmark_dim'),
            'P0': {
                'pos': self._get_param('ukf.P0.pos'),
                'ori': self._get_param('ukf.P0.ori'),
                'vel': self._get_param('ukf.P0.vel'),
                'acc_bias': self._get_param('ukf.P0.acc_bias'),
                'gyro_bias': self._get_param('ukf.P0.gyro_bias')
            },
            'Q': {
                'pos': self._get_param('ukf.Q.pos'),
                'ori': self._get_param('ukf.Q.ori'),
                'vel': self._get_param('ukf.Q.vel'),
                'acc_bias': self._get_param('ukf.Q.acc_bias'),
                'gyro_bias': self._get_param('ukf.Q.gyro_bias')
            },
            'R': {
                'pos': self._get_param('ukf.R.pos'),
                'ori': self._get_param('ukf.R.ori')
            }
            # Add alpha, beta, kappa retrieval if declared
        }

    def get_init_params(self) -> dict:
        """Returns initialization parameters as a dictionary."""
        return {
            'num_imu_samples': self._get_param('init.num_imu_samples'),
            'gravity_magnitude': self._get_param('init.gravity_magnitude'),
            'gravity_tolerance': self._get_param('init.gravity_tolerance'),
            'bias_stability_threshold': self._get_param('init.bias_stability_threshold'),
        } 
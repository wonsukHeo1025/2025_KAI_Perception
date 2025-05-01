#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TF publishing utilities for the Simple SLAM package.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Transform
from simple_slam.utils.common import matrix_to_transform


class TFPublisher:
    """
    A class to manage publishing transforms to the TF tree.
    """
    
    def __init__(self, node: Node):
        """
        Initialize the TF publisher.
        
        Args:
            node (Node): The ROS2 node to publish from
        """
        self.node = node
        self.broadcaster = TransformBroadcaster(node)
        self.static_broadcaster = StaticTransformBroadcaster(node)
        self.static_transforms: Dict[str, TransformStamped] = {}
    
    def publish_transform(self, transform: TransformStamped):
        """
        Publish a single transform.
        
        Args:
            transform (TransformStamped): The transform to publish
        """
        self.broadcaster.sendTransform(transform)
    
    def publish_transforms(self, transforms: List[TransformStamped]):
        """
        Publish multiple transforms at once.
        
        Args:
            transforms (List[TransformStamped]): The transforms to publish
        """
        self.broadcaster.sendTransform(transforms)
    
    def publish_static_transform(self, transform: TransformStamped):
        """
        Publish a static transform.
        
        Args:
            transform (TransformStamped): The static transform to publish
        """
        key = f"{transform.header.frame_id}->{transform.child_frame_id}"
        self.static_transforms[key] = transform
        self.static_broadcaster.sendTransform(transform)
    
    def publish_static_transforms(self, transforms: List[TransformStamped]):
        """
        Publish multiple static transforms at once.
        
        Args:
            transforms (List[TransformStamped]): The static transforms to publish
        """
        for transform in transforms:
            key = f"{transform.header.frame_id}->{transform.child_frame_id}"
            self.static_transforms[key] = transform
        self.static_broadcaster.sendTransform(transforms)
    
    def get_static_transform(self, parent_frame: str, child_frame: str) -> Optional[TransformStamped]:
        """
        Get a previously published static transform.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            
        Returns:
            Optional[TransformStamped]: The static transform, or None if not found
        """
        key = f"{parent_frame}->{child_frame}"
        return self.static_transforms.get(key)
    
    def update_static_transform(self, transform: TransformStamped):
        """
        Update a previously published static transform.
        
        Args:
            transform (TransformStamped): The updated static transform
        """
        self.publish_static_transform(transform)
    
    def publish_matrix_as_transform(self, matrix: np.ndarray, parent_frame: str, child_frame: str, 
                                    timestamp: Optional[Time] = None, is_static: bool = False):
        """
        Publish a transformation matrix as a TF transform.
        
        Args:
            matrix (np.ndarray): 4x4 homogeneous transformation matrix
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
            is_static (bool): Whether this is a static transform
        """
        transform = matrix_to_transform(matrix, parent_frame, child_frame, timestamp)
        
        if is_static:
            self.publish_static_transform(transform)
        else:
            self.publish_transform(transform)
    
    def publish_pose_as_transform(self, position: Tuple[float, float, float], 
                                  orientation: Union[Tuple[float, float, float, float], Tuple[float, float, float]],
                                  parent_frame: str, child_frame: str, timestamp: Optional[Time] = None,
                                  is_static: bool = False):
        """
        Publish a pose as a TF transform.
        
        Args:
            position (Tuple[float, float, float]): Position (x, y, z)
            orientation: Either quaternion (x, y, z, w) or euler angles (roll, pitch, yaw)
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
            is_static (bool): Whether this is a static transform
        """
        transform = TransformStamped()
        if timestamp is None:
            transform.header.stamp = self.node.get_clock().now().to_msg()
        else:
            transform.header.stamp = timestamp.to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        
        # Handle different orientation formats
        if len(orientation) == 4:
            # Quaternion
            transform.transform.rotation.x = float(orientation[0])
            transform.transform.rotation.y = float(orientation[1])
            transform.transform.rotation.z = float(orientation[2])
            transform.transform.rotation.w = float(orientation[3])
        elif len(orientation) == 3:
            # Euler angles (roll, pitch, yaw)
            from tf_transformations import quaternion_from_euler
            quat = quaternion_from_euler(orientation[0], orientation[1], orientation[2])
            transform.transform.rotation.x = float(quat[0])
            transform.transform.rotation.y = float(quat[1])
            transform.transform.rotation.z = float(quat[2])
            transform.transform.rotation.w = float(quat[3])
        else:
            raise ValueError(f"Invalid orientation format: {orientation}")
        
        if is_static:
            self.publish_static_transform(transform)
        else:
            self.publish_transform(transform)
            
    def publish_transform_identity(self, parent_frame: str, child_frame: str, 
                                timestamp: Optional[Time] = None, is_static: bool = False):
        """
        Publish an identity (no transformation) TF transform.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
            is_static (bool): Whether this is a static transform
        """
        transform = TransformStamped()
        if timestamp is None:
            transform.header.stamp = self.node.get_clock().now().to_msg()
        else:
            transform.header.stamp = timestamp.to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        if is_static:
            self.publish_static_transform(transform)
        else:
            self.publish_transform(transform)
            

class TFBroadcasterWrapper:
    """
    A simple wrapper around the TF2 broadcaster for easier transform publishing.
    """
    
    def __init__(self, node: Node):
        """
        Initialize the TF broadcaster wrapper.
        
        Args:
            node (Node): The ROS2 node to publish from
        """
        self.node = node
        self.broadcaster = TransformBroadcaster(node)
    
    def publish_transform(self, parent_frame: str, child_frame: str,
                         translation: Tuple[float, float, float],
                         rotation: Tuple[float, float, float, float],
                         timestamp: Optional[Time] = None):
        """
        Publish a TF transform.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            translation (Tuple[float, float, float]): Translation (x, y, z)
            rotation (Tuple[float, float, float, float]): Rotation quaternion (x, y, z, w)
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
        """
        transform = TransformStamped()
        if timestamp is None:
            transform.header.stamp = self.node.get_clock().now().to_msg()
        else:
            transform.header.stamp = timestamp.to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        
        transform.transform.rotation.x = float(rotation[0])
        transform.transform.rotation.y = float(rotation[1])
        transform.transform.rotation.z = float(rotation[2])
        transform.transform.rotation.w = float(rotation[3])
        
        self.broadcaster.sendTransform(transform)
    
    def publish_transform_with_euler(self, parent_frame: str, child_frame: str,
                                    translation: Tuple[float, float, float],
                                    euler: Tuple[float, float, float],
                                    timestamp: Optional[Time] = None):
        """
        Publish a TF transform with Euler angles.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame_id (str): Child frame ID
            translation (Tuple[float, float, float]): Translation (x, y, z)
            euler (Tuple[float, float, float]): Euler angles (roll, pitch, yaw)
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
        """
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(euler[0], euler[1], euler[2])
        self.publish_transform(parent_frame, child_frame, translation, quat, timestamp)
    
    def publish_transform_from_matrix(self, parent_frame: str, child_frame: str,
                                     matrix: np.ndarray, timestamp: Optional[Time] = None):
        """
        Publish a TF transform from a 4x4 homogeneous transformation matrix.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            matrix (np.ndarray): 4x4 homogeneous transformation matrix
            timestamp (Optional[Time]): Timestamp for the transform, defaults to current time
        """
        transform = matrix_to_transform(matrix, parent_frame, child_frame, timestamp)
        self.broadcaster.sendTransform(transform)


class StaticTFBroadcasterWrapper:
    """
    A simple wrapper around the TF2 static broadcaster for easier static transform publishing.
    """
    
    def __init__(self, node: Node):
        """
        Initialize the static TF broadcaster wrapper.
        
        Args:
            node (Node): The ROS2 node to publish from
        """
        self.node = node
        self.broadcaster = StaticTransformBroadcaster(node)
        self.static_transforms = {}
    
    def publish_static_transform(self, parent_frame: str, child_frame: str,
                                translation: Tuple[float, float, float],
                                rotation: Tuple[float, float, float, float]):
        """
        Publish a static TF transform.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            translation (Tuple[float, float, float]): Translation (x, y, z)
            rotation (Tuple[float, float, float, float]): Rotation quaternion (x, y, z, w)
        """
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        
        transform.transform.rotation.x = float(rotation[0])
        transform.transform.rotation.y = float(rotation[1])
        transform.transform.rotation.z = float(rotation[2])
        transform.transform.rotation.w = float(rotation[3])
        
        key = f"{parent_frame}->{child_frame}"
        self.static_transforms[key] = transform
        
        self.broadcaster.sendTransform([transform])
    
    def publish_static_transform_with_euler(self, parent_frame: str, child_frame: str,
                                          translation: Tuple[float, float, float],
                                          euler: Tuple[float, float, float]):
        """
        Publish a static TF transform with Euler angles.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            translation (Tuple[float, float, float]): Translation (x, y, z)
            euler (Tuple[float, float, float]): Euler angles (roll, pitch, yaw)
        """
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(euler[0], euler[1], euler[2])
        self.publish_static_transform(parent_frame, child_frame, translation, quat)
    
    def publish_static_transform_from_matrix(self, parent_frame: str, child_frame: str,
                                            matrix: np.ndarray):
        """
        Publish a static TF transform from a 4x4 homogeneous transformation matrix.
        
        Args:
            parent_frame (str): Parent frame ID
            child_frame (str): Child frame ID
            matrix (np.ndarray): 4x4 homogeneous transformation matrix
        """
        transform = matrix_to_transform(matrix, parent_frame, child_frame, 
                                        self.node.get_clock().now())
        
        key = f"{parent_frame}->{child_frame}"
        self.static_transforms[key] = transform
        
        self.broadcaster.sendTransform([transform])
    
    def publish_all_static_transforms(self):
        """
        Republish all static transforms.
        """
        if self.static_transforms:
            transforms = list(self.static_transforms.values())
            self.broadcaster.sendTransform(transforms) 
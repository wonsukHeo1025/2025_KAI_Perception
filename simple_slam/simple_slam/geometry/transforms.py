#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometric transformation utilities for the Simple SLAM package.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import tf_transformations

class TransformHelpers:
    """
    Helper class for transformations between coordinate frames.
    """
    
    @staticmethod
    def euler_to_rotation_matrix(roll, pitch, yaw):
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            roll (float): Roll angle in radians
            pitch (float): Pitch angle in radians
            yaw (float): Yaw angle in radians
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        r = Rotation.from_euler('xyz', [roll, pitch, yaw])
        return r.as_matrix()
    
    @staticmethod
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        """
        Convert quaternion to rotation matrix.
        
        Args:
            qx (float): x component of quaternion
            qy (float): y component of quaternion
            qz (float): z component of quaternion
            qw (float): w component of quaternion
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        r = Rotation.from_quat([qx, qy, qz, qw])
        return r.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix):
        """
        Convert rotation matrix to Euler angles.
        
        Args:
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix
            
        Returns:
            tuple: (roll, pitch, yaw) in radians
        """
        r = Rotation.from_matrix(rotation_matrix)
        return r.as_euler('xyz')
    
    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix):
        """
        Convert rotation matrix to quaternion.
        
        Args:
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix
            
        Returns:
            numpy.ndarray: quaternion as [x, y, z, w]
        """
        r = Rotation.from_matrix(rotation_matrix)
        return r.as_quat()
    
    @staticmethod
    def create_transform_matrix(translation, rotation):
        """
        Create a 4x4 transformation matrix from translation and rotation.
        
        Args:
            translation (numpy.ndarray): Translation vector [x, y, z]
            rotation: Rotation representation, one of:
                      - Rotation matrix (3x3 numpy.ndarray)
                      - Quaternion [x, y, z, w] (4-element numpy.ndarray)
                      - Euler angles [roll, pitch, yaw] (3-element numpy.ndarray)
                      
        Returns:
            numpy.ndarray: 4x4 transformation matrix
        """
        transform = np.eye(4)
        
        # Handle translation
        transform[:3, 3] = translation
        
        # Handle rotation
        if isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                # Rotation matrix
                transform[:3, :3] = rotation
            elif rotation.shape == (4,):
                # Quaternion
                transform[:3, :3] = Rotation.from_quat(rotation).as_matrix()
            elif rotation.shape == (3,):
                # Euler angles
                transform[:3, :3] = Rotation.from_euler('xyz', rotation).as_matrix()
            else:
                raise ValueError(f"Invalid rotation shape: {rotation.shape}")
        else:
            raise ValueError(f"Unsupported rotation type: {type(rotation)}")
        
        return transform
    
    @staticmethod
    def integrate_velocity(previous_pose, linear_velocity, angular_velocity, dt):
        """
        Integrate velocities to update pose using simple Euler integration.
        
        Args:
            previous_pose (numpy.ndarray): 4x4 transformation matrix representing previous pose
            linear_velocity (numpy.ndarray): Linear velocity [vx, vy, vz] in local frame
            angular_velocity (numpy.ndarray): Angular velocity [wx, wy, wz] in local frame
            dt (float): Time step
            
        Returns:
            numpy.ndarray: Updated 4x4 transformation matrix
        """
        # Extract rotation and translation from previous pose
        R_prev = previous_pose[:3, :3]
        t_prev = previous_pose[:3, 3]
        
        # Integrate angular velocity to get rotation change
        angle = np.linalg.norm(angular_velocity)
        if angle > 1e-6:  # Check if angular velocity is significant
            axis = angular_velocity / angle
            angle_change = angle * dt
            # Create rotation matrix for the change
            R_change = Rotation.from_rotvec(axis * angle_change).as_matrix()
            # Update rotation
            R_new = R_prev @ R_change
        else:
            R_new = R_prev
        
        # Integrate linear velocity in local frame to position change
        position_change = R_prev @ (linear_velocity * dt)
        t_new = t_prev + position_change
        
        # Create new pose
        new_pose = np.eye(4)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = t_new
        
        return new_pose
    
    @staticmethod
    def interpolate_transforms(transform1, transform2, alpha):
        """
        Interpolate between two transformation matrices.
        
        Args:
            transform1 (numpy.ndarray): First 4x4 transformation matrix
            transform2 (numpy.ndarray): Second 4x4 transformation matrix
            alpha (float): Interpolation factor (0 to 1)
            
        Returns:
            numpy.ndarray: Interpolated 4x4 transformation matrix
        """
        # Ensure alpha is between 0 and 1
        alpha = max(0.0, min(1.0, alpha))
        
        # Extract translations
        t1 = transform1[:3, 3]
        t2 = transform2[:3, 3]
        
        # Extract rotations
        R1 = transform1[:3, :3]
        R2 = transform2[:3, :3]
        
        # Convert to quaternions
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        
        # Interpolate translation (linear)
        t_interp = t1 + alpha * (t2 - t1)
        
        # Interpolate rotation (spherical)
        from scipy.spatial.transform import Slerp
        from scipy.spatial.transform import RotationSpline
        
        times = np.array([0, 1])
        rotations = Rotation.from_quat(np.vstack([q1, q2]))
        slerp = Slerp(times, rotations)
        
        r_interp = slerp([alpha])[0]
        R_interp = r_interp.as_matrix()
        
        # Create interpolated transform
        transform_interp = np.eye(4)
        transform_interp[:3, :3] = R_interp
        transform_interp[:3, 3] = t_interp
        
        return transform_interp


class Pose:
    """
    A class for handling poses in 3D space.
    
    A pose consists of a position and an orientation, and can be represented
    as a 4x4 homogeneous transformation matrix.
    """
    
    def __init__(self, position=None, orientation=None):
        """
        Initialize a pose.
        
        Args:
            position (Optional[numpy.ndarray]): Position vector [x, y, z]
            orientation (Optional): Orientation, can be:
                                    - Quaternion [x, y, z, w] (numpy.ndarray)
                                    - Euler angles [roll, pitch, yaw] (numpy.ndarray)
                                    - Rotation matrix (3x3 numpy.ndarray)
        """
        if position is None:
            position = np.zeros(3)
        
        if orientation is None:
            # Default is identity rotation (quaternion [0, 0, 0, 1])
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        self.matrix = TransformHelpers.create_transform_matrix(position, orientation)
    
    @classmethod
    def from_matrix(cls, matrix):
        """
        Create a Pose from a 4x4 homogeneous transformation matrix.
        
        Args:
            matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
            
        Returns:
            Pose: A new Pose object
        """
        pose = cls()
        pose.matrix = matrix.copy()
        return pose
    
    @property
    def position(self):
        """Get the position vector [x, y, z]."""
        return self.matrix[:3, 3]
    
    @position.setter
    def position(self, value):
        """Set the position vector [x, y, z]."""
        self.matrix[:3, 3] = value
    
    @property
    def orientation_quaternion(self):
        """Get the orientation as a quaternion [x, y, z, w]."""
        return TransformHelpers.rotation_matrix_to_quaternion(self.matrix[:3, :3])
    
    @orientation_quaternion.setter
    def orientation_quaternion(self, value):
        """Set the orientation from a quaternion [x, y, z, w]."""
        self.matrix[:3, :3] = TransformHelpers.quaternion_to_rotation_matrix(*value)
    
    @property
    def orientation_euler(self):
        """Get the orientation as Euler angles [roll, pitch, yaw]."""
        return TransformHelpers.rotation_matrix_to_euler(self.matrix[:3, :3])
    
    @orientation_euler.setter
    def orientation_euler(self, value):
        """Set the orientation from Euler angles [roll, pitch, yaw]."""
        self.matrix[:3, :3] = TransformHelpers.euler_to_rotation_matrix(*value)
    
    @property
    def rotation_matrix(self):
        """Get the rotation matrix (3x3)."""
        return self.matrix[:3, :3]
    
    @rotation_matrix.setter
    def rotation_matrix(self, value):
        """Set the rotation matrix (3x3)."""
        self.matrix[:3, :3] = value
    
    def inverse(self):
        """
        Compute the inverse of this pose.
        
        Returns:
            Pose: The inverse pose
        """
        from simple_slam.utils.common import invert_transform_matrix
        inv_matrix = invert_transform_matrix(self.matrix)
        return Pose.from_matrix(inv_matrix)
    
    def transform_point(self, point):
        """
        Transform a point from the local frame to the global frame.
        
        Args:
            point (numpy.ndarray): Point in local frame [x, y, z]
            
        Returns:
            numpy.ndarray: Point in global frame [x, y, z]
        """
        # Convert to homogeneous coordinates
        if len(point) == 3:
            point_h = np.append(point, 1.0)
        else:
            point_h = point.copy()
            point_h[3] = 1.0
        
        # Transform and return
        transformed_h = self.matrix @ point_h
        return transformed_h[:3] / transformed_h[3]  # Perspective division if needed
    
    def compose(self, other):
        """
        Compose this pose with another pose (this * other).
        
        Args:
            other (Pose): The other pose
            
        Returns:
            Pose: The composed pose
        """
        return Pose.from_matrix(self.matrix @ other.matrix)
    
    def __mul__(self, other):
        """
        Overload the multiplication operator for pose composition.
        
        Args:
            other (Pose): The other pose
            
        Returns:
            Pose: The composed pose
        """
        return self.compose(other)
    
    def to_transform_stamped(self, frame_id, child_frame_id, stamp=None):
        """
        Convert to a geometry_msgs.msg.TransformStamped message.
        
        Args:
            frame_id (str): Parent frame ID
            child_frame_id (str): Child frame ID
            stamp (Optional[Time]): Timestamp for the transform
            
        Returns:
            geometry_msgs.msg.TransformStamped: The transform message
        """
        from simple_slam.utils.common import matrix_to_transform
        return matrix_to_transform(self.matrix, frame_id, child_frame_id, stamp)
    
    @classmethod
    def from_transform_stamped(cls, transform):
        """
        Create a Pose from a geometry_msgs.msg.TransformStamped message.
        
        Args:
            transform (geometry_msgs.msg.TransformStamped): The transform message
            
        Returns:
            Pose: A new Pose object
        """
        from simple_slam.utils.common import transform_to_matrix
        matrix = transform_to_matrix(transform)
        return cls.from_matrix(matrix)
    
    def __str__(self):
        """String representation of the pose."""
        pos = self.position
        euler = self.orientation_euler
        return f"Pose(position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], " \
               f"orientation=[{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}])" 
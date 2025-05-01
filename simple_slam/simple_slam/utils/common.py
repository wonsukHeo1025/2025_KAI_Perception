#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common utility functions for the Simple SLAM package.
"""

import numpy as np
from geometry_msgs.msg import TransformStamped, Quaternion, Point, Vector3


def homogeneous_to_position_quaternion(transform_matrix):
    """
    Convert a 4x4 homogeneous transformation matrix to position and quaternion.
    
    Args:
        transform_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
        
    Returns:
        tuple: (position, quaternion)
            - position (numpy.ndarray): [x, y, z]
            - quaternion (numpy.ndarray): [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation
    
    # Extract rotation matrix (3x3) and position vector (3x1)
    rotation_matrix = transform_matrix[:3, :3]
    position = transform_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()  # [x, y, z, w]
    
    return position, quaternion


def transform_to_matrix(transform):
    """
    Convert a geometry_msgs.msg.TransformStamped message to a 4x4 homogeneous transformation matrix.
    
    Args:
        transform (TransformStamped): The transform message
        
    Returns:
        numpy.ndarray: 4x4 homogeneous transformation matrix
    """
    from tf_transformations import quaternion_matrix
    
    # Extract translation and rotation
    translation = np.array([
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    ])
    
    rotation = np.array([
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w
    ])
    
    # Create transformation matrix
    transformation_matrix = quaternion_matrix(rotation)
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix


def matrix_to_transform(matrix, frame_id, child_frame_id, stamp=None):
    """
    Convert a 4x4 homogeneous transformation matrix to a geometry_msgs.msg.TransformStamped message.
    
    Args:
        matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
        frame_id (str): Parent frame ID
        child_frame_id (str): Child frame ID
        stamp (Optional[Time]): Timestamp for the transform
        
    Returns:
        TransformStamped: The transform message
    """
    from tf_transformations import quaternion_from_matrix
    
    transform = TransformStamped()
    if stamp:
        transform.header.stamp = stamp
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    
    # Extract translation from the matrix
    transform.transform.translation.x = float(matrix[0, 3])
    transform.transform.translation.y = float(matrix[1, 3])
    transform.transform.translation.z = float(matrix[2, 3])
    
    # Extract rotation (quaternion) from the matrix
    q = quaternion_from_matrix(matrix)
    transform.transform.rotation.x = float(q[0])
    transform.transform.rotation.y = float(q[1])
    transform.transform.rotation.z = float(q[2])
    transform.transform.rotation.w = float(q[3])
    
    return transform


def invert_transform_matrix(transform_matrix):
    """
    Invert a 4x4 homogeneous transformation matrix.
    
    Args:
        transform_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
        
    Returns:
        numpy.ndarray: Inverted 4x4 homogeneous transformation matrix
    """
    # Extract rotation matrix and translation vector
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]
    
    # Calculate inverse
    R_inv = R.T
    t_inv = -R_inv @ t
    
    # Create inverse transformation matrix
    transform_inv = np.eye(4)
    transform_inv[:3, :3] = R_inv
    transform_inv[:3, 3] = t_inv
    
    return transform_inv


def clip_value(value, min_val, max_val):
    """
    Clip a value between min_val and max_val.
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clipped value
    """
    return max(min_val, min(value, max_val))


def quaternion_msg_to_array(q_msg):
    """
    Convert a geometry_msgs.msg.Quaternion message to a numpy array.
    
    Args:
        q_msg (Quaternion): Quaternion message
        
    Returns:
        numpy.ndarray: Quaternion as [x, y, z, w]
    """
    return np.array([q_msg.x, q_msg.y, q_msg.z, q_msg.w])


def point_msg_to_array(p_msg):
    """
    Convert a geometry_msgs.msg.Point message to a numpy array.
    
    Args:
        p_msg (Point): Point message
        
    Returns:
        numpy.ndarray: Point as [x, y, z]
    """
    return np.array([p_msg.x, p_msg.y, p_msg.z])


def vector3_msg_to_array(v_msg):
    """
    Convert a geometry_msgs.msg.Vector3 message to a numpy array.
    
    Args:
        v_msg (Vector3): Vector3 message
        
    Returns:
        numpy.ndarray: Vector as [x, y, z]
    """
    return np.array([v_msg.x, v_msg.y, v_msg.z])


def array_to_point_msg(array):
    """
    Convert a numpy array to a geometry_msgs.msg.Point message.
    
    Args:
        array (numpy.ndarray): Array with shape (3,) representing [x, y, z]
        
    Returns:
        Point: Point message
    """
    p = Point()
    p.x = float(array[0])
    p.y = float(array[1])
    p.z = float(array[2])
    return p


def array_to_quaternion_msg(array):
    """
    Convert a numpy array to a geometry_msgs.msg.Quaternion message.
    
    Args:
        array (numpy.ndarray): Array with shape (4,) representing [x, y, z, w]
        
    Returns:
        Quaternion: Quaternion message
    """
    q = Quaternion()
    q.x = float(array[0])
    q.y = float(array[1])
    q.z = float(array[2])
    q.w = float(array[3])
    return q


def array_to_vector3_msg(array):
    """
    Convert a numpy array to a geometry_msgs.msg.Vector3 message.
    
    Args:
        array (numpy.ndarray): Array with shape (3,) representing [x, y, z]
        
    Returns:
        Vector3: Vector3 message
    """
    v = Vector3()
    v.x = float(array[0])
    v.y = float(array[1])
    v.z = float(array[2])
    return v 
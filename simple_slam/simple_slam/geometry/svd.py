#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVD-based pose estimation for point cloud alignment.
"""

import numpy as np
from scipy.spatial.transform import Rotation

class SVDPoseEstimator:
    """
    A class to estimate the pose between two point sets using SVD.
    
    This implementation is based on the Arun's method (Least-Squares Fitting of Two 3-D Point Sets).
    """
    
    def __init__(self, min_points=3, stability_threshold=0.1, condition_number_threshold=100.0):
        """
        Initialize the SVD pose estimator.
        
        Args:
            min_points (int): Minimum number of point correspondences required for SVD.
            stability_threshold (float): Threshold for stability check (variance ratio).
            condition_number_threshold (float): Maximum allowed condition number for stability.
        """
        self.min_points = min_points
        self.stability_threshold = stability_threshold
        self.condition_number_threshold = condition_number_threshold
    
    def estimate_pose(self, src_points, dst_points):
        """
        Estimate the pose from source points to destination points using SVD.
        
        The transformation returned maps points from source frame to destination frame.
        
        Args:
            src_points (numpy.ndarray): Source points, shape (N, 3) or (3, N)
            dst_points (numpy.ndarray): Destination points, shape (N, 3) or (3, N)
            
        Returns:
            tuple: (transform_matrix, quality_metrics)
                - transform_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
                - quality_metrics (dict): Dictionary with quality metrics
        """
        # Ensure points are shape (3, N)
        if src_points.shape[0] == 3 and src_points.shape[1] >= 3:
            src_pts = src_points
            dst_pts = dst_points
        elif src_points.shape[1] == 3 and src_points.shape[0] >= 3:
            src_pts = src_points.T
            dst_pts = dst_points.T
        else:
            raise ValueError(f"Invalid point shapes: src={src_points.shape}, dst={dst_points.shape}")
        
        # Check minimum number of points
        n_points = src_pts.shape[1]
        if n_points < self.min_points:
            return np.eye(4), {"success": False, "reason": f"Not enough points: {n_points} < {self.min_points}"}
        
        try:
            # Compute centroids
            src_centroid = np.mean(src_pts, axis=1, keepdims=True)
            dst_centroid = np.mean(dst_pts, axis=1, keepdims=True)
            
            # Center the points
            src_centered = src_pts - src_centroid
            dst_centered = dst_pts - dst_centroid
            
            # Compute covariance matrix H = dst_centered @ src_centered.T
            H = dst_centered @ src_centered.T
            
            # Perform SVD
            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            
            # Check stability using singular values
            quality_metrics = self.check_stability(src_centered, S)
            
            # Compute rotation matrix
            # R = U @ Vt 
            # Check for reflection case
            det_R = np.linalg.det(U @ Vt)
            if det_R < 0:
                # Handle reflection case (improper rotation)
                V_adj = V.copy()
                V_adj[:, -1] *= -1  # Flip the last column of V
                R = U @ V_adj.T
            else:
                R = U @ Vt
            
            # Compute translation
            t = dst_centroid - R @ src_centroid
            
            # Assemble transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t.flatten()
            
            return transform, quality_metrics
            
        except np.linalg.LinAlgError as e:
            return np.eye(4), {"success": False, "reason": f"SVD computation failed: {e}"}
        except Exception as e:
            return np.eye(4), {"success": False, "reason": f"Unexpected error: {e}"}
    
    def check_stability(self, centered_points, singular_values):
        """
        Check the stability of the SVD result based on various metrics.
        
        Args:
            centered_points (numpy.ndarray): Centered points used for SVD
            singular_values (numpy.ndarray): Singular values from SVD
            
        Returns:
            dict: Dictionary with quality metrics
        """
        metrics = {"success": True}
        
        # Compute condition number (ratio of largest to smallest singular value)
        if len(singular_values) >= 3:
            condition_number = singular_values[0] / max(singular_values[2], 1e-10)
            metrics["condition_number"] = condition_number
            
            if condition_number > self.condition_number_threshold:
                metrics["success"] = False
                metrics["reason"] = f"High condition number: {condition_number:.1f} > {self.condition_number_threshold:.1f}"
                return metrics
        
        # Check point distribution
        # Compute covariance of centered points
        cov = centered_points @ centered_points.T
        
        # Compute eigenvalues of covariance matrix to check point distribution
        try:
            eig_vals = np.linalg.eigvalsh(cov)
            eig_vals = np.abs(eig_vals)  # Ensure positive (numerical issues)
            metrics["eigenvalues"] = eig_vals
            
            # Sort in descending order
            eig_vals = np.sort(eig_vals)[::-1]
            
            # Check if points are colinear (smallest eigenvalue close to zero)
            if eig_vals[2] / max(eig_vals[0], 1e-10) < self.stability_threshold:
                metrics["success"] = False
                metrics["reason"] = f"Points are near-colinear: {eig_vals[2]/eig_vals[0]:.5f} < {self.stability_threshold:.5f}"
                return metrics
            
            # Check if points are coplanar (significant drop from 2nd to 3rd eigenvalue)
            if len(eig_vals) >= 3 and eig_vals[2] / max(eig_vals[1], 1e-10) < self.stability_threshold:
                metrics["success"] = False
                metrics["reason"] = f"Points are near-coplanar: {eig_vals[2]/eig_vals[1]:.5f} < {self.stability_threshold:.5f}"
                return metrics
            
        except np.linalg.LinAlgError:
            metrics["success"] = False
            metrics["reason"] = "Failed to compute eigenvalues for stability check"
            return metrics
        
        # Check point spread in each dimension
        spread = np.max(centered_points, axis=1) - np.min(centered_points, axis=1)
        metrics["point_spread"] = spread
        
        # Check if spread is too small in any dimension
        min_spread = 0.05  # 5cm
        if np.any(spread < min_spread):
            metrics["success"] = False
            metrics["reason"] = f"Insufficient point spread in some dimension: {spread} < {min_spread}"
            return metrics
        
        return metrics


def estimate_pose_svd(src_points, dst_points, min_points=3, stability_threshold=0.1):
    """
    Wrapper function to estimate pose using SVD without creating an instance.
    
    Args:
        src_points (numpy.ndarray): Source points, shape (N, 3) or (3, N)
        dst_points (numpy.ndarray): Destination points, shape (N, 3) or (3, N)
        min_points (int): Minimum number of point correspondences required
        stability_threshold (float): Threshold for stability check
        
    Returns:
        tuple: (transform_matrix, quality_ok)
            - transform_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix
            - quality_ok (bool): Whether the pose estimate is reliable
    """
    estimator = SVDPoseEstimator(min_points, stability_threshold)
    transform, metrics = estimator.estimate_pose(src_points, dst_points)
    return transform, metrics["success"] 
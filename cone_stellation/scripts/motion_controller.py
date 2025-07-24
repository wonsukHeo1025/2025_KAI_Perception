#!/usr/bin/env python3
"""
Motion Controller Module for CC-SLAM-SYM

Handles vehicle motion simulation along predefined paths.
Supports different scenarios like straight tracks and Formula Student circuits.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.interpolate import splprep, splev, interp1d


class MotionScenario(Enum):
    """Available motion scenarios"""
    STRAIGHT_TRACK = 1  # Straight line with AEB test
    FORMULA_STUDENT = 2  # Elliptical Formula Student track


@dataclass
class VehicleState:
    """Complete vehicle state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw]
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    linear_acceleration: np.ndarray  # [ax, ay, az]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for sensor simulator"""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'linear_velocity': self.linear_velocity.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'linear_acceleration': self.linear_acceleration.tolist()
        }


class TrajectoryGenerator:
    """Generates smooth trajectories for different scenarios"""
    
    @staticmethod
    def generate_straight_track_centerline(length: float = 150.0, 
                                         num_points: int = 300) -> List[Tuple[float, float]]:
        """Generate centerline for straight track scenario"""
        points = []
        for i in range(num_points):
            x = i * length / (num_points - 1)
            y = 0.0
            points.append((x, y))
        return points
    
    @staticmethod
    def generate_formula_student_centerline(num_points: int = 400) -> List[Tuple[float, float]]:
        """Generate centerline for Formula Student elliptical track using spline interpolation"""
        # Use proven waypoints from successful implementation
        base_waypoints = [
            (35.0, 12.5), (88.0, 12.5),
            (88.50, 12.00), (90.10, 6.72), (94.36, 3.22), (99.85, 2.68), (104.72, 5.28), 
            (107.32, 10.15), (106.78, 15.64), (103.28, 19.90), (98.00, 21.50),
            (95.0, 22.0), (80.0, 27.5),
            (75.0, 27.5), (70.0, 27.5), (65.0, 27.5), (60.0, 27.5), (55.0, 27.5),
            (50.0, 27.5), (45.0, 27.5), (40.0, 27.5), (35.0, 27.5), (30.0, 27.5),
            (27.6, 26.6),  # Removed duplicate (12.0, 21.5)
            (6.72, 19.90), (3.22, 15.64), 
            (2.68, 10.15), (5.28, 5.28), (10.15, 2.68), (15.64, 3.22), (19.90, 6.72), (21.50, 12.00),
            (23.0, 12.5), (30.0, 12.5),
        ]
        
        # Remove any duplicate waypoints
        unique_waypoints = []
        for wp in base_waypoints:
            if not unique_waypoints or (wp != unique_waypoints[-1]):
                unique_waypoints.append(wp)
        
        # Convert to numpy arrays for spline fitting
        base_waypoints_array = np.array(unique_waypoints)
        x = base_waypoints_array[:, 0]
        y = base_waypoints_array[:, 1]
        
        # Create periodic spline (closed track)
        # s: smoothing parameter (0 = interpolating spline)
        # k: degree of spline (3 = cubic)
        # per: periodic spline for closed track
        # Simple approach: linear interpolation with corner smoothing
        points = []
        
        # First pass: linear interpolation between waypoints
        for i in range(len(unique_waypoints)):
            current = unique_waypoints[i]
            next_wp = unique_waypoints[(i + 1) % len(unique_waypoints)]
            
            # Calculate segment length
            dx = next_wp[0] - current[0]
            dy = next_wp[1] - current[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Add interpolated points every 0.3 meters for higher resolution
            num_interp = max(1, int(distance / 0.3))
            for j in range(num_interp):
                t = j / float(num_interp)
                x = current[0] + t * dx
                y = current[1] + t * dy
                points.append((x, y))
        
        # Second pass: smooth sharp corners
        # Smooth only at waypoint locations for controlled smoothing
        corner_radius = 3.0  # meters - increased for smoother curves
        smoothed_points = []
        
        for i in range(len(points)):
            prev_idx = (i - 10) % len(points)  # Increased window for smoother transitions
            next_idx = (i + 10) % len(points)
            
            # Check if we're near a waypoint (sharp corner)
            near_waypoint = False
            for wp in unique_waypoints:
                dist = np.sqrt((points[i][0] - wp[0])**2 + (points[i][1] - wp[1])**2)
                if dist < corner_radius:
                    near_waypoint = True
                    break
            
            if near_waypoint and i > 10 and i < len(points) - 10:
                # Apply smoothing using weighted average
                prev_point = points[prev_idx]
                curr_point = points[i]
                next_point = points[next_idx]
                
                # Weighted average (reduced current point weight for smoother curves)
                weight_curr = 0.4  # Reduced from 0.5
                weight_neighbors = 0.3  # Increased from 0.25
                
                smooth_x = (weight_neighbors * prev_point[0] + 
                           weight_curr * curr_point[0] + 
                           weight_neighbors * next_point[0])
                smooth_y = (weight_neighbors * prev_point[1] + 
                           weight_curr * curr_point[1] + 
                           weight_neighbors * next_point[1])
                
                smoothed_points.append((smooth_x, smooth_y))
            else:
                smoothed_points.append(points[i])
        
        # Ensure we have approximately the requested number of points
        if len(smoothed_points) != num_points:
            # Resample to get exact number of points
            indices = np.linspace(0, len(smoothed_points)-1, num_points, dtype=int)
            smoothed_points = [smoothed_points[i] for i in indices]
        
        return smoothed_points
    
    @staticmethod
    def smooth_centerline(points: List[Tuple[float, float]], 
                         smoothing_factor: float = 0.1) -> List[Tuple[float, float]]:
        """Apply smoothing to centerline points using spline interpolation"""
        if len(points) < 4:  # Need at least 4 points for cubic spline
            return points
        
        # Convert to numpy arrays
        points_array = np.array(points)
        x = points_array[:, 0]
        y = points_array[:, 1]
        
        # Check if the path is closed (first and last points are close)
        is_closed = np.linalg.norm(points_array[0] - points_array[-1]) < 1.0
        
        if is_closed:
            # Use periodic spline for closed tracks
            tck, u = splprep([x, y], s=smoothing_factor * len(points), per=True)
        else:
            # Use regular spline for open tracks
            tck, u = splprep([x, y], s=smoothing_factor * len(points), per=False)
        
        # Generate same number of points
        u_new = np.linspace(0, 1, len(points))
        x_smooth, y_smooth = splev(u_new, tck)
        
        # Convert back to list of tuples
        smoothed = [(x_smooth[i], y_smooth[i]) for i in range(len(points))]
        
        return smoothed


class ContinuousTrajectory:
    """Continuous spline-based trajectory for smooth motion"""
    
    def __init__(self, waypoints: List[Tuple[float, float]], is_closed: bool = True):
        """
        Initialize continuous trajectory from waypoints
        
        Args:
            waypoints: List of (x, y) waypoint coordinates
            is_closed: Whether the trajectory is closed (forms a loop)
        """
        # Convert to numpy arrays
        waypoints_array = np.array(waypoints)
        x = waypoints_array[:, 0]
        y = waypoints_array[:, 1]
        
        # Create spline representation
        smoothing_factor = 0.02 if is_closed else 0.05
        self.tck, self.u = splprep([x, y], s=smoothing_factor * len(waypoints), per=is_closed)
        self.is_closed = is_closed
        
        # Create distance mapping for arc-length parameterization
        self._create_distance_mapping()
    
    def _create_distance_mapping(self):
        """Create mapping from distance to spline parameter u"""
        # Sample the spline at high resolution
        u_samples = np.linspace(0, 1, 2000)
        positions = splev(u_samples, self.tck)
        
        # Calculate cumulative distances
        distances = [0.0]
        for i in range(1, len(positions[0])):
            dx = positions[0][i] - positions[0][i-1]
            dy = positions[1][i] - positions[1][i-1]
            distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))
        
        # Create interpolation function: distance → u_param
        self.distance_to_u = interp1d(distances, u_samples, 
                                     kind='linear', bounds_error=False, 
                                     fill_value=(0, 1))
        self.total_distance = distances[-1]
    
    def get_position_and_heading(self, distance: float) -> Tuple[np.ndarray, float]:
        """
        Get position and heading at given distance along trajectory
        
        Args:
            distance: Distance along trajectory
            
        Returns:
            Tuple of (position, heading) where position is [x, y] and heading is in radians
        """
        # Handle closed trajectories by wrapping distance
        if self.is_closed:
            distance = distance % self.total_distance
        else:
            distance = np.clip(distance, 0, self.total_distance)
        
        # Convert distance to spline parameter
        u_param = self.distance_to_u(distance)
        
        # Get position (0th derivative)
        pos = splev(u_param, self.tck, der=0)
        position = np.array([pos[0], pos[1]])
        
        # Get tangent vector (1st derivative) for heading
        tangent = splev(u_param, self.tck, der=1)
        heading = np.arctan2(tangent[1], tangent[0])
        
        return position, heading
    
    def get_curvature(self, distance: float) -> float:
        """Get curvature at given distance for speed control"""
        if self.is_closed:
            distance = distance % self.total_distance
        else:
            distance = np.clip(distance, 0, self.total_distance)
        
        u_param = self.distance_to_u(distance)
        
        # Get first and second derivatives
        first_deriv = splev(u_param, self.tck, der=1)
        second_deriv = splev(u_param, self.tck, der=2)
        
        # Calculate curvature: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        dx, dy = first_deriv[0], first_deriv[1]
        ddx, ddy = second_deriv[0], second_deriv[1]
        
        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        
        if denominator < 1e-8:
            return 0.0
        
        return numerator / denominator


class MotionController:
    """Controls vehicle motion along trajectories"""
    
    def __init__(self, scenario: MotionScenario, base_speed: float = 5.0):
        """
        Initialize motion controller
        
        Args:
            scenario: Motion scenario to use
            base_speed: Base vehicle speed in m/s
        """
        self.scenario = scenario
        self.base_speed = base_speed
        
        # Create continuous trajectory based on scenario
        if scenario == MotionScenario.STRAIGHT_TRACK:
            waypoints = TrajectoryGenerator.generate_straight_track_centerline()
            self.trajectory = ContinuousTrajectory(waypoints, is_closed=False)
        else:
            waypoints = TrajectoryGenerator.generate_formula_student_centerline()
            self.trajectory = ContinuousTrajectory(waypoints, is_closed=True)
        
        # Initialize distance tracking
        self.current_distance = 0.0
        
        # Initialize vehicle state based on scenario
        if scenario == MotionScenario.STRAIGHT_TRACK:
            # Scenario 1: Start at beginning of track
            initial_pos = np.array([0.0, 0.0, 0.0])
            self.current_distance = 0.0
        else:
            # Scenario 2: Start at specific position (find closest distance)
            target_pos = np.array([30.0, 12.5])
            # Find closest point on trajectory
            min_dist = float('inf')
            best_distance = 0.0
            
            # Sample along trajectory to find closest point
            for dist in np.linspace(0, self.trajectory.total_distance, 1000):
                pos, _ = self.trajectory.get_position_and_heading(dist)
                distance_to_target = np.linalg.norm(pos - target_pos)
                if distance_to_target < min_dist:
                    min_dist = distance_to_target
                    best_distance = dist
            
            self.current_distance = best_distance
            initial_pos = np.array([30.0, 12.5, 0.0])  # Keep original for consistency
        
        self.state = VehicleState(
            position=initial_pos,
            orientation=np.array([0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            linear_acceleration=np.array([0.0, 0.0, 0.0])
        )
        
        # Previous values for derivative calculations
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
    
    def get_centerline_points(self) -> List[Tuple[float, float]]:
        """Get centerline points for visualization"""
        # Sample the continuous trajectory for visualization
        num_points = 400
        points = []
        for i in range(num_points):
            distance = i * self.trajectory.total_distance / (num_points - 1)
            pos, _ = self.trajectory.get_position_and_heading(distance)
            points.append((pos[0], pos[1]))
        return points
    
    def get_current_speed(self, elapsed_time: float) -> float:
        """
        Get current speed based on scenario and time
        
        Args:
            elapsed_time: Time since start in seconds
        
        Returns:
            Current speed in m/s
        """
        if self.scenario == MotionScenario.STRAIGHT_TRACK:
            # Accelerate for first 5 seconds, then constant
            if elapsed_time < 5.0:
                return min(self.base_speed, elapsed_time * self.base_speed / 5.0)
            else:
                return self.base_speed
        else:
            # Formula Student: Vary speed based on track curvature
            # Get curvature at current position
            curvature = self.trajectory.get_curvature(self.current_distance)
            
            # Reduce speed in high curvature areas
            if curvature > 0.1:  # High curvature threshold
                speed_factor = max(0.5, 1.0 - curvature / 0.5)
                return self.base_speed * speed_factor
            
            return self.base_speed
    
    def update_motion(self, dt: float, elapsed_time: float) -> VehicleState:
        """
        Update vehicle motion for one time step using continuous trajectory
        
        Args:
            dt: Time step in seconds
            elapsed_time: Total elapsed time since start
        
        Returns:
            Updated vehicle state
        """
        # Get current speed
        speed = self.get_current_speed(elapsed_time)
        
        # Update distance along trajectory
        distance_to_move = speed * dt
        self.current_distance += distance_to_move
        
        # Handle trajectory wrapping for closed tracks
        if self.scenario == MotionScenario.FORMULA_STUDENT:
            self.current_distance = self.current_distance % self.trajectory.total_distance
        else:
            # For open tracks, clamp to trajectory bounds
            self.current_distance = min(self.current_distance, self.trajectory.total_distance)
        
        # Get smooth position and heading from continuous trajectory
        position, heading = self.trajectory.get_position_and_heading(self.current_distance)
        
        # Update vehicle state
        self.state.position[0] = position[0]
        self.state.position[1] = position[1]
        self.state.orientation[2] = heading
        
        # Calculate velocities from heading (mathematically smooth)
        self.state.linear_velocity[0] = speed * np.cos(heading)
        self.state.linear_velocity[1] = speed * np.sin(heading)
        
        # Calculate angular velocity from heading derivative
        if dt > 0:
            # Get heading slightly ahead for derivative calculation
            look_ahead_distance = min(0.1, self.trajectory.total_distance * 0.001)  # Small look-ahead
            future_distance = self.current_distance + look_ahead_distance
            
            if self.scenario == MotionScenario.FORMULA_STUDENT:
                future_distance = future_distance % self.trajectory.total_distance
            else:
                future_distance = min(future_distance, self.trajectory.total_distance)
            
            _, future_heading = self.trajectory.get_position_and_heading(future_distance)
            
            # Calculate angular velocity from heading change
            heading_diff = self._normalize_angle(future_heading - heading)
            self.state.angular_velocity[2] = heading_diff / (look_ahead_distance / speed) if speed > 0 else 0.0
        
        # Calculate acceleration
        new_velocity = self.state.linear_velocity.copy()
        if dt > 0:
            self.state.linear_acceleration = (new_velocity - self.prev_velocity) / dt
        self.prev_velocity = new_velocity.copy()
        
        return self.state
    
    def get_look_ahead_point(self, look_ahead_distance: float) -> Optional[Tuple[float, float]]:
        """
        Get a point ahead on the trajectory for trajectory following
        
        Args:
            look_ahead_distance: Distance to look ahead in meters
        
        Returns:
            Look-ahead point or None if end of track
        """
        # Calculate future distance along trajectory
        future_distance = self.current_distance + look_ahead_distance
        
        # Handle trajectory bounds based on scenario
        if self.scenario == MotionScenario.FORMULA_STUDENT:
            # For closed tracks, wrap around
            future_distance = future_distance % self.trajectory.total_distance
        else:
            # For open tracks, check if we're beyond the end
            if future_distance > self.trajectory.total_distance:
                return None
        
        # Get position at future distance
        position, _ = self.trajectory.get_position_and_heading(future_distance)
        return (position[0], position[1])
    
    def reset(self) -> None:
        """Reset motion controller to start position"""
        # Reset distance tracking
        self.current_distance = 0.0
        
        # Reset to initial position based on scenario
        if self.scenario == MotionScenario.STRAIGHT_TRACK:
            initial_pos = np.array([0.0, 0.0, 0.0])
            self.current_distance = 0.0
        else:
            # For Formula Student, start at specific position
            target_pos = np.array([30.0, 12.5])
            # Find closest point on trajectory
            min_dist = float('inf')
            best_distance = 0.0
            
            for dist in np.linspace(0, self.trajectory.total_distance, 1000):
                pos, _ = self.trajectory.get_position_and_heading(dist)
                distance_to_target = np.linalg.norm(pos - target_pos)
                if distance_to_target < min_dist:
                    min_dist = distance_to_target
                    best_distance = dist
            
            self.current_distance = best_distance
            initial_pos = np.array([30.0, 12.5, 0.0])
        
        self.state = VehicleState(
            position=initial_pos,
            orientation=np.array([0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            linear_acceleration=np.array([0.0, 0.0, 0.0])
        )
        
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
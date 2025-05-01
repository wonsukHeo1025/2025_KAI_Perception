#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the Simple SLAM package.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Quaternion, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class MarkerPublisher:
    """
    A class to help publish visualization markers for RViz.
    """
    
    def __init__(self, node: Node, topic: str = '/visualization_markers', queue_size: int = 10):
        """
        Initialize the MarkerPublisher.
        
        Args:
            node (Node): The ROS2 node to publish from
            topic (str): The topic to publish markers to
            queue_size (int): Publisher queue size
        """
        self.node = node
        self.publisher = node.create_publisher(MarkerArray, topic, queue_size)
        self.marker_id = 0
        self.markers = {}
        self.default_frame_id = 'map'
        self.default_namespace = 'visualization'
        self.default_lifetime = 0.0  # 0 = persistent
    
    def reset(self):
        """
        Reset the marker publisher, clearing all current markers.
        """
        self.marker_id = 0
        self.markers = {}
    
    def delete_all_markers(self):
        """
        Send a message to delete all markers.
        """
        marker = Marker()
        marker.action = Marker.DELETEALL
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)
    
    def delete_marker(self, marker_id: int, namespace: str = None):
        """
        Delete a specific marker.
        
        Args:
            marker_id (int): ID of the marker to delete
            namespace (str): Namespace of the marker to delete
        """
        if namespace is None:
            namespace = self.default_namespace
        
        marker = Marker()
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)
        
        key = f"{namespace}_{marker_id}"
        if key in self.markers:
            del self.markers[key]
    
    def create_marker(self, marker_type: int, frame_id: str = None, namespace: str = None,
                      id: int = None, pose: Pose = None, scale: Vector3 = None,
                      color: ColorRGBA = None, lifetime: float = None,
                      points: List[Point] = None, colors: List[ColorRGBA] = None,
                      text: str = None, mesh_resource: str = None, mesh_use_embedded_materials: bool = False):
        """
        Create a marker with the specified properties.
        
        Args:
            marker_type (int): Type of the marker (CUBE, SPHERE, POINTS, etc.)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker (assigned automatically if None)
            pose (Pose): Pose of the marker
            scale (Vector3): Scale of the marker
            color (ColorRGBA): Color of the marker
            lifetime (float): Lifetime of the marker in seconds (0 = persistent)
            points (List[Point]): Points for LINE_STRIP, LINE_LIST, etc.
            colors (List[ColorRGBA]): Colors for points
            text (str): Text for TEXT_VIEW_FACING marker
            mesh_resource (str): Mesh resource for MESH_RESOURCE marker
            mesh_use_embedded_materials (bool): Whether to use embedded materials for MESH_RESOURCE
            
        Returns:
            Marker: The created marker
        """
        marker = Marker()
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.header.frame_id = frame_id if frame_id else self.default_frame_id
        marker.ns = namespace if namespace else self.default_namespace
        marker.id = id if id is not None else self.marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        
        if pose:
            marker.pose = pose
        else:
            marker.pose.orientation.w = 1.0
        
        if scale:
            marker.scale = scale
        else:
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
        
        if color:
            marker.color = color
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
        
        if lifetime is not None:
            marker.lifetime = Duration(sec=int(lifetime), nanosec=int((lifetime % 1) * 1e9))
        else:
            marker.lifetime = Duration(sec=int(self.default_lifetime), 
                                      nanosec=int((self.default_lifetime % 1) * 1e9))
        
        if points:
            marker.points = points
        
        if colors:
            marker.colors = colors
        
        if text:
            marker.text = text
        
        if mesh_resource:
            marker.mesh_resource = mesh_resource
            marker.mesh_use_embedded_materials = mesh_use_embedded_materials
        
        if id is None:
            self.marker_id += 1
        
        key = f"{marker.ns}_{marker.id}"
        self.markers[key] = marker
        
        return marker
    
    def create_sphere_marker(self, position: Union[Tuple[float, float, float], Point],
                           radius: float = 0.1, color: Optional[Tuple[float, float, float, float]] = None,
                           frame_id: str = None, namespace: str = None, id: int = None,
                           lifetime: float = None):
        """
        Create a sphere marker at the specified position.
        
        Args:
            position: Position of the sphere, either as a tuple (x, y, z) or a Point
            radius (float): Radius of the sphere
            color: Color of the sphere as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        pose = Pose()
        if isinstance(position, Point):
            pose.position = position
        else:
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        scale = Vector3(x=radius*2, y=radius*2, z=radius*2)
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        return self.create_marker(Marker.SPHERE, frame_id, namespace, id, pose, scale, marker_color, lifetime)
    
    def create_cube_marker(self, position: Union[Tuple[float, float, float], Point],
                         scale: Union[float, Tuple[float, float, float], Vector3] = 0.1,
                         color: Optional[Tuple[float, float, float, float]] = None,
                         frame_id: str = None, namespace: str = None, id: int = None,
                         lifetime: float = None):
        """
        Create a cube marker at the specified position.
        
        Args:
            position: Position of the cube, either as a tuple (x, y, z) or a Point
            scale: Scale of the cube, either as a float for uniform scale or a tuple (x, y, z) or a Vector3
            color: Color of the cube as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        pose = Pose()
        if isinstance(position, Point):
            pose.position = position
        else:
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        if isinstance(scale, (float, int)):
            scale_vec = Vector3(x=scale, y=scale, z=scale)
        elif isinstance(scale, Vector3):
            scale_vec = scale
        else:
            scale_vec = Vector3(x=scale[0], y=scale[1], z=scale[2])
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        return self.create_marker(Marker.CUBE, frame_id, namespace, id, pose, scale_vec, marker_color, lifetime)
    
    def create_text_marker(self, position: Union[Tuple[float, float, float], Point],
                         text: str, height: float = 0.1, color: Optional[Tuple[float, float, float, float]] = None,
                         frame_id: str = None, namespace: str = None, id: int = None,
                         lifetime: float = None):
        """
        Create a text marker at the specified position.
        
        Args:
            position: Position of the text, either as a tuple (x, y, z) or a Point
            text (str): Text to display
            height (float): Height of the text
            color: Color of the text as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        pose = Pose()
        if isinstance(position, Point):
            pose.position = position
        else:
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        scale = Vector3(x=0.0, y=0.0, z=height)
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        return self.create_marker(Marker.TEXT_VIEW_FACING, frame_id, namespace, id, pose, scale, marker_color, lifetime, text=text)
    
    def create_line_marker(self, points: List[Union[Tuple[float, float, float], Point]],
                         width: float = 0.01, color: Optional[Tuple[float, float, float, float]] = None,
                         frame_id: str = None, namespace: str = None, id: int = None,
                         lifetime: float = None, line_strip: bool = True):
        """
        Create a line marker with the specified points.
        
        Args:
            points: List of points, either as tuples (x, y, z) or Points
            width (float): Width of the line
            color: Color of the line as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            line_strip (bool): Whether to create a LINE_STRIP (True) or LINE_LIST (False)
            
        Returns:
            Marker: The created marker
        """
        if not points or len(points) < 2:
            raise ValueError("At least two points are required for a line marker")
        
        marker_points = []
        for point in points:
            if isinstance(point, Point):
                marker_points.append(point)
            else:
                marker_points.append(Point(x=point[0], y=point[1], z=point[2]))
        
        pose = Pose()
        pose.orientation.w = 1.0
        
        scale = Vector3(x=width, y=0.0, z=0.0)
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        marker_type = Marker.LINE_STRIP if line_strip else Marker.LINE_LIST
        
        return self.create_marker(marker_type, frame_id, namespace, id, pose, scale, marker_color, lifetime, points=marker_points)
    
    def create_arrow_marker(self, start: Union[Tuple[float, float, float], Point],
                          end: Union[Tuple[float, float, float], Point],
                          shaft_diameter: float = 0.02, head_diameter: float = 0.05,
                          head_length: float = 0.1, color: Optional[Tuple[float, float, float, float]] = None,
                          frame_id: str = None, namespace: str = None, id: int = None,
                          lifetime: float = None):
        """
        Create an arrow marker from start to end.
        
        Args:
            start: Start position of the arrow, either as a tuple (x, y, z) or a Point
            end: End position of the arrow, either as a tuple (x, y, z) or a Point
            shaft_diameter (float): Diameter of the arrow shaft
            head_diameter (float): Diameter of the arrow head
            head_length (float): Length of the arrow head
            color: Color of the arrow as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        start_point = Point()
        if isinstance(start, Point):
            start_point = start
        else:
            start_point.x = start[0]
            start_point.y = start[1]
            start_point.z = start[2]
        
        end_point = Point()
        if isinstance(end, Point):
            end_point = end
        else:
            end_point.x = end[0]
            end_point.y = end[1]
            end_point.z = end[2]
        
        pose = Pose()
        pose.orientation.w = 1.0
        
        scale = Vector3(x=shaft_diameter, y=head_diameter, z=head_length)
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        points = [start_point, end_point]
        
        return self.create_marker(Marker.ARROW, frame_id, namespace, id, pose, scale, marker_color, lifetime, points=points)
    
    def create_pose_marker(self, pose: Union[Pose, PoseStamped],
                         axis_length: float = 0.2, axis_width: float = 0.02,
                         frame_id: str = None, namespace: str = None, id: int = None,
                         lifetime: float = None):
        """
        Create markers to visualize a pose (position and orientation).
        
        Args:
            pose: Pose to visualize, either a Pose or PoseStamped
            axis_length (float): Length of the orientation axes
            axis_width (float): Width of the orientation axes
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            List[Marker]: List of markers for the pose
        """
        if isinstance(pose, PoseStamped):
            actual_pose = pose.pose
            if frame_id is None:
                frame_id = pose.header.frame_id
        else:
            actual_pose = pose
        
        position = actual_pose.position
        orientation = actual_pose.orientation
        
        # Use quaternion to get the orientation axes
        from tf_transformations import quaternion_matrix
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        matrix = quaternion_matrix(q)
        
        x_axis = matrix[:3, 0] * axis_length
        y_axis = matrix[:3, 1] * axis_length
        z_axis = matrix[:3, 2] * axis_length
        
        # Create markers for each axis
        markers = []
        
        # X axis (red)
        x_end = (position.x + x_axis[0], position.y + x_axis[1], position.z + x_axis[2])
        x_marker = self.create_arrow_marker((position.x, position.y, position.z), x_end,
                                          shaft_diameter=axis_width, head_diameter=axis_width*2,
                                          head_length=axis_length*0.2, color=(1, 0, 0, 1),
                                          frame_id=frame_id, namespace=namespace+'_x' if namespace else 'pose_x',
                                          id=id if id is not None else None, lifetime=lifetime)
        markers.append(x_marker)
        
        # Y axis (green)
        y_end = (position.x + y_axis[0], position.y + y_axis[1], position.z + y_axis[2])
        y_marker = self.create_arrow_marker((position.x, position.y, position.z), y_end,
                                          shaft_diameter=axis_width, head_diameter=axis_width*2,
                                          head_length=axis_length*0.2, color=(0, 1, 0, 1),
                                          frame_id=frame_id, namespace=namespace+'_y' if namespace else 'pose_y',
                                          id=id if id is not None else None, lifetime=lifetime)
        markers.append(y_marker)
        
        # Z axis (blue)
        z_end = (position.x + z_axis[0], position.y + z_axis[1], position.z + z_axis[2])
        z_marker = self.create_arrow_marker((position.x, position.y, position.z), z_end,
                                          shaft_diameter=axis_width, head_diameter=axis_width*2,
                                          head_length=axis_length*0.2, color=(0, 0, 1, 1),
                                          frame_id=frame_id, namespace=namespace+'_z' if namespace else 'pose_z',
                                          id=id if id is not None else None, lifetime=lifetime)
        markers.append(z_marker)
        
        return markers
    
    def create_trajectory_marker(self, poses: List[Union[Tuple[float, float, float], Point, Pose, PoseStamped]],
                               width: float = 0.01, color: Optional[Tuple[float, float, float, float]] = None,
                               frame_id: str = None, namespace: str = None, id: int = None,
                               lifetime: float = None):
        """
        Create a trajectory marker from a list of poses or positions.
        
        Args:
            poses: List of poses or positions to create a trajectory from
            width (float): Width of the trajectory line
            color: Color of the trajectory as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        if not poses:
            raise ValueError("No poses provided for trajectory marker")
        
        points = []
        for pose in poses:
            if isinstance(pose, Pose) or isinstance(pose, PoseStamped):
                if isinstance(pose, PoseStamped):
                    position = pose.pose.position
                else:
                    position = pose.position
                points.append((position.x, position.y, position.z))
            elif isinstance(pose, Point):
                points.append((pose.x, pose.y, pose.z))
            else:
                points.append(pose)
        
        return self.create_line_marker(points, width, color, frame_id, namespace, id, lifetime, line_strip=True)
    
    def create_point_cloud_marker(self, points: List[Union[Tuple[float, float, float], Point]],
                                point_size: float = 0.01, color: Optional[Tuple[float, float, float, float]] = None,
                                frame_id: str = None, namespace: str = None, id: int = None,
                                lifetime: float = None):
        """
        Create a point cloud marker from a list of points.
        
        Args:
            points: List of points, either as tuples (x, y, z) or Points
            point_size (float): Size of the points
            color: Color of the points as a tuple (r, g, b, a)
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        if not points:
            raise ValueError("No points provided for point cloud marker")
        
        marker_points = []
        for point in points:
            if isinstance(point, Point):
                marker_points.append(point)
            else:
                marker_points.append(Point(x=point[0], y=point[1], z=point[2]))
        
        pose = Pose()
        pose.orientation.w = 1.0
        
        scale = Vector3(x=point_size, y=point_size, z=point_size)
        
        marker_color = None
        if color:
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        return self.create_marker(Marker.POINTS, frame_id, namespace, id, pose, scale, marker_color, lifetime, points=marker_points)
    
    def publish_marker(self, marker: Marker):
        """
        Publish a single marker.
        
        Args:
            marker (Marker): The marker to publish
        """
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)
    
    def publish_markers(self, markers: List[Marker]):
        """
        Publish multiple markers.
        
        Args:
            markers (List[Marker]): The markers to publish
        """
        marker_array = MarkerArray()
        marker_array.markers.extend(markers)
        self.publisher.publish(marker_array)
    
    def publish_all_markers(self):
        """
        Publish all markers that have been created.
        """
        if not self.markers:
            return
        
        marker_array = MarkerArray()
        marker_array.markers = list(self.markers.values())
        self.publisher.publish(marker_array)
    
    def create_frame_marker(self, frame_id: str, parent_frame_id: str,
                          axis_length: float = 0.2, axis_width: float = 0.02,
                          namespace: str = None, id: int = None, lifetime: float = None):
        """
        Create markers to visualize a coordinate frame.
        
        Args:
            frame_id (str): The frame ID to visualize
            parent_frame_id (str): The parent frame ID for the markers
            axis_length (float): Length of the coordinate axes
            axis_width (float): Width of the coordinate axes
            namespace (str): Namespace for the markers
            id (int): Base ID for the markers
            lifetime (float): Lifetime of the markers in seconds
            
        Returns:
            List[Marker]: List of markers for the coordinate frame
        """
        # Try to look up the transform from parent to child frame
        try:
            transform = self.node.tf_buffer.lookup_transform(
                parent_frame_id, frame_id, rclpy.time.Time())
            
            # Create a pose from the transform
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            
            # Create pose markers
            return self.create_pose_marker(pose, axis_length, axis_width,
                                         parent_frame_id, namespace, id, lifetime)
        except Exception as e:
            self.node.get_logger().error(f"Failed to look up transform from {parent_frame_id} to {frame_id}: {e}")
            return []
    
    def create_cone_marker(self, position: Union[Tuple[float, float, float], Point],
                         color_name: str, radius: float = 0.1, height: float = 0.2,
                         frame_id: str = None, namespace: str = None, id: int = None,
                         lifetime: float = None):
        """
        Create a cone marker at the specified position with a specific color.
        
        Args:
            position: Position of the cone, either as a tuple (x, y, z) or a Point
            color_name (str): Color name ('Blue cone', 'Yellow cone', 'Red cone', etc.)
            radius (float): Radius of the cone base
            height (float): Height of the cone
            frame_id (str): Frame ID for the marker
            namespace (str): Namespace for the marker
            id (int): ID for the marker
            lifetime (float): Lifetime of the marker in seconds
            
        Returns:
            Marker: The created marker
        """
        pose = Pose()
        if isinstance(position, Point):
            pose.position = position
        else:
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        scale = Vector3(x=radius*2, y=radius*2, z=height)
        
        # Determine color from the color name
        marker_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)  # Default gray
        
        if 'blue' in color_name.lower():
            marker_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        elif 'yellow' in color_name.lower():
            marker_color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        elif 'red' in color_name.lower():
            marker_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        elif 'orange' in color_name.lower():
            marker_color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0)
        
        # Use CONE if available in RViz, otherwise use CYLINDER
        try:
            return self.create_marker(Marker.CYLINDER, frame_id, namespace, id, pose, scale, marker_color, lifetime)
        except Exception:
            return self.create_marker(Marker.CYLINDER, frame_id, namespace, id, pose, scale, marker_color, lifetime)
    
    def visualize_cone_landmarks(self, landmarks: Dict[int, Dict[str, Any]], frame_id: str = 'map',
                                namespace: str = 'landmarks', lifetime: float = None):
        """
        Visualize cone landmarks from the SLAM map.
        
        Args:
            landmarks (Dict[int, Dict[str, Any]]): Dictionary of landmark track IDs to landmark data,
                                                where each landmark data contains at least 'pos_map' and 'color'
            frame_id (str): Frame ID for the markers
            namespace (str): Namespace for the markers
            lifetime (float): Lifetime of the markers in seconds
            
        Returns:
            List[Marker]: List of created markers
        """
        markers = []
        
        # First add a deletion marker to clear previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.ns = namespace
        markers.append(delete_marker)
        
        # Then add a marker for each landmark
        for track_id, landmark in landmarks.items():
            if 'pos_map' not in landmark or 'color' not in landmark:
                continue
            
            position = landmark['pos_map']
            color_name = landmark['color']
            
            # Use track_id as the marker id for consistent visualization
            marker = self.create_cone_marker(position, color_name, radius=0.15, height=0.3,
                                          frame_id=frame_id, namespace=namespace, id=track_id,
                                          lifetime=lifetime)
            markers.append(marker)
            
            # Add a text marker with the track ID
            text_position = (position[0], position[1], position[2] + 0.4)
            text_marker = self.create_text_marker(text_position, f"ID: {track_id}", height=0.1,
                                              color=(1.0, 1.0, 1.0, 0.8), frame_id=frame_id,
                                              namespace=f"{namespace}_labels", id=track_id,
                                              lifetime=lifetime)
            markers.append(text_marker)
        
        return markers 
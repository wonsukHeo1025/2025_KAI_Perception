import os
import rclpy
from rclpy.node import Node
from typing import Tuple
import cv2
import numpy as np
import yaml

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from custom_interface.msg import ModifiedFloat32MultiArray
from yolo_msgs.msg import *

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

from yolo_ros.yolo_debug_node import YoloDebugNode

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    return T

def load_camera_calibration(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs

def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Get field names and print for debugging
    field_names = [f.name for f in cloud_msg.fields]
    print(f"Available fields in pointcloud: {field_names}")
    
    # Try to find x, y, z fields with different possible names
    x_field = None
    y_field = None
    z_field = None
    
    for field in cloud_msg.fields:
        if field.name.lower() in ['x', 'point_x']:
            x_field = field
        elif field.name.lower() in ['y', 'point_y']:
            y_field = field
        elif field.name.lower() in ['z', 'point_z']:
            z_field = field
    
    if not all([x_field, y_field, z_field]):
        print(f"Missing required fields. Found: x={x_field}, y={y_field}, z={z_field}")
        return np.zeros((0,3), dtype=np.float32)
    
    # Create dtype based on actual field names
    dtype = np.dtype([
        (x_field.name, np.float32),
        (y_field.name, np.float32),
        (z_field.name, np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])
    
    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:,0] = raw_data[x_field.name]
    points[:,1] = raw_data[y_field.name]
    points[:,2] = raw_data[z_field.name]
    
    if skip_rate > 1:
        points = points[::skip_rate]
    
    return points

class FusionProjectionNode(Node):  
    def __init__(self):
        super().__init__('fusion_projection_node')
        
        config_file = extract_configuration() 
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        cones_topic = "/sorted_cones_time"  # cones 토픽
        boxes_topic = "/detections"  # bounding boxes 토픽

        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")
        self.get_logger().info(f"Subscribing to boxes topic: {boxes_topic}")

        # message_filters를 이용해 4개 토픽을 동기화
        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic, qos_profile=best_effort_qos)
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)
        self.boxes_sub = Subscriber(self, DetectionArray, boxes_topic, qos_profile=best_effort_qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.cones_sub, self.boxes_sub],
            queue_size=5,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)

        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        self.skip_rate = 1

        # Color mapping for visualization (BGR format)
        self.color_mapping = {
            "red cone": (0, 0, 255),   # Red
            "yellow cone":  (0, 255, 255), # Yellow
            "blue cone":    (255, 0, 0),   # Blue
            "Unknown":      (0, 255, 0)    # Green (default)
        }

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2, cones_msg: ModifiedFloat32MultiArray, boxes_msg: DetectionArray):
        try:
            if not image_msg.data:
                self.get_logger().warn("Received empty image message, skipping processing")
                return
                
            # 1. 이미지 메시지를 OpenCV 이미지로 변환
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
                if image_msg.encoding != 'bgr8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                return

            # 2. 포인트 클라우드 메시지(LiDAR)를 XYZ 배열로 변환
            self.get_logger().info(f"Processing pointcloud with {lidar_msg.width}x{lidar_msg.height} points")
            xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
            n_points = xyz_lidar.shape[0]
            self.get_logger().info(f"Converted to {n_points} XYZ points")
            
            if n_points > 0:  # LiDAR 데이터가 있는 경우 처리
                # 3. 포인트 클라우드 데이터 준비: 동차 좌표 변환
                xyz_lidar_f64 = xyz_lidar.astype(np.float64)
                ones = np.ones((n_points, 1), dtype=np.float64)
                xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))  # 동차 좌표 (N, 4)

                # 4. LiDAR 좌표계를 카메라 좌표계로 변환
                xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
                xyz_cam = xyz_cam_h[:, :3]  # 동차 좌표에서 3D 좌표 추출

                # 5. 카메라 앞에 있는 포인트 필터링
                mask_in_front = (xyz_cam[:, 2] > 0.0)  # z > 0: 카메라 앞에 위치한 포인트
                xyz_cam_front = xyz_cam[mask_in_front]
                
                n_front = xyz_cam_front.shape[0]
                if n_front > 0:
                    # 6. 3D 포인트를 2D 이미지로 투영
                    rvec = np.zeros((3,1), dtype=np.float64)
                    tvec = np.zeros((3,1), dtype=np.float64)
                    lidar_image_points, _ = cv2.projectPoints(
                        xyz_cam_front,
                        rvec, tvec,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    lidar_image_points = lidar_image_points.reshape(-1, 2)

                    # 7. 투영된 LiDAR 포인트를 이미지 위에 시각화 (초록색 점으로)
                    h, w = cv_image.shape[:2]
                    for (u, v) in lidar_image_points:
                        u_int = int(round(u))
                        v_int = int(round(v))
                        if 0 <= u_int < w and 0 <= v_int < h:
                            cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)  # 초록색 원
                else:
                    self.get_logger().info("No LiDAR points in front of camera (z>0).")

            # 8. cones 메시지 처리 - Simplified for 3D only
            cone_data = np.array(cones_msg.data, dtype=np.float32)
            if cone_data.size == 0:
                self.get_logger().warn("Empty cones data.")
            else:
                # Check for layout structure
                if len(cones_msg.layout.dim) < 1:
                    self.get_logger().error("Cone layout dimension is missing or invalid.")
                    return
                
                num_points = cones_msg.layout.dim[0].size
                
                # --- SIMPLIFIED: Only support 3D format ---
                # Check for 3D format (either with explicit layout or inferred)
                if len(cones_msg.layout.dim) == 2 and cones_msg.layout.dim[1].size == 3:
                    # Explicitly defined 3D data [x, y, z]
                    expected_size = num_points * 3
                    if expected_size != cone_data.size:
                        self.get_logger().error(
                            f"3D Cone data size ({cone_data.size}) does not match layout "
                            f"({num_points} cones * 3 values = {expected_size})."
                        )
                        return
                elif cone_data.size % 3 == 0:
                    # Infer from data size if divisible by 3
                    num_points = cone_data.size // 3
                    self.get_logger().info(f"Inferred 3D data with {num_points} points")
                else:
                    # Not 3D data
                    self.get_logger().error(
                        "Data is not in 3D format. Expected 3 values per point."
                    )
                    return
                
                # Get 3D points directly
                self.get_logger().info("Processing 3D cone data (X, Y, Z)")
                cones_xyz = cone_data.reshape(num_points, 3)  # (N,3) array
                
                # 9. Project cone points to image plane
                # Convert to homogeneous coordinates
                cones_xyz_h = np.hstack((cones_xyz, np.ones((cones_xyz.shape[0], 1), dtype=np.float32)))
                # Transform from LiDAR to camera coordinate system
                cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T
                cones_cam = cones_cam_h[:, :3]  # Extract 3D coordinates from homogeneous
                
                # Filter points in front of camera
                mask_cones_front = (cones_cam[:, 2] > 0.0)
                cones_cam_front = cones_cam[mask_cones_front]
                
                if cones_cam_front.shape[0] > 0:
                    # Project to image plane
                    rvec = np.zeros((3,1), dtype=np.float64)
                    tvec = np.zeros((3,1), dtype=np.float64)
                    cone_image_points, _ = cv2.projectPoints(
                        cones_cam_front.astype(np.float64),
                        rvec, tvec,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    cone_image_points = cone_image_points.reshape(-1, 2)
                    
                    # 10. Visualize projected cone points on the image
                    h, w = cv_image.shape[:2]
                    
                    # If we have class names, use those for colors
                    has_class_names = hasattr(cones_msg, 'class_names') and len(cones_msg.class_names) > 0
                    valid_points_count = mask_cones_front.sum()
                    
                    for i, (u, v) in enumerate(cone_image_points):
                        u_int = int(round(u))
                        v_int = int(round(v))
                        if 0 <= u_int < w and 0 <= v_int < h:
                            # Default color: red
                            color = (0, 0, 255)
                            
                            # If class names are available, use appropriate colors
                            if has_class_names and i < valid_points_count and i < len(cones_msg.class_names):
                                class_name = cones_msg.class_names[i] 
                                color = self.color_mapping.get(class_name, (0, 0, 255))
                            
                            # Draw the cone marker
                            cv2.circle(cv_image, (u_int, v_int), 4, color, -1)  # Filled circle
                            cv2.circle(cv_image, (u_int, v_int), 6, (255, 255, 255), 1)  # White border
                            
                            # Add Z-depth info
                            z_depth = cones_cam_front[i, 2]
                            z_text = f"{z_depth:.1f}m"
                            cv2.putText(cv_image, z_text, (u_int+7, v_int), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 11. bounding boxes 메시지 처리
            for i, detection in enumerate(boxes_msg.detections):
                # Get bounding box center coordinates
                x_center = detection.bbox.center.position.x
                y_center = detection.bbox.center.position.y

                # Get bounding box dimensions
                width = detection.bbox.size.x
                height = detection.bbox.size.y

                # Get class information
                cls_name = detection.class_name
                conf = detection.score

                # Get bounding box coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Get color from color mapping
                color = self.color_mapping.get(cls_name, (0, 255, 0))  # Default: green

                # Draw bounding box
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # Add label text
                label = f"{cls_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(cv_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 10), color, -1)
                cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 12. 결과 이미지를 퍼블리시
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = FusionProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

import os
import rclpy
from rclpy.node import Node
from typing import Tuple
import cv2
import numpy as np
import yaml
from numba import jit, prange
import time

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType
from custom_interface.msg import ModifiedFloat32MultiArray

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

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

@jit(nopython=True, parallel=True, cache=True)
def extract_xyz_points_fast(raw_data_flat: np.ndarray, x_offset: int, y_offset: int, z_offset: int, 
                           point_step: int, num_points: int, skip_rate: int) -> np.ndarray:
    """Numba JIT 최적화된 XYZ 포인트 추출"""
    effective_points = (num_points + skip_rate - 1) // skip_rate
    points = np.zeros((effective_points, 3), dtype=np.float32)
    
    point_idx = 0
    for i in prange(0, num_points, skip_rate):
        if point_idx >= effective_points:
            break
            
        base_offset = i * point_step
        x_bytes = raw_data_flat[base_offset + x_offset:base_offset + x_offset + 4]
        y_bytes = raw_data_flat[base_offset + y_offset:base_offset + y_offset + 4]
        z_bytes = raw_data_flat[base_offset + z_offset:base_offset + z_offset + 4]
        
        # Convert bytes to float32 (assuming little-endian)
        x_val = np.frombuffer(x_bytes.tobytes(), dtype=np.float32)[0]
        y_val = np.frombuffer(y_bytes.tobytes(), dtype=np.float32)[0]
        z_val = np.frombuffer(z_bytes.tobytes(), dtype=np.float32)[0]
        
        points[point_idx, 0] = x_val
        points[point_idx, 1] = y_val
        points[point_idx, 2] = z_val
        point_idx += 1
    
    return points[:point_idx]

@jit(nopython=True, parallel=True, cache=True)
def transform_points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """포인트를 동차 좌표로 변환 (Numba 최적화)"""
    n_points = points.shape[0]
    points_h = np.zeros((n_points, 4), dtype=np.float64)
    
    for i in prange(n_points):
        points_h[i, 0] = points[i, 0]
        points_h[i, 1] = points[i, 1]
        points_h[i, 2] = points[i, 2]
        points_h[i, 3] = 1.0
    
    return points_h

@jit(nopython=True, parallel=True, cache=True)
def batch_matrix_multiply(points_h: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """배치 행렬 곱셈 (Numba 최적화)"""
    n_points = points_h.shape[0]
    result = np.zeros((n_points, 4), dtype=np.float64)
    
    for i in prange(n_points):
        for j in range(4):
            for k in range(4):
                result[i, j] += points_h[i, k] * transform_matrix[k, j]
    
    return result

@jit(nopython=True, parallel=True, cache=True)
def filter_points_in_front(points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """카메라 앞에 있는 포인트만 필터링 (Numba 최적화)"""
    n_points = points_cam.shape[0]
    mask = np.zeros(n_points, dtype=np.bool_)
    
    for i in prange(n_points):
        mask[i] = points_cam[i, 2] > 0.0
    
    # Count valid points
    count = 0
    for i in range(n_points):
        if mask[i]:
            count += 1
    
    # Extract valid points
    valid_points = np.zeros((count, 3), dtype=np.float64)
    valid_indices = np.zeros(count, dtype=np.int32)
    idx = 0
    for i in range(n_points):
        if mask[i]:
            valid_points[idx, 0] = points_cam[i, 0]
            valid_points[idx, 1] = points_cam[i, 1]
            valid_points[idx, 2] = points_cam[i, 2]
            valid_indices[idx] = i
            idx += 1
    
    return valid_points, valid_indices

@jit(nopython=True, parallel=True, cache=True)
def project_points_to_image_fast(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """3D 포인트를 2D 이미지로 투영 (Numba 최적화, 왜곡 보정 없이)"""
    n_points = points_3d.shape[0]
    image_points = np.zeros((n_points, 2), dtype=np.float64)
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    for i in prange(n_points):
        if points_3d[i, 2] > 1e-6:  # Avoid division by zero or very small z
            x = points_3d[i, 0] / points_3d[i, 2]
            y = points_3d[i, 1] / points_3d[i, 2]
            
            image_points[i, 0] = fx * x + cx
            image_points[i, 1] = fy * y + cy
    
    return image_points

def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32), None, None

    field_map = {field.name: field for field in cloud_msg.fields}

    x_field = field_map.get('x') or field_map.get('point_x')
    y_field = field_map.get('y') or field_map.get('point_y')
    z_field = field_map.get('z') or field_map.get('point_z')
    intensity_field = field_map.get('intensity')
    range_field = field_map.get('range')

    if not all([x_field, y_field, z_field]):
        print(f"Missing required XYZ fields. Found: x={x_field}, y={y_field}, z={z_field}")
        return np.zeros((0, 3), dtype=np.float32), None, None

    parsed_fields_info = []
    if x_field: parsed_fields_info.append((x_field.name, np.float32, x_field.offset))
    if y_field: parsed_fields_info.append((y_field.name, np.float32, y_field.offset))
    if z_field: parsed_fields_info.append((z_field.name, np.float32, z_field.offset))
    if intensity_field and intensity_field.datatype == 7: # float32
        parsed_fields_info.append((intensity_field.name, np.float32, intensity_field.offset))
    elif intensity_field: # 다른 데이터 타입의 intensity는 일단 경고
        print(f"Warning: Intensity field '{intensity_field.name}' found but has unhandled datatype {intensity_field.datatype}. Expected 7 (float32).")
        intensity_field = None # 처리하지 않음
        
    if range_field and range_field.datatype == 6: # uint32
        parsed_fields_info.append((range_field.name, np.uint32, range_field.offset))
    elif range_field: # 다른 데이터 타입의 range는 일단 경고
        print(f"Warning: Range field '{range_field.name}' found but has unhandled datatype {range_field.datatype}. Expected 6 (uint32).")
        range_field = None # 처리하지 않음

    # 오프셋 기준으로 정렬하여 dtype 생성 시 순서 보장
    parsed_fields_info.sort(key=lambda f: f[2])
    
    dtype_build_list = []
    last_offset = 0

    for name, dtype_val, offset in parsed_fields_info:
        padding_bytes = offset - last_offset
        if padding_bytes < 0:
            print(f"Error: Negative padding bytes calculated for field {name}. Offsets may be incorrect or overlapping.")
            return np.zeros((0,3), dtype=np.float32), None, None
        if padding_bytes > 0:
            dtype_build_list.append( ('', f'V{padding_bytes}')) # Padding
        dtype_build_list.append((name, dtype_val))
        
        field_byte_size = np.dtype(dtype_val).itemsize
        last_offset = offset + field_byte_size
        
    remaining_bytes = cloud_msg.point_step - last_offset
    if remaining_bytes < 0:
        print(f"Error: Negative remaining bytes calculated. Point_step {cloud_msg.point_step} may be smaller than declared fields.")
        return np.zeros((0,3), dtype=np.float32), None, None
    if remaining_bytes > 0:
        dtype_build_list.append(('', f'V{remaining_bytes}'))

    try:
        final_dtype = np.dtype(dtype_build_list)
    except Exception as e:
        print(f"Error creating numpy dtype: {e}. Dtype list: {dtype_build_list}")
        return np.zeros((0,3), dtype=np.float32), None, None
        
    raw_data = np.frombuffer(cloud_msg.data, dtype=final_dtype)

    if skip_rate > 1:
        raw_data = raw_data[::skip_rate]

    points_xyz = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points_xyz[:, 0] = raw_data[x_field.name]
    points_xyz[:, 1] = raw_data[y_field.name]
    points_xyz[:, 2] = raw_data[z_field.name]

    intensity_values = None
    if intensity_field and intensity_field.name in raw_data.dtype.names:
        intensity_values = raw_data[intensity_field.name].astype(np.float32)

    range_values = None
    if range_field and range_field.name in raw_data.dtype.names:
        range_values = raw_data[range_field.name].astype(np.float32) / 1000.0 # mm to m

    return points_xyz, intensity_values, range_values

class FusionProjectionNode(Node):  
    def __init__(self):
        super().__init__('fusion_projection_node')
        
        # ROS 파라미터 선언
        self.declare_parameter('colorization_mode', 'intensity', 
                                ParameterDescriptor(description='Colorization mode for LiDAR points (intensity, range, or none)',
                                                    type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('min_value_display', 0.0, 
                                ParameterDescriptor(description='Minimum value for color mapping (intensity or range)',
                                                    type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('max_value_display', 100.0, 
                                ParameterDescriptor(description='Maximum value for color mapping (intensity or range)',
                                                    type=ParameterType.PARAMETER_DOUBLE))
        self.colorization_mode = self.get_parameter('colorization_mode').get_parameter_value().string_value
        self.min_value_display = self.get_parameter('min_value_display').get_parameter_value().double_value
        self.max_value_display = self.get_parameter('max_value_display').get_parameter_value().double_value

        # 파라미터 변경 콜백 등록
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(f"Colorization mode: {self.colorization_mode}, Min: {self.min_value_display}, Max: {self.max_value_display}")

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

        self.T_sensor_to_lidar = np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1, -0.038195],
            [ 0,  0,  0,  1]
        ], dtype=np.float64)
        self.T_sensor_to_cam = self.T_lidar_to_cam @ self.T_sensor_to_lidar

        self.get_logger().info("Loaded T_lidar_to_cam:\\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Calculated T_sensor_to_lidar:\\n{}".format(self.T_sensor_to_lidar))
        self.get_logger().info("Calculated T_sensor_to_cam:\\n{}".format(self.T_sensor_to_cam))
        self.get_logger().info("Camera matrix:\\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        yolo_image_topic = "/camera_1/dbg_image"
        cones_topic = "/sorted_cones_time"

        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to YOLO processed image topic: {yolo_image_topic}")
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")

        self.image_sub = Subscriber(self, Image, yolo_image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic, qos_profile=best_effort_qos)
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.cones_sub],
            queue_size=5,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)

        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        self.skip_rate = 1

        self.color_mapping = {
            "red cone": (0, 0, 255),
            "yellow cone":  (0, 255, 255),
            "blue cone":    (255, 0, 0),
            "Unknown":      (0, 255, 0)
        }

        self.frame_count = 0
        self.total_processing_time = 0.0

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'colorization_mode':
                self.colorization_mode = param.value
            elif param.name == 'min_value_display':
                self.min_value_display = param.value
            elif param.name == 'max_value_display':
                self.max_value_display = param.value
        self.get_logger().info(f"Updated parameters: Mode: {self.colorization_mode}, Min: {self.min_value_display}, Max: {self.max_value_display}")
        return SetParametersResult(successful=True)

    def _get_rainbow_colors_array(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """값들의 배열에 따라 무지개 색상 배열을 반환합니다."""
        if values is None or values.size == 0:
            return np.array([], dtype=np.uint8).reshape(0, 3) # 빈 (0,3) 배열 반환
        
        if min_val >= max_val:
            # 유효하지 않은 범위면 기본 색상 (회색) 배열 반환
            return np.full((values.shape[0], 3), (128, 128, 128), dtype=np.uint8)
        
        # 값을 0-1로 정규화
        normalized_values = (values - min_val) / (max_val - min_val)
        normalized_values = np.clip(normalized_values, 0.0, 1.0)
        
        # 0-255 범위로 변환 (OpenCV 컬러맵은 uint8 이미지를 기대함)
        scaled_values = (normalized_values * 255).astype(np.uint8)
        
        # cv2.applyColorMap은 (H, W) 또는 (H, W, 1) 형태의 입력을 기대합니다.
        # scaled_values는 1D 배열이므로 (N, 1) 형태로 변환합니다.
        colormap_input = scaled_values.reshape(-1, 1)
        
        colored_bgr_array = cv2.applyColorMap(colormap_input, cv2.COLORMAP_JET)
        # colored_bgr_array는 (N, 1, 3) 형태이므로 (N, 3) 형태로 변환합니다.
        return colored_bgr_array.reshape(-1, 3)

    def get_rainbow_color(self, value: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
        """값에 따라 무지개 색상을 반환합니다. (단일 값 처리용 - 레거시 또는 개별 사용 가능)"""
        if min_val >= max_val:
            # 유효하지 않은 범위면 기본 색상 (회색) 반환
            return (128, 128, 128)
        
        # 값을 0-1로 정규화
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_value = np.clip(normalized_value, 0.0, 1.0)
        
        # 0-255 범위로 변환 (OpenCV 컬러맵은 uint8 이미지를 기대함)
        scaled_value = int(normalized_value * 255)
        
        # JET 컬러맵 적용을 위해 단일 픽셀 이미지 생성
        jet_pixel = np.array([[[scaled_value]]], dtype=np.uint8)
        colored_pixel = cv2.applyColorMap(jet_pixel, cv2.COLORMAP_JET)
        
        # BGR 튜플로 반환
        return (int(colored_pixel[0,0,0]), int(colored_pixel[0,0,1]), int(colored_pixel[0,0,2]))

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2, cones_msg: ModifiedFloat32MultiArray):
        try:
            start_time = time.time()
            
            if not image_msg.data:
                self.get_logger().warn("Received empty image message, skipping processing")
                return
                
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
                if image_msg.encoding != 'bgr8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                return

            lidar_start = time.time()
            xyz_lidar, intensity_lidar, range_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
            n_points = xyz_lidar.shape[0]
            lidar_time = time.time() - lidar_start
            
            if n_points > 0:
                transform_start = time.time()
                xyz_lidar_f64 = xyz_lidar.astype(np.float64)
                xyz_lidar_h = transform_points_to_homogeneous(xyz_lidar_f64)
                transform_time = time.time() - transform_start

                matrix_mult_start = time.time()
                xyz_cam_h = batch_matrix_multiply(xyz_lidar_h, self.T_lidar_to_cam.T)
                xyz_cam = xyz_cam_h[:, :3]
                matrix_mult_time = time.time() - matrix_mult_start

                filter_start = time.time()
                xyz_cam_front, valid_indices = filter_points_in_front(xyz_cam)
                filter_time = time.time() - filter_start
                
                # 필터링된 포인트에 해당하는 intensity 및 range 값 가져오기
                intensity_front = None
                if intensity_lidar is not None and valid_indices.size > 0:
                    intensity_front = intensity_lidar[valid_indices]
                
                range_front = None
                if range_lidar is not None and valid_indices.size > 0:
                    range_front = range_lidar[valid_indices]

                n_front = xyz_cam_front.shape[0]
                if n_front > 0:
                    project_start = time.time()
                    if np.any(self.dist_coeffs != 0):
                        rvec = np.zeros((3,1), dtype=np.float64)
                        tvec = np.zeros((3,1), dtype=np.float64)
                        lidar_image_points, _ = cv2.projectPoints(
                            xyz_cam_front,
                            rvec, tvec,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                        lidar_image_points = lidar_image_points.reshape(-1, 2)
                    else:
                        lidar_image_points = project_points_to_image_fast(xyz_cam_front, self.camera_matrix)
                    project_time = time.time() - project_start

                    viz_start = time.time()
                    h, w = cv_image.shape[:2]
                    default_color_tuple = (0, 255, 0) # 기본 초록색

                    if n_front > 0: # Process only if there are points in front
                        u_centers = np.round(lidar_image_points[:, 0]).astype(np.int32)
                        v_centers = np.round(lidar_image_points[:, 1]).astype(np.int32)

                        active_point_colors_bgr = None
                        data_to_colorize_np = None

                        if self.colorization_mode == 'intensity' and intensity_front is not None and intensity_front.size > 0:
                            data_to_colorize_np = intensity_front
                        elif self.colorization_mode == 'range' and range_front is not None and range_front.size > 0:
                            data_to_colorize_np = range_front

                        if data_to_colorize_np is not None:
                            active_point_colors_bgr = self._get_rainbow_colors_array(
                                data_to_colorize_np, 
                                self.min_value_display, 
                                self.max_value_display
                            )
                        else:
                            default_color_bgr_array = np.array([list(default_color_tuple)], dtype=np.uint8)
                            if n_front > 0: # Ensure n_front is positive before repeat
                                active_point_colors_bgr = np.repeat(default_color_bgr_array, n_front, axis=0)
                            else:
                                active_point_colors_bgr = np.array([], dtype=np.uint8).reshape(0,3)
                        
                        if active_point_colors_bgr.size > 0: # Ensure there are colors to draw
                            # 5x5 정사각형 (반지름 2)을 위한 오프셋
                            dv_offsets, du_offsets = np.mgrid[-1:2, -1:2] 
                            dv_flat = dv_offsets.flatten() # Shape (25,)
                            du_flat = du_offsets.flatten() # Shape (25,)

                            for dv, du in zip(dv_flat, du_flat):
                                current_v_coords = v_centers + dv
                                current_u_coords = u_centers + du

                                valid_mask = (current_v_coords >= 0) & (current_v_coords < h) & \
                                             (current_u_coords >= 0) & (current_u_coords < w)
                                
                                final_v = current_v_coords[valid_mask]
                                final_u = current_u_coords[valid_mask]
                                
                                if final_v.size > 0:
                                    colors_for_this_layer = active_point_colors_bgr[valid_mask]
                                    cv_image[final_v, final_u] = colors_for_this_layer
                    viz_time = time.time() - viz_start

            cone_start = time.time()
            cone_data = np.array(cones_msg.data, dtype=np.float32)
            if cone_data.size == 0:
                self.get_logger().warn("Empty cones data.")
            else:
                # Check for layout structure
                if len(cones_msg.layout.dim) < 1:
                    self.get_logger().error("Cone layout dimension is missing or invalid.")
                    return
                
                num_points_cone = cones_msg.layout.dim[0].size # 변수명 변경 (n_points와 충돌 방지)
                
                # --- SIMPLIFIED: Only support 3D format ---
                # Check for 3D format (either with explicit layout or inferred)
                if len(cones_msg.layout.dim) == 2 and cones_msg.layout.dim[1].size == 3:
                    # Explicitly defined 3D data [x, y, z]
                    expected_size = num_points_cone * 3
                    if expected_size != cone_data.size:
                        self.get_logger().error(
                            f"3D Cone data size ({cone_data.size}) does not match layout "
                            f"({num_points_cone} cones * 3 values = {expected_size})."
                        )
                        return
                elif cone_data.size % 3 == 0:
                    # Infer from data size if divisible by 3
                    num_points_cone = cone_data.size // 3 
                else:
                    # Not 3D data
                    self.get_logger().error(
                        "Data is not in 3D format. Expected 3 values per point."
                    )
                    return
                
                # Get 3D points directly
                cones_xyz = cone_data.reshape(num_points_cone, 3)  # (N,3) array
                
                # 9. Project cone points to image plane (Numba 최적화 적용)
                # Convert to homogeneous coordinates
                cones_xyz_h = transform_points_to_homogeneous(cones_xyz.astype(np.float64))
                # Transform from os_sensor to camera coordinate system using T_sensor_to_cam
                cones_cam_h = batch_matrix_multiply(cones_xyz_h, self.T_sensor_to_cam.T)
                cones_cam = cones_cam_h[:, :3]  # Extract 3D coordinates from homogeneous
                
                # Filter points in front of camera
                cones_cam_front, cones_valid_indices = filter_points_in_front(cones_cam)
                
                if cones_cam_front.shape[0] > 0:
                    # Project to image plane
                    if np.any(self.dist_coeffs != 0):
                        rvec = np.zeros((3,1), dtype=np.float64)
                        tvec = np.zeros((3,1), dtype=np.float64)
                        cone_image_points, _ = cv2.projectPoints(
                            cones_cam_front.astype(np.float64),
                            rvec, tvec,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                        cone_image_points = cone_image_points.reshape(-1, 2)
                    else:
                        cone_image_points = project_points_to_image_fast(cones_cam_front, self.camera_matrix)
                    
                    # 10. Visualize projected cone points on the image
                    h, w = cv_image.shape[:2]
                    
                    # If we have class names, use those for colors
                    has_class_names = hasattr(cones_msg, 'class_names') and len(cones_msg.class_names) > 0
                    # Get the class names corresponding to the points that are in front of the camera
                    valid_class_names = []
                    if has_class_names:
                       valid_class_names = [cones_msg.class_names[i] for i in cones_valid_indices if i < len(cones_msg.class_names)]

                    for i, (u, v) in enumerate(cone_image_points):
                        u_int = int(round(u))
                        v_int = int(round(v))
                        if 0 <= u_int < w and 0 <= v_int < h:
                            # Default color: red
                            color = (0, 0, 255)

                            # If class names are available and valid for this front point, use appropriate colors
                            if i < len(valid_class_names):
                                class_name = valid_class_names[i]
                                color = self.color_mapping.get(class_name, (0, 0, 255)) # Use mapped color, default red

                            # Draw the cone marker
                            cv2.circle(cv_image, (u_int, v_int), 4, color, -1)  # Filled circle
                            cv2.circle(cv_image, (u_int, v_int), 6, (255, 255, 255), 1)  # White border
                            
                            # Add Z-depth info
                            z_depth = cones_cam_front[i, 2]
                            z_text = f"{z_depth:.1f}m"
                            cv2.putText(cv_image, z_text, (u_int+7, v_int), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cone_time = time.time() - cone_start
            
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

            total_time = time.time() - start_time
            self.frame_count += 1
            self.total_processing_time += total_time
            
            if self.frame_count % 10 == 0:
                avg_time = self.total_processing_time / self.frame_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f"Performance: {avg_time:.3f}s/frame, {fps:.1f} FPS, {n_points} points processed")
                self.get_logger().info(f"LiDAR points: {n_points}, Intensity valid: {intensity_lidar is not None and np.any(intensity_lidar)}, Range valid: {range_lidar is not None and np.any(range_lidar)}")

                if n_points > 0:
                    self.get_logger().info(f"Timing breakdown - LiDAR_parse: {lidar_time:.4f}s, Transform: {transform_time:.4f}s, "
                                         f"MatrixMult: {matrix_mult_time:.4f}s, Filter: {filter_time:.4f}s, "
                                         f"Project: {project_time:.4f}s, Viz: {viz_time:.4f}s, Cone: {cone_time:.4f}s")

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

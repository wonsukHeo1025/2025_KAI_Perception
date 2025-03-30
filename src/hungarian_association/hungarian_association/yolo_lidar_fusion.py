import os
import cv2
import yaml
import numpy as np
import rclpy
from typing import Tuple, List, Optional

# ROS2 노드(Node) 관련 클래스 임포트
from rclpy.node import Node  # ROS2의 Node 클래스를 상속받아 사용

# ROS2 QoS (Quality of Service) 관련 설정 모듈 임포트
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # 메시지 전달 품질 설정

# message_filters를 사용해 여러 토픽의 메시지를 동기화하기 위한 Subscriber 및 ApproximateTimeSynchronizer 임포트
from message_filters import Subscriber, ApproximateTimeSynchronizer  # 메시지 동기화 도구 제공

# 헝가리안 알고리즘(최소 비용 매칭)을 위한 SciPy 모듈 임포트
from scipy.optimize import linear_sum_assignment  # 비용 행렬에 대해 최적 매칭 계산

# 헝가리안 알고리즘 설정을 로드하기 위한 커스텀 모듈 임포트
from hungarian_association.config_utils import load_hungarian_config  # 설정 파일 로드 함수 제공

# YOLO 감지 결과를 위한 메시지 타입 임포트
from yolo_msgs.msg import DetectionArray  # YOLO 감지 결과 메시지 (예: bounding box 정보)

# 커스텀 인터페이스 메시지 임포트 (라이다로부터 받은 데이터를 포함하는 메시지)
from custom_interface.msg import ModifiedFloat32MultiArray  # 수정된 Float32 배열 메시지 (예: 라이다 데이터 및 클래스 정보)


# ============================================================================
# 함수: load_extrinsic_matrix
# 설명: 주어진 YAML 파일에서 extrinsic matrix (라이다→카메라 변환 행렬)를 로드하여 numpy 배열로 반환함.
# ============================================================================
def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    # YAML 파일을 읽기 위해 파일 열기
    with open(yaml_path, 'r') as f:  # 파일을 읽기 모드로 엶
        data = yaml.safe_load(f)  # YAML 파일의 내용을 파싱하여 data 변수에 저장
    # YAML 파일에서 extrinsic_matrix 항목을 추출
    matrix_list = data['extrinsic_matrix']  # extrinsic_matrix 리스트 추출
    # 리스트를 numpy 배열로 변환 (데이터 타입은 float64)
    T = np.array(matrix_list, dtype=np.float64)  # 리스트를 float64 numpy 배열로 변환
    return T  # 변환된 extrinsic matrix 반환


# ============================================================================
# 함수: load_camera_calibration
# 설명: 주어진 YAML 파일에서 카메라의 내부 파라미터(카메라 행렬)와 왜곡 계수를 로드하여 반환함.
# ============================================================================
def load_camera_calibration(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # YAML 파일 열기 및 파싱
    with open(yaml_path, 'r') as f:  # 파일을 읽기 모드로 엶
        calib_data = yaml.safe_load(f)  # 카메라 캘리브레이션 데이터를 파싱하여 저장
    # 카메라 행렬 데이터 추출 및 numpy 배열로 변환
    cam_mat_data = calib_data['camera_matrix']['data']  # camera_matrix의 data 항목 추출
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)  # numpy 배열로 변환
    # 왜곡 계수 데이터 추출 및 numpy 배열로 변환, 배열 형태 변경
    dist_data = calib_data['distortion_coefficients']['data']  # distortion_coefficients의 data 항목 추출
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))  # 1행 다열 배열로 reshape
    return camera_matrix, dist_coeffs  # 카메라 행렬과 왜곡 계수 반환


# ============================================================================
# 클래스: YoloLidarFusion
# 설명: YOLO 감지 결과와 LiDAR 센서 데이터를 헝가리안 알고리즘을 사용해 융합(매칭)하는 ROS2 노드를 정의함.
# ============================================================================
class YoloLidarFusion(Node):
    # 노드 초기화 함수
    def __init__(self):
        # ROS2 Node 클래스 초기화 (노드 이름: 'hungarian_association_node')
        super().__init__('hungarian_association_node')  # 상위 클래스(Node) 초기화
        
        # 헝가리안 알고리즘 설정을 위한 구성 파일 로드
        self.config = load_hungarian_config()  # 설정을 로드함
        if self.config is None:  # 설정 로드 실패 시
            self.get_logger().error("Failed to load hungarian_association configuration.")  # 에러 메시지 출력
            return  # 초기화 중단
            
        # 설정 파일 내에서 'hungarian_association' 관련 파라미터 가져오기 (기본값 포함)
        hungarian_config = self.config.get('hungarian_association', {})  # 'hungarian_association' 키의 설정을 가져옴
        
        # ROS2 파라미터 선언: cone_z_offset (라이다 포인트의 z 좌표 오프셋)
        self.declare_parameter('cone_z_offset', hungarian_config.get('cone_z_offset', -0.6))  # 파라미터 선언 (기본값: -0.6)
        # 파라미터 값 읽기
        self.cone_z_offset = self.get_parameter('cone_z_offset').value  # 파라미터 값 저장
        self.get_logger().info(f"Using cone z offset: {self.cone_z_offset} meters")  # 설정값 로그 출력
        
        # 최대 매칭 거리 설정 (비용 행렬 계산 시 임계값)
        self.max_matching_distance = hungarian_config.get('max_matching_distance', 5.0)  # 최대 매칭 거리 설정 (기본값: 5.0)
        self.get_logger().info(f"Max matching distance: {self.max_matching_distance}")  # 설정값 로그 출력

        # 파라미터 변경 콜백 함수 등록 (파라미터 업데이트 시 호출)
        self.add_on_set_parameters_callback(self.parameters_callback)  # 파라미터 변경 감지 콜백 등록
        
        # 캘리브레이션 파일 경로를 설정 파일에서 읽어오기
        calib_config = hungarian_config.get('calibration', {})  # 캘리브레이션 관련 설정 읽기
        config_folder = calib_config.get('config_folder', '')  # 설정 폴더 경로
        extrinsic_file = calib_config.get('camera_extrinsic_calibration', '')  # extrinsic 캘리브레이션 파일명
        intrinsic_file = calib_config.get('camera_intrinsic_calibration', '')  # intrinsic 캘리브레이션 파일명
        
        # extrinsic 캘리브레이션 파일 경로 구성 및 로드
        extrinsic_yaml = os.path.join(config_folder, extrinsic_file)  # 파일 경로 생성
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)  # extrinsic matrix 로드

        # intrinsic 캘리브레이션 파일 경로 구성 및 로드
        camera_yaml = os.path.join(config_folder, intrinsic_file)  # 파일 경로 생성
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)  # 카메라 행렬 및 왜곡 계수 로드

        # 로드된 캘리브레이션 값들을 로그로 출력
        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))  # extrinsic matrix 출력
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))  # 카메라 행렬 출력
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))  # 왜곡 계수 출력

        # 설정 파일에서 토픽 이름들을 읽어오기
        cones_topic = hungarian_config.get('cones_topic', "/sorted_cones_time")  # 라이다(콘) 데이터 토픽
        boxes_topic = hungarian_config.get('boxes_topic', "/detections")  # YOLO 감지 토픽
        output_topic = hungarian_config.get('output_topic', "/fused_sorted_cones")  # 융합 결과 토픽
        
        # 구독할 토픽 이름들을 로그로 출력
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")  # cones 토픽 로그
        self.get_logger().info(f"Subscribing to boxes topic: {boxes_topic}")  # boxes 토픽 로그

        # QoS (Quality of Service) 설정 읽기
        qos_config = hungarian_config.get('qos', {})  # QoS 관련 설정 읽기
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT 신뢰성 설정
            history=HistoryPolicy.KEEP_LAST,  # 최근 메시지만 유지
            depth=qos_config.get('history_depth', 1)  # history depth (기본값: 1)
        )

        # message_filters의 Subscriber를 사용하여 cones 토픽 구독 (동기화를 위해)
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)  # cones 데이터 구독자 생성
        # message_filters의 Subscriber를 사용하여 YOLO 감지 토픽 구독
        self.boxes_sub = Subscriber(self, DetectionArray, boxes_topic, qos_profile=best_effort_qos)  # YOLO 데이터 구독자 생성

        # ApproximateTimeSynchronizer를 사용해 두 토픽의 메시지를 시간 동기화
        self.ats = ApproximateTimeSynchronizer(
            [self.cones_sub, self.boxes_sub],  # 동기화할 두 구독자 리스트
            queue_size=qos_config.get('sync_queue_size', 10),  # 큐 크기 (기본값: 10)
            slop=qos_config.get('sync_slop', 0.1)  # 허용 시간 오차 (기본값: 0.1초)
        )
        
        # 동기화된 메시지가 도착하면 hungarian_callback 함수 호출
        self.ats.registerCallback(self.hungarian_callback)  # 콜백 함수 등록

        # 시각화를 위한 색상 매핑 (BGR 포맷)
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # 크림슨 색: 빨간색
            "Yellow Cone":  (0, 255, 255), # 옐로우: 노란색
            "Blue Cone":    (255, 0, 0),   # 블루: 파란색
            "Unknown":      (0, 255, 0)    # Unknown: 초록색 (기본값)
        }

        # 융합된 좌표를 퍼블리시할 퍼블리셔 생성
        self.coord_pub = self.create_publisher(
            ModifiedFloat32MultiArray,  # 퍼블리시할 메시지 타입
            output_topic,  # 퍼블리시할 토픽 이름
            qos_profile=best_effort_qos  # QoS 설정
        )

        # 노드 초기화 완료 로그 출력
        self.get_logger().info('YoloLidarFusion node initialized')

    # ============================================================================
    # 함수: parameters_callback
    # 설명: 파라미터가 변경될 때 호출되어, cone_z_offset 값을 업데이트함.
    # ============================================================================
    def parameters_callback(self, params):
        # 변경된 모든 파라미터에 대해 반복
        for param in params:  # 각 파라미터를 순회
            # 파라미터 이름이 'cone_z_offset'이면
            if param.name == 'cone_z_offset':  
                self.cone_z_offset = param.value  # cone_z_offset 값을 업데이트
                self.get_logger().info(f"Updated cone z offset: {self.cone_z_offset} meters")  # 업데이트된 값 로그 출력

    # ============================================================================
    # 함수: convert_yolo_msg_to_array
    # 설명: DetectionArray 메시지(YOLO 감지 결과)를 numpy 배열로 변환함.
    #         각 바운딩 박스는 [center_x, center_y, size_x, size_y] 형식임.
    # ============================================================================
    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg):
        # 빈 리스트 생성: 바운딩 박스 데이터를 저장할 리스트
        boxes = []  
        # 메시지 내의 각 detection (감지 결과)에 대해 반복
        for detection in yolo_msg.detections:  
            # 바운딩 박스 중심 좌표와 크기를 리스트로 추가
            boxes.append([
                detection.bbox.center.position.x,  # 바운딩 박스 중심의 x 좌표
                detection.bbox.center.position.y,  # 바운딩 박스 중심의 y 좌표
                detection.bbox.size.x,             # 바운딩 박스 너비
                detection.bbox.size.y              # 바운딩 박스 높이
            ])
        # 리스트를 numpy 배열로 변환하여 반환
        return np.array(boxes)

    # ============================================================================
    # 함수: convert_cone_msg_to_array
    # 설명: ModifiedFloat32MultiArray 메시지(라이다 콘 데이터)를 numpy 배열로 변환하고,
    #         LiDAR 좌표를 카메라 좌표계로 변환 후 이미지 평면으로 투영함.
    # ============================================================================
    def convert_cone_msg_to_array(self, cone_msg):
        # 메시지의 data 필드를 numpy 배열로 변환 (float32 타입)
        cone_data = np.array(cone_msg.data, dtype=np.float32)  
        
        # 데이터가 없으면 경고 로그를 출력하고 빈 배열 반환
        if cone_data.size == 0:
            self.get_logger().warn("Empty cones data.")  # 데이터가 없음을 경고
            return np.array([]), np.array([])  # 빈 배열 두 개 반환
        
        # layout 정보를 통해 포인트 개수를 얻음
        num_points = cone_msg.layout.dim[0].size  # 첫 번째 dimension의 사이즈가 포인트 개수임
        # 데이터의 크기가 (포인트 수 * 2)와 맞지 않으면 오류 로그 출력
        if num_points * 2 != cone_data.size:
            self.get_logger().error(f"Cone data size ({cone_data.size}) does not match layout dimensions ({num_points}*2).")
            return np.array([]), np.array([])  # 오류 시 빈 배열 반환
        
        # 데이터를 (N, 2) 형태로 재구성 (각 포인트의 x, y 좌표)
        cones_xy = cone_data.reshape(num_points, 2)  
        
        # 각 포인트에 cone_z_offset 값을 z 좌표로 추가하여 (N, 3) 배열 생성
        cones_xyz = np.hstack((cones_xy, np.ones((num_points, 1), dtype=np.float32) * self.cone_z_offset))
        
        # 동차 좌표로 변환하기 위해 1을 추가하여 (N, 4) 배열 생성
        cones_xyz_h = np.hstack((cones_xyz, np.ones((cones_xyz.shape[0], 1), dtype=np.float32)))
        
        # LiDAR 좌표계를 카메라 좌표계로 변환하기 위해 extrinsic matrix 적용 (행렬 곱)
        cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T  
        # 동차 좌표에서 3D 좌표만 추출 (x, y, z)
        cones_cam = cones_cam_h[:, :3]  
        
        # 원본 인덱스를 유지하기 위해 0부터 num_points-1까지 배열 생성
        original_indices = np.arange(num_points)  
        
        # 포인트가 존재하면 이미지 평면으로 투영
        if num_points > 0:
            # 회전 벡터와 병진 벡터를 0으로 초기화 (이미 카메라 좌표계에 있으므로)
            rvec = np.zeros((3,1), dtype=np.float64)  
            tvec = np.zeros((3,1), dtype=np.float64)  
            # cv2.projectPoints() 함수를 사용해 3D 포인트를 이미지 평면의 2D 좌표로 투영
            cone_image_points, _ = cv2.projectPoints(
                cones_cam.astype(np.float64),  # 3D 포인트 (float64 타입)
                rvec,  # 회전 벡터
                tvec,  # 병진 벡터
                self.camera_matrix,  # 카메라 내부 파라미터
                self.dist_coeffs  # 왜곡 계수
            )
            # 결과를 (N, 2) 형태로 재구성
            cone_image_points = cone_image_points.reshape(-1, 2)  
            
            # 투영된 포인트 개수를 디버그 로그로 출력
            self.get_logger().debug(f"Projected {len(cone_image_points)} cones to image plane")
            
            # 투영된 2D 좌표와 원본 인덱스 배열을 반환
            return cone_image_points, original_indices  
        
        # 포인트가 없는 경우 빈 배열 두 개 반환
        return np.array([]), np.array([])

    # ============================================================================
    # 함수: compute_cost_matrix
    # 설명: YOLO 바운딩 박스와 투영된 LiDAR 포인트 사이의 유클리드 거리를 계산하여 비용 행렬을 생성함.
    #       최대 매칭 거리를 초과하면 비용을 매우 큰 값(1e6)으로 설정.
    # ============================================================================
    def compute_cost_matrix(self, yolo_bboxes, cone_points):
        # YOLO 바운딩 박스 수와 투영된 LiDAR 포인트 수를 계산
        num_boxes = yolo_bboxes.shape[0]  # YOLO 감지된 바운딩 박스 개수
        num_cones = cone_points.shape[0]  # 투영된 LiDAR 포인트 개수
        # 비용 행렬을 0으로 초기화 (크기: num_boxes x num_cones)
        cost_matrix = np.zeros((num_boxes, num_cones))
        
        # 모든 YOLO 바운딩 박스와 LiDAR 포인트 쌍에 대해 비용(유클리드 거리) 계산
        for i in range(num_boxes):
            # 각 YOLO 바운딩 박스의 중심 좌표 계산
            center_x = yolo_bboxes[i, 0]  # 바운딩 박스 중심 x 좌표
            center_y = yolo_bboxes[i, 1]  # 바운딩 박스 중심 y 좌표
            for j in range(num_cones):
                # 두 점 사이의 유클리드 거리 계산
                distance = np.linalg.norm([
                    center_x - cone_points[j, 0],  # x 좌표 차이
                    center_y - cone_points[j, 1]   # y 좌표 차이
                ])
                # 최대 매칭 거리보다 작으면 거리를 비용으로, 그렇지 않으면 매우 큰 비용(1e6) 설정
                cost_matrix[i, j] = distance if distance < self.max_matching_distance else 1e6
        
        # 비용 행렬을 정방행렬로 만들기 위해 패딩 (행 또는 열 추가)
        if num_boxes < num_cones:
            # YOLO 박스 수가 적으면, dummy 행(비용을 0.0으로 설정)을 추가하여 행 수를 맞춤
            dummy_rows = np.full((num_cones - num_boxes, num_cones), 0.0)
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        elif num_boxes > num_cones:
            # YOLO 박스 수가 많으면, dummy 열(비용을 0.0으로 설정)을 추가하여 열 수를 맞춤
            dummy_cols = np.full((num_boxes, num_boxes - num_cones), 0.0)
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
        
        # 계산된 비용 행렬 반환
        return cost_matrix

    # ============================================================================
    # 함수: hungarian_callback
    # 설명: 동기화된 YOLO 감지 결과와 LiDAR 콘 데이터를 받아 헝가리안 알고리즘을 적용하여 매칭 후,
    #         매칭 결과(클래스 정보)를 업데이트한 메시지를 퍼블리시함.
    # ============================================================================
    def hungarian_callback(self, cone_msg, yolo_msg):
        try:
            # YOLO 감지 결과 메시지를 numpy 배열로 변환 (바운딩 박스 중심 및 크기)
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            
            # LiDAR 콘 메시지를 2D 이미지 평면 좌표로 투영하고, 원본 인덱스 배열도 반환
            cone_image_points, original_indices = self.convert_cone_msg_to_array(cone_msg)
            
            # YOLO 감지나 LiDAR 투영 결과가 없으면 경고를 출력하고 원본 메시지 그대로 퍼블리시
            if len(yolo_bboxes) == 0 or len(cone_image_points) == 0:
                self.get_logger().warn('ZERO detections in one or both sensors')
                self.coord_pub.publish(cone_msg)  # 원본 cones 메시지를 퍼블리시
                return  # 함수 종료
            
            # YOLO 바운딩 박스와 투영된 LiDAR 포인트 사이의 비용 행렬 계산
            cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_image_points)
            # 헝가리안 알고리즘을 적용하여 최적의 매칭 인덱스 (행, 열 인덱스) 계산
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 매칭 결과(행 인덱스, 열 인덱스)를 디버그 로그로 출력
            self.get_logger().debug(f"Hungarian matching results - row indices: {row_ind}, col indices: {col_ind}")
            
            # 매칭 결과를 담을 새 메시지 생성 (원본 cones 메시지를 기반으로 함)
            matched_msg = ModifiedFloat32MultiArray()
            matched_msg.header = cone_msg.header  # 헤더 복사
            matched_msg.layout = cone_msg.layout  # layout 복사
            matched_msg.data = list(cone_msg.data)  # 원본 data를 복사하여 리스트로 변환
            
            # 매칭 전, 모든 포인트의 클래스명을 "Unknown"으로 초기화 (나중에 업데이트 예정)
            num_cones = matched_msg.layout.dim[0].size  # 전체 LiDAR 포인트 개수
            matched_msg.class_names = ["Unknown"] * num_cones  # 각 포인트의 클래스명을 "Unknown"으로 설정
            
            # 디버깅을 위해 각 포인트가 어떤 YOLO 바운딩 박스와 매칭되었는지 추적할 배열 초기화 (초기값 -1)
            matched_indices = [-1] * num_cones  
            
            # 유효한 매칭 개수를 세기 위한 변수 초기화
            valid_matches = 0  
            # 헝가리안 알고리즘의 매칭 결과를 순회
            for i, j in zip(row_ind, col_ind):
                # 만약 매칭 비용이 최대 매칭 거리보다 크거나, 인덱스가 범위를 벗어나면 스킵
                if (i >= len(yolo_bboxes) or j >= len(cone_image_points) or 
                    cost_matrix[i, j] >= self.max_matching_distance):
                    continue  # 조건에 맞지 않으면 해당 매칭 건은 무시
                
                # 현재 매칭된 LiDAR 포인트의 원본 인덱스 추출
                original_idx = original_indices[j]
                
                # 디버깅을 위해 해당 LiDAR 포인트의 매칭된 YOLO 인덱스 저장
                matched_indices[original_idx] = i
                
                # YOLO 감지 결과에서 클래스명을 가져옴
                yolo_class = yolo_msg.detections[i].class_name
                
                # 원본 인덱스가 유효하면, 해당 인덱스의 클래스명을 YOLO 감지 결과의 클래스명으로 업데이트
                if original_idx < num_cones:
                    matched_msg.class_names[original_idx] = yolo_class
                    valid_matches += 1  # 유효 매칭 개수 증가
                    
                    # 매칭된 결과에 대한 상세 정보를 디버그 로그로 출력
                    self.get_logger().debug(
                        f"Match: YOLO idx={i}, img point idx={j}, original lidar idx={original_idx}, "
                        f"class={yolo_class}, cost={cost_matrix[i, j]:.2f}, "
                        f"image pos=({cone_image_points[j][0]:.1f}, {cone_image_points[j][1]:.1f}), "
                        f"bbox center=({yolo_bboxes[i][0]:.1f}, {yolo_bboxes[i][1]:.1f})"
                    )
            
            # 매칭된 인덱스와 최종 클래스 할당 결과를 디버그 로그로 출력 (문제 진단 용도)
            self.get_logger().debug(f"Matched indices (lidar_idx -> yolo_idx): {list(enumerate(matched_indices))}")
            self.get_logger().debug(f"Final class assignments: {list(enumerate(matched_msg.class_names))}")
            
            # 매칭 결과 요약을 로그에 출력
            self.get_logger().info(
                f'Matched {valid_matches} cones out of {len(yolo_bboxes)} YOLO detections, '
                f'{len(cone_image_points)} projected LiDAR points, and {num_cones} total LiDAR points'
            )
            
            # 최종적으로 매칭된 결과 메시지를 퍼블리시
            self.coord_pub.publish(matched_msg)
            
        # 예외가 발생할 경우 에러 로그와 traceback을 출력
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')
            import traceback  # traceback 모듈 임포트 (에러 추적용)
            self.get_logger().error(traceback.format_exc())  # 전체 traceback 출력

# ============================================================================
# 함수: main
# 설명: ROS2 노드를 초기화하고, YoloLidarFusion 노드를 실행하는 메인 함수.
# ============================================================================
def main(args=None):
    rclpy.init(args=args)  # ROS2 클라이언트 라이브러리 초기화
    hungarian_association_node = YoloLidarFusion()  # YoloLidarFusion 노드 인스턴스 생성
    try:
        rclpy.spin(hungarian_association_node)  # 노드가 종료될 때까지 스핀(메시지 처리 루프)
    except KeyboardInterrupt:  # 키보드 인터럽트(CTRL+C) 발생 시
        pass  # 예외 무시
    finally:
        hungarian_association_node.destroy_node()  # 노드 소멸 (자원 해제)
        rclpy.shutdown()  # ROS2 클라이언트 종료

# 스크립트가 메인 프로그램으로 실행될 때 main 함수 호출
if __name__ == '__main__':
    main()

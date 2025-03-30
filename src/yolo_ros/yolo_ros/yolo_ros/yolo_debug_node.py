#!/usr/bin/env python3                              
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from yolo_msgs.msg import BoundingBox2D, Detection, DetectionArray 
from cv_bridge import CvBridge                      # ROS와 OpenCV 이미지 변환 모듈 임포트
from ultralytics import YOLO                        # YOLO 모델을 제공하는 ultralytics 모듈 임포트

class YoloDebugNode(Node):                           # YoloDebugNode 클래스를 정의 (ROS2 Node 상속)
    def __init__(self):                              # 생성자 함수
        super().__init__("yolo_debug_node")          # 부모 Node 생성자 호출, 노드 이름을 "yolo_debug_node"로 설정
        # 파라미터 선언
        self.declare_parameter("device", "cuda:0")   # 사용할 디바이스 (GPU) 파라미터 선언 (기본값: "cuda:0")
        self.declare_parameter("threshold", 0.5)       # 감지 신뢰도 임계값 파라미터 선언 (기본값: 0.5)
        self.declare_parameter("iou", 0.5)             # IoU 임계값 파라미터 선언 (기본값: 0.5)
        self.declare_parameter("max_det", 100)         # 최대 검출 수 파라미터 선언 (기본값: 100)
        self.declare_parameter("imgsz_height", 360)    # 입력 이미지 높이 파라미터 선언 (기본값: 360)
        self.declare_parameter("imgsz_width", 640)     # 입력 이미지 너비 파라미터 선언 (기본값: 640)
        self.declare_parameter("image_topic", "/image_raw")  # 구독할 이미지 토픽 파라미터 선언 (기본값: "/image_raw")

        # 파라미터 값 가져오기
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value   # threshold 값 가져오기
        self.iou = self.get_parameter("iou").get_parameter_value().double_value               # iou 값 가져오기
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value      # max_det 값 가져오기
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value  # imgsz_height 값 가져오기
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value    # imgsz_width 값 가져오기
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value     # image_topic 값 가져오기

        # YOLO 모델 로드 (모델 경로는 환경에 맞게 수정)
        model_path = "/home/wonsuk1025/kai_ws/src/cam_yolo/runs/detect/train4/weights/best.pt" # 모델 파일 경로 지정
        self.get_logger().info(f"YOLO 모델 로드 중: {model_path}") # 모델 로드 시작 로그 출력
        self.model = YOLO(model_path) # YOLO 모델 로드

        # 클래스별 색상 정의 (BGR 형식)
        self.class_colors = { # 각 클래스에 대해 BGR 색상 정의
            'blue cone': (255, 0, 0),    # 'blue cone'은 파란색
            'red cone': (0, 0, 255),     # 'red cone'은 빨간색
            'yellow cone': (0, 255, 255) # 'yellow cone'은 노란색
        }

        # 퍼블리셔 생성
        self._detection_pub = self.create_publisher(DetectionArray, "detections", 10) # "detections" 토픽에 DetectionArray 메시지 퍼블리셔 생성 (큐 사이즈 10)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10) # "dbg_image" 토픽에 Image 메시지 퍼블리셔 생성 (큐 사이즈 10)
        self._info_pub = self.create_publisher(String, "cone_info", 10) # "cone_info" 토픽에 String 메시지 퍼블리셔 생성 (큐 사이즈 10)

        # 이미지 토픽 구독 (QoS 프로파일 간단 설정)
        qos_profile = QoSProfile(                         # QoS 프로파일 생성
            reliability=QoSReliabilityPolicy.BEST_EFFORT, # 신뢰성: BEST_EFFORT
            history=QoSHistoryPolicy.KEEP_LAST,           # 히스토리: 최신 메시지 유지
            depth=1,                                      # 큐 깊이: 1
            durability=QoSDurabilityPolicy.VOLATILE,      # 내구성: VOLATILE
        )
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile) # 이미지 토픽 구독, 콜백 함수: image_cb, QoS 적용
        self.get_logger().info(f"이미지 토픽 구독 시작: {self.image_topic}")              # 구독 시작 로그 출력

        # cv_bridge 초기화
        self.cv_bridge = CvBridge()           # cv_bridge 객체 생성 (ROS 이미지와 OpenCV 이미지 간 변환)
        self.get_logger().info("노드 설정 완료") # 노드 설정 완료 로그 출력

    def image_cb(self, msg: Image): # 이미지 콜백 함수 정의 (Image 메시지 처리)
        try:
            # ROS 이미지 메시지를 OpenCV의 BGR 이미지로 변환 (색 변환 없이 그대로 사용)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") # ROS Image 메시지를 BGR 형식의 OpenCV 이미지로 변환
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}") # 변환 실패 시 에러 로그 출력
            return # 함수 종료

        # 이미지 크기를 파라미터에 맞게 조정
        cv_image = cv2.resize(cv_image, (self.imgsz_width, self.imgsz_height)) # OpenCV 이미지를 지정된 크기로 리사이즈
        # 여기서는 BGR 이미지를 그대로 모델에 전달
        try:
            # YOLO 추론 수행 (BGR 이미지 그대로 사용)
            results = self.model(cv_image)[0].cpu() # YOLO 모델에 BGR 이미지를 입력하여 추론 수행, 결과의 첫 번째 항목을 CPU로 이동
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 오류: {e}") # 추론 오류 발생 시 에러 로그 출력
            return # 함수 종료

        # DetectionArray 메시지 생성 및 헤더 설정
        detection_array = DetectionArray() # DetectionArray 메시지 객체 생성
        detection_array.header = msg.header  # 수신된 이미지 메시지의 헤더를 복사

        cone_info_list = [] # 콘 좌표 정보를 저장할 문자열 리스트 초기화

        # 2번 코드와 같이 xywh 좌표를 사용하여 객체 정보를 추출
        if results.boxes:                                        # 추론 결과에 박스가 존재하면
            for box in results.boxes:                            # 각 박스에 대해 반복
                # xywh 좌표: [x_center, y_center, w, h]
                box_xywh = box.xywh[0].cpu().numpy()             # 박스의 xywh 좌표를 NumPy 배열로 변환
                x_center, y_center, w, h = box_xywh.astype(int)  # 중심 좌표와 너비, 높이를 정수형으로 변환

                detection = Detection()                          # Detection 메시지 객체 생성
                class_idx = int(box.cls[0].cpu().numpy())        # 박스의 클래스 인덱스 추출 (정수형)
                detection.class_id = class_idx                   # Detection 메시지에 클래스 인덱스 설정

                # 모델에서 나온 클래스 이름 그대로 사용
                yolo_label = results.names[class_idx]              # 클래스 이름 추출
                detection.class_name = yolo_label                  # Detection 메시지에 클래스 이름 설정
                detection.score = float(box.conf[0].cpu().numpy()) # Detection 메시지에 신뢰도(score) 설정

                # BoundingBox2D 메시지 생성
                bbox = BoundingBox2D()                    # BoundingBox2D 객체 생성
                bbox.center.position.x = float(x_center)  # 중심 x 좌표 설정
                bbox.center.position.y = float(y_center)  # 중심 y 좌표 설정
                bbox.size.x = float(w)                    # 바운딩 박스 너비 설정
                bbox.size.y = float(h)                    # 바운딩 박스 높이 설정
                detection.bbox = bbox                     # Detection 메시지에 bbox 할당

                detection_array.detections.append(detection)                     # Detection을 DetectionArray에 추가
                cone_info_list.append(f"{yolo_label}: ({x_center}, {y_center})") # 콘 정보 문자열 리스트에 추가

        # detections 토픽에 DetectionArray 메시지 퍼블리시
        self._detection_pub.publish(detection_array) # DetectionArray 메시지를 "detections" 토픽에 퍼블리시

        # cone_info 토픽에 콘 정보 문자열 퍼블리시
        info_msg = String()                       # String 메시지 객체 생성
        info_msg.data = "; ".join(cone_info_list) # 리스트를 세미콜론으로 구분한 문자열로 변환하여 설정
        self._info_pub.publish(info_msg)          # "cone_info" 토픽에 String 메시지 퍼블리시

        # 디버그용 이미지에 바운딩 박스 및 라벨 그리기 (rviz2에서 dbg_image 토픽으로 시각화)
        debug_image = cv_image.copy()                  # 원본 이미지를 복사하여 디버그 이미지 생성
        for detection in detection_array.detections:   # DetectionArray 내의 각 Detection에 대해 반복
            cx = int(detection.bbox.center.position.x) # Detection의 중심 x 좌표 추출
            cy = int(detection.bbox.center.position.y) # Detection의 중심 y 좌표 추출
            w = int(detection.bbox.size.x)             # Detection의 너비 추출
            h = int(detection.bbox.size.y)             # Detection의 높이 추출
            x1 = int(cx - w / 2)                       # 좌측 상단 x 좌표 계산
            y1 = int(cy - h / 2)                       # 좌측 상단 y 좌표 계산
            x2 = x1 + w                                # 우측 하단 x 좌표 계산
            y2 = y1 + h                                # 우측 하단 y 좌표 계산
            # 클래스별 색상 적용 (정의된 색상이 없으면 흰색 사용)
            color = self.class_colors.get(detection.class_name, (255, 255, 255)) # 해당 클래스에 맞는 색상 선택, 없으면 흰색
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)             # 바운딩 박스 사각형 그리기
            label = f"{detection.class_name} {detection.score:.2f}"              # 라벨 문자열 생성 (클래스 이름과 신뢰도)
            cv2.putText(debug_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)                 # 라벨 텍스트를 이미지에 그리기

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8") # 디버그 이미지를 ROS Image 메시지로 변환 (BGR8 인코딩)
        debug_msg.header = msg.header    # 디버그 이미지 메시지의 헤더를 원본 메시지 헤더로 설정
        self._dbg_pub.publish(debug_msg) # "dbg_image" 토픽에 디버그 이미지 퍼블리시

def main(args=None): # 메인 함수 정의
    rclpy.init(args=args) # ROS2 초기화
    node = YoloDebugNode() # YoloDebugNode 객체 생성
    try:
        rclpy.spin(node) # 노드를 스핀하여 콜백 함수 실행
    except KeyboardInterrupt:
        node.get_logger().info("노드 종료 중...") # 인터럽트 발생 시 종료 로그 출력
    finally:
        node.destroy_node() # 노드 자원 해제
        rclpy.shutdown() # ROS2 종료

if __name__ == "__main__": # 스크립트가 직접 실행될 때
    main() # 메인 함수 호출

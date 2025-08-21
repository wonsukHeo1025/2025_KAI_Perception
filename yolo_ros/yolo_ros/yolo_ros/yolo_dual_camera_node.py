#!/usr/bin/env python3

import cv2
import numpy as np
import os
from pathlib import Path
import rclpy
from rclpy.node import Node
import threading
from queue import Queue
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from yolo_msgs.msg import BoundingBox2D, Detection, DetectionArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

class YoloCameraProcessor:
    def __init__(self, name, model, threshold, iou, max_det, bridge, node, input_topic, 
                detection_topic, debug_topic, info_topic):
        self.name = name
        self.model = model
        self.threshold = threshold
        self.iou = iou
        self.max_det = max_det
        self.bridge = bridge
        self.node = node
        
        # 클래스별 색상 정의 (BGR 형식)
        self.class_colors = {
            'blue cone': (255, 0, 0),    # 파란색
            'red cone': (0, 0, 255),     # 빨간색
            'yellow cone': (0, 255, 255) # 노란색
        }
        
        # 퍼블리셔 설정
        self.detection_pub = node.create_publisher(DetectionArray, detection_topic, 10)
        self.debug_pub = node.create_publisher(Image, debug_topic, 10)
        self.info_pub = node.create_publisher(String, info_topic, 10)
        
        # 이미지 큐
        self.image_queue = Queue(maxsize=5)
        
        # 이미지 구독
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            qos_profile
        )
        node.get_logger().info(f"{self.name}: 이미지 토픽 구독 시작: {input_topic}")
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self.process_data, daemon=True)
        self.processing_thread.start()
    
    def image_callback(self, msg: Image):
        # 큐가 가득 차 있으면 오래된 항목 제거
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except:
                pass
        
        # 새 이미지 큐에 추가
        try:
            self.image_queue.put_nowait(msg)
        except:
            pass
    
    def process_data(self):
        while rclpy.ok():
            if not self.image_queue.empty():
                try:
                    # 이미지 가져오기
                    msg = self.image_queue.get()
                    self.process_image(msg)
                except Exception as e:
                    self.node.get_logger().error(f"{self.name} 처리 오류: {str(e)}")
                    import traceback
                    self.node.get_logger().error(traceback.format_exc())
            
            # 스레드 부하 감소
            import time
            time.sleep(0.01)
    
    def process_image(self, msg: Image):
        try:
            # ROS 이미지 메시지를 OpenCV의 BGR 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.node.get_logger().error(f"{self.name} 이미지 변환 실패: {e}")
            return

        # 이미지 크기가 정의되어 있다면 적용
        if hasattr(self, 'imgsz_width') and hasattr(self, 'imgsz_height'):
            cv_image = cv2.resize(cv_image, (self.imgsz_width, self.imgsz_height))
            
        try:
            # YOLO 추론 수행
            results = self.model(cv_image)[0].cpu()
        except Exception as e:
            self.node.get_logger().error(f"{self.name} YOLO 추론 오류: {e}")
            return

        # DetectionArray 메시지 생성 및 헤더 설정
        detection_array = DetectionArray()
        detection_array.header = msg.header

        cone_info_list = []  # 콘 좌표 정보를 문자열로 저장

        # 객체 정보 추출
        if results.boxes:
            for box in results.boxes:
                # xywh 좌표: [x_center, y_center, w, h]
                box_xywh = box.xywh[0].cpu().numpy()
                x_center, y_center, w, h = box_xywh.astype(int)

                detection = Detection()
                class_idx = int(box.cls[0].cpu().numpy())
                detection.class_id = class_idx

                # 모델에서 나온 클래스 이름 사용
                yolo_label = results.names[class_idx]
                detection.class_name = yolo_label
                detection.score = float(box.conf[0].cpu().numpy())

                # 임계값 이상인 것만 추가
                if detection.score > self.threshold:
                    # BoundingBox2D 메시지 생성
                    bbox = BoundingBox2D()
                    bbox.center.position.x = float(x_center)
                    bbox.center.position.y = float(y_center)
                    bbox.size.x = float(w)
                    bbox.size.y = float(h)
                    detection.bbox = bbox

                    detection_array.detections.append(detection)
                    cone_info_list.append(f"{yolo_label}: ({x_center}, {y_center})")

        # detections 토픽에 DetectionArray 메시지 퍼블리시
        self.detection_pub.publish(detection_array)

        # cone_info 토픽에 콘 정보 문자열 퍼블리시
        info_msg = String()
        info_msg.data = f"{self.name}: " + "; ".join(cone_info_list)
        self.info_pub.publish(info_msg)

        # 디버그용 이미지에 바운딩 박스 및 라벨 그리기
        debug_image = cv_image.copy()
        
        # 카메라 이름 추가
        cv2.putText(debug_image, self.name, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for detection in detection_array.detections:
            cx = int(detection.bbox.center.position.x)
            cy = int(detection.bbox.center.position.y)
            w = int(detection.bbox.size.x)
            h = int(detection.bbox.size.y)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = x1 + w
            y2 = y1 + h
            # 클래스별 색상 적용 (정의된 색상이 없으면 흰색)
            color = self.class_colors.get(detection.class_name, (255, 255, 255))
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.class_name} {detection.score:.2f}"
            cv2.putText(debug_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

class YoloDualCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_dual_camera_node")
        
        # 파라미터 선언
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("imgsz_height", 360)
        self.declare_parameter("imgsz_width", 640)
        
        # 카메라 토픽 설정
        self.declare_parameter("camera1_image_topic", "/usb_cam_1/image_raw")
        self.declare_parameter("camera2_image_topic", "/usb_cam_2/image_raw")
        
        # 파라미터 값 가져오기
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        
        self.camera1_image_topic = self.get_parameter("camera1_image_topic").get_parameter_value().string_value
        self.camera2_image_topic = self.get_parameter("camera2_image_topic").get_parameter_value().string_value

        # ========================================================================
        # YOLO 모델 로드 - get_package_share_directory 사용
        # ========================================================================
        try:
            # 패키지 설치 경로에서 모델 파일 찾기
            package_share_dir = get_package_share_directory('yolo_ros')
            model_path = os.path.join(package_share_dir, 'models', 'best.pt')
        except Exception as e:
            self.get_logger().warning(f"패키지 경로에서 모델을 찾을 수 없습니다: {e}")
            # 환경변수 fallback
            model_path = os.environ.get('YOLO_MODEL_PATH')
            if not model_path:
                # 소스 디렉토리 fallback - 상대 경로 사용
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, '..', 'models', 'best.pt')
                if not os.path.exists(model_path):
                    # 파라미터로 전달된 경로 시도
                    model_path = self.get_parameter("model").get_parameter_value().string_value
                    if not model_path:
                        # 최종 fallback
                        model_path = "models/best.pt"
        
        # 모델 파일 존재 여부 확인
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
            self.get_logger().error(f"❌ 다음 위치 중 하나에 best.pt 파일을 배치하세요:")
            self.get_logger().error(f"   1. 패키지 설치: share/yolo_ros/models/best.pt")
            self.get_logger().error(f"   2. 환경 변수: YOLO_MODEL_PATH")
            self.get_logger().error(f"   3. 소스 디렉토리: src/yolo_ros/yolo_ros/models/best.pt")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.get_logger().info(f"✅ YOLO 모델 로드 중: {model_path}")
        self.model = YOLO(model_path)
        # ========================================================================

        # cv_bridge 초기화
        self.cv_bridge = CvBridge()
        
        # 카메라 프로세서 초기화
        self.camera_processors = {}
        
        # 카메라 1 프로세서 설정
        self.camera_processors['camera_1'] = YoloCameraProcessor(
            name='Camera 1',
            model=self.model,
            threshold=self.threshold,
            iou=self.iou,
            max_det=self.max_det,
            bridge=self.cv_bridge,
            node=self,
            input_topic=self.camera1_image_topic,
            detection_topic="/camera_1/detections",
            debug_topic="/camera_1/dbg_image",
            info_topic="/camera_1/cone_info"
        )
        
        # 이미지 크기 설정 (선택적)
        self.camera_processors['camera_1'].imgsz_width = self.imgsz_width
        self.camera_processors['camera_1'].imgsz_height = self.imgsz_height
        
        # 카메라 2 프로세서 설정
        self.camera_processors['camera_2'] = YoloCameraProcessor(
            name='Camera 2',
            model=self.model,
            threshold=self.threshold,
            iou=self.iou,
            max_det=self.max_det,
            bridge=self.cv_bridge,
            node=self,
            input_topic=self.camera2_image_topic,
            detection_topic="/camera_2/detections",
            debug_topic="/camera_2/dbg_image",
            info_topic="/camera_2/cone_info"
        )
        
        # 이미지 크기 설정 (선택적)
        self.camera_processors['camera_2'].imgsz_width = self.imgsz_width
        self.camera_processors['camera_2'].imgsz_height = self.imgsz_height
        
        self.get_logger().info("듀얼 카메라 YOLO 노드 설정 완료")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDualCameraNode()
    
    # 멀티스레드 실행자 사용
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=3)  # 메인 + 2개 카메라 처리
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("노드 종료 중...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main() 
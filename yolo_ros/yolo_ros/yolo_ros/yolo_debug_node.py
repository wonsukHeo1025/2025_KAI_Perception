#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from yolo_msgs.msg import BoundingBox2D, Detection, DetectionArray
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloDebugNode(Node):
    def __init__(self):
        super().__init__("yolo_debug_node")
        # 파라미터 선언
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("imgsz_height", 360)
        self.declare_parameter("imgsz_width", 640)
        # 구독할 이미지 토픽 (필요 시 "/image_raw/uncompressed"로 변경 가능)
        self.declare_parameter("image_topic", "/image_raw/uncompressed")

        # 파라미터 값 가져오기
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        # YOLO 모델 로드 (모델 경로는 실제 환경에 맞게 수정)
        model_path = "/home/user1/YOLO/pretrained_models/best.pt"
        self.get_logger().info(f"YOLO 모델 로드 중: {model_path}")
        self.model = YOLO(model_path)

        # 클래스별 색상 정의 (BGR 형식)
        self.class_colors = {
            'blue cone': (255, 0, 0),    # 파란색
            'red cone': (0, 0, 255),     # 빨간색
            'yellow cone': (0, 255, 255) # 노란색
        }

        # 퍼블리셔 생성
        self._detection_pub = self.create_publisher(DetectionArray, "detections", 10)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._info_pub = self.create_publisher(String, "cone_info", 10)

        # 이미지 토픽 구독 (QoS 프로파일 간단 설정)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile)
        self.get_logger().info(f"이미지 토픽 구독 시작: {self.image_topic}")

        # cv_bridge 초기화
        self.cv_bridge = CvBridge()
        self.get_logger().info("노드 설정 완료")

    def image_cb(self, msg: Image):
        try:
            # ROS 이미지 메시지를 OpenCV의 BGR 이미지로 변환 (색 변환 없이 그대로 사용)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        # 이미지 크기를 파라미터에 맞게 조정
        cv_image = cv2.resize(cv_image, (self.imgsz_width, self.imgsz_height))
        # 여기서는 BGR 이미지를 그대로 모델에 전달합니다.
        try:
            # YOLO 추론 수행 (BGR 이미지 그대로 사용)
            results = self.model(cv_image)[0].cpu()
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 오류: {e}")
            return

        # DetectionArray 메시지 생성 및 헤더 설정
        detection_array = DetectionArray()
        detection_array.header = msg.header

        cone_info_list = []  # 콘 좌표 정보를 문자열로 저장

        # 2번 코드와 같이 xywh 좌표를 사용하여 객체 정보를 추출
        if results.boxes:
            for box in results.boxes:
                # xywh 좌표: [x_center, y_center, w, h]
                box_xywh = box.xywh[0].cpu().numpy()
                x_center, y_center, w, h = box_xywh.astype(int)

                detection = Detection()
                class_idx = int(box.cls[0].cpu().numpy())
                detection.class_id = class_idx

                # 모델에서 나온 클래스 이름 그대로 사용
                yolo_label = results.names[class_idx]
                detection.class_name = yolo_label
                detection.score = float(box.conf[0].cpu().numpy())

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
        self._detection_pub.publish(detection_array)

        # cone_info 토픽에 콘 정보 문자열 퍼블리시
        info_msg = String()
        info_msg.data = "; ".join(cone_info_list)
        self._info_pub.publish(info_msg)

        # 디버그용 이미지에 바운딩 박스 및 라벨 그리기 (rviz2에서 dbg_image 토픽으로 시각화)
        debug_image = cv_image.copy()
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

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self._dbg_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("노드 종료 중...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node
from custom_interface.msg import ModifiedFloat32MultiArray
import numpy as np
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ConeVisualizer(Node):
    def __init__(self):
        super().__init__('cone_visualizer')
        
        # QoS 프로파일 설정 - Best Effort 사용
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 올바른 토픽 이름으로 수정 - UKF 결과 구독
        self.subscription = self.create_subscription(
            ModifiedFloat32MultiArray,
            '/fused_sorted_cones_ukf',
            self.cone_callback,
            qos_profile)
        self.subscription  # unused variable 방지

        # OpenCV 창 설정
        self.window_name = "Cone Visualization (UKF Filtered)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.image_width = 800
        # scale을 50 픽셀/미터로 설정하여 더 넓은 영역을 볼 수 있도록 함.
        self.scale = 50  
        self.margin = 50  # 하단 마진
        # 최소 x축 양의 방향 12m(12*50=600픽셀)를 포함하고, 여유를 주어 이미지 높이를 670픽셀로 설정
        self.image_height = int(12 * self.scale + self.margin + 20)  
        self.origin = (self.image_width // 2, self.image_height - self.margin)
        
        # 콘 클래스별 색상 정의 (단순화)
        self.cone_colors = {
            "blue cone": (255, 0, 0),      # BGR 형식 - 파란색
            "red cone": (0, 0, 255),       # BGR 형식 - 빨간색
            "yellow cone": (0, 255, 255),  # BGR 형식 - 노란색
            "unknown": (0, 255, 0)         # BGR 형식 - 초록색 (기본값)
        }

    def cone_callback(self, msg: ModifiedFloat32MultiArray):
        """UKF 필터링된 콘 메시지 처리"""
        try:
            # 콘 개수 확인 (기존 코드와 같은 방식으로 layout 사용)
            if len(msg.layout.dim) > 0:
                num_cones = msg.layout.dim[0].size
                offset = msg.layout.data_offset
            else:
                # layout이 없는 경우 데이터를 기반으로 계산
                num_cones = len(msg.class_names)
                offset = 0
            
            # 데이터 확인
            if len(msg.data) % 2 != 0:
                self.get_logger().error(f"Invalid data length: {len(msg.data)}")
                return
                
            # 데이터 형식: [x1, y1, x2, y2, ...]
            data = np.array(msg.data[offset:])
            cones = data.reshape(-1, 2)
            
            # 혹시 클래스 이름과 콘 개수가 다를 경우 둘 중 작은 값으로 조정
            if len(cones) != len(msg.class_names):
                self.get_logger().warn(f"Data mismatch: {len(cones)} points vs {len(msg.class_names)} classes")
                num_cones = min(len(cones), len(msg.class_names))
            else:
                num_cones = len(cones)

        except Exception as e:
            self.get_logger().error(f"Data processing error: {e}")
            return

        # 흰색 배경 이미지 생성
        img = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)

        # grid를 그리기 위한 world 좌표 범위 설정
        # x: 전방 방향, y: 좌우(양의 방향이 왼쪽)
        x_min, x_max = -2, 12
        y_min, y_max = -8, 8

        # world 좌표 -> 이미지 좌표 변환 함수 (working version과 동일하게 수정)
        def world_to_image(x, y):
            # x: 전방, y: 좌우 (양의 방향이 왼쪽)
            u = int(self.origin[0] + y * self.scale)  # y positive → 왼쪽
            v = int(self.origin[1] + x * self.scale)  # x positive → 위쪽 (전방), 부호 수정
            return (u, v)

        # 1m 간격의 horizontal grid line (x 고정) 그리기
        for x_val in np.arange(x_min, x_max + 1, 1):
            pt1 = world_to_image(x_val, y_min)
            pt2 = world_to_image(x_val, y_max)
            cv2.line(img, pt1, pt2, (220, 220, 220), 1)  # 연한 회색 선
            # x 좌표 레이블 (왼쪽 하단)
            label_pt = world_to_image(x_val, y_min)
            cv2.putText(img, f"{x_val}m", (label_pt[0] - 20, label_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # 1m 간격의 vertical grid line (y 고정) 그리기
        for y_val in np.arange(y_min, y_max + 1, 1):
            pt1 = world_to_image(x_min, y_val)
            pt2 = world_to_image(x_max, y_val)
            cv2.line(img, pt1, pt2, (220, 220, 220), 1)
            # y 좌표 레이블 (좌측 상단)
            label_pt = world_to_image(x_min, y_val)
            cv2.putText(img, f"{y_val}m", (label_pt[0] - 20, label_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # 상단 왼쪽에 cone 개수 표시
        cone_count_text = f"Cones: {num_cones}"
        cv2.putText(img, cone_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # 내 차량 원점에 좌표 축 그리기 (파란색 화살표)
        axis_length = 50  # 화살표 길이 (픽셀)
        # x축: 전방 (이미지에서 위쪽 방향)
        cv2.arrowedLine(img, self.origin, (self.origin[0], self.origin[1] - axis_length),
                        (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(img, 'x', (self.origin[0] - 15, self.origin[1] - axis_length - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # y축: 양의 방향이 왼쪽이므로, 원점에서 왼쪽으로 화살표
        cv2.arrowedLine(img, self.origin, (self.origin[0] - axis_length, self.origin[1]),
                        (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(img, 'y', (self.origin[0] - axis_length - 15, self.origin[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # 콘들을 이미지 좌표로 변환하여 그리기
        for i, cone in enumerate(cones[:num_cones]):
            x, y = cone
            u, v = world_to_image(x, y)
            
            # 클래스 이름에 따라 색상 결정
            if i < len(msg.class_names):
                class_name = msg.class_names[i].lower()  # 소문자로 변환
                self.get_logger().info(f"Cone {i}: class_name = {class_name}")  # 디버깅용
                
                if class_name in self.cone_colors:
                    color = self.cone_colors[class_name]
                else:
                    self.get_logger().warn(f"Unknown cone class: {class_name}")
                    color = self.cone_colors["unknown"]
            else:
                color = self.cone_colors["unknown"]

            # 콘 그리기 - 원
            cv2.circle(img, (u, v), 5, color, -1)
            
            # 콘 모양 삼각형 추가
            triangle_size = 8
            triangle_pts = np.array([
                [u, v - triangle_size*2],             # 꼭대기
                [u - triangle_size, v - triangle_size], # 왼쪽 바닥
                [u + triangle_size, v - triangle_size]  # 오른쪽 바닥
            ], np.int32)
            cv2.fillPoly(img, [triangle_pts], color)
            
            # 좌표 텍스트 표시
            text = f"({x:.2f}, {y:.2f})"
            cv2.putText(img, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            # 색상 텍스트 표시 (색상과 동일한 색으로 표시)
            if i < len(msg.class_names):
                cv2.putText(img, msg.class_names[i], (u + 5, v + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 내 차량(원점) 표시 (빨간색 원)
        cv2.circle(img, self.origin, 8, (0, 0, 255), -1)
        
        # 화면에 "UKF Filtered" 텍스트 표시
        cv2.putText(img, "UKF Filtered Cones", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ConeVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

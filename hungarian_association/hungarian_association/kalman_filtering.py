import rclpy
from rclpy.node import Node
import numpy as np

# 인터페이스에 맞춰 실제 사용하는 메시지 타입을 임포트합니다.
from custom_interface.msg import ModifiedFloat32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Track:
    def __init__(self, initial_position, color, dt=0.056): # 18hz, 1/18 = 0.056
        self.dt = dt
        # 상태벡터: x, y, vx, vy (초기 속도는 0으로 가정)
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        # 초기 공분산 행렬 (불확실성이 큰 값으로 초기화)
        # 값이 클수록 초기 추정에 대한 불확실성이 커져서 초기 측정값을 더 많이 신뢰
        self.P = np.eye(4) * 10.0

        # 칼만 필터 모델: 등속도 모델
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        # 프로세스 노이즈 조정 - 예측에 더 신뢰
        self.Q = np.eye(4) * 1.0  # 10.0 -> 1.0

        # 측정 모델: 위치만 측정한다고 가정 (x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # 측정 노이즈 조정 - 측정값에 더 신뢰
        self.R = np.eye(2) * 2.0  # 10.0 -> 2.0

        # 색상 관련 변수 개선
        self.color_history = []
        self.color_counts = {"unknown": 0, "blue cone": 0, "crimson cone": 0, "yellow cone": 0}
        self.max_history_size = 20  # 히스토리 크기 제한
        self.color_confidence_threshold = 3  # 색상 확정에 필요한 최소 관측 횟수
        
        # 초기 색상 처리
        color_lower = color.lower()
        self.add_color_to_history(color_lower)
        self.definite_color = None  # 아직 확정 색상 없음
        
        # 검출 누락 횟수 (트랙 종료 기준)
        self.missed_detections = 0

    def add_color_to_history(self, color):
        """색상을 히스토리에 추가하고 카운트 업데이트"""
        # 소문자로 통일
        color_lower = color.lower()
        
        # 히스토리에 추가
        self.color_history.append(color_lower)
        
        # 히스토리 크기 제한
        if len(self.color_history) > self.max_history_size:
            old_color = self.color_history.pop(0)
            # 빠지는 색상의 카운트 감소
            if old_color in self.color_counts:
                self.color_counts[old_color] -= 1
        
        # 새 색상 카운트 증가
        if color_lower in self.color_counts:
            self.color_counts[color_lower] += 1
        else:
            self.color_counts[color_lower] = 1
            
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement, color):
        z = np.array(measurement)
        y = z - (self.H @ self.state)  # 잔차
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # 색상 처리 로직 개선
        color_lower = color.lower()
        self.add_color_to_history(color_lower)
        
        # 확정 색상이 없고, unknown이 아닌 유효한 색상이 충분히 관측된 경우
        if self.definite_color is None:
            for cone_color, count in self.color_counts.items():
                if (cone_color != "unknown" and 
                    count >= self.color_confidence_threshold and 
                    cone_color in ["blue cone", "crimson cone", "yellow cone"]):
                    self.definite_color = cone_color
                    # self.get_logger().info(f"색상 확정: {self.definite_color}")
                    break
                    
        self.missed_detections = 0  # 업데이트 되었으므로 리셋

    def get_predicted_position(self):
        # 칼만 필터에 의한 추정 위치 (x, y)
        return self.state[:2]

    def get_smoothed_color(self):
        # 확정된 색상이 있으면 그 색상 반환
        if self.definite_color is not None:
            return self.definite_color
            
        # 확정된 색상이 없으면 현재까지 가장 많이 관측된 non-unknown 색상 반환
        best_color = "unknown"
        best_count = 0
        
        for color, count in self.color_counts.items():
            if color != "unknown" and count > best_count:
                best_color = color
                best_count = count
                
        # 유효한 색상이 없으면 unknown 반환
        if best_color == "unknown" and "unknown" in self.color_counts:
            return "Unknown"  # 원래 대소문자 유지
            
        return best_color.capitalize()  # 첫 글자 대문자화

class ConeTracker(Node):
    def __init__(self):
        super().__init__('cone_tracker')

        # 좀 더 실용적인 값으로 조정 (약 2-3초 정도)
        self.max_missed_detections = 36  # 18Hz 기준 약 2초
        
        # 거리 임계값은 현실적인 값으로 유지
        self.distance_threshold = 0.5  # 미터 단위
        
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10  # 버퍼 크기
        )

        # 원래 토픽 구독 (필터링 전 데이터) - QoS 프로파일 적용
        self.subscription = self.create_subscription(
            ModifiedFloat32MultiArray,
            '/fused_sorted_cones',
            self.listener_callback,
            qos_profile)

        # 필터링된 데이터를 발행할 퍼블리셔 생성 - 동일한 QoS 프로파일 적용
        self.publisher_ = self.create_publisher(
            ModifiedFloat32MultiArray,
            '/fused_sorted_cones_kalman',
            qos_profile)

        self.tracks = {}  # {track_id: Track 객체}
        self.next_track_id = 0
        self.original_detection_to_track_map = {}  # 원본 검출과 트랙 ID 매핑

    def listener_callback(self, msg):
        # 메시지 data는 [x1, y1, x2, y2, ...] 형태로 들어옴
        num_detections = len(msg.data) // 2
        detections = []
        for i in range(num_detections):
            x = msg.data[2 * i]
            y = msg.data[2 * i + 1]
            detections.append((x, y))
        
        # 색상 정보는 msg.class_names에 순서대로 들어있다고 가정
        colors = msg.class_names

        # 모든 트랙에 대해 예측 단계 실행
        for track in self.tracks.values():
            track.predict()

        # 원본 검출 인덱스와 트랙 매핑 초기화
        self.original_detection_to_track_map = {}
        assigned_tracks = set()

        # 각 검출에 대해 기존 트랙과 매칭 (최근접 매칭)
        for i, detection in enumerate(detections):
            best_track_id = None
            best_distance = float('inf')
            for track_id, track in self.tracks.items():
                pred_pos = track.get_predicted_position()
                dist = np.linalg.norm(np.array(detection) - pred_pos)
                if dist < best_distance and dist < self.distance_threshold:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                # 기존 트랙에 업데이트 적용
                self.tracks[best_track_id].update(detection, colors[i])
                assigned_tracks.add(best_track_id)
                # 원본 인덱스와 트랙 매핑 저장
                self.original_detection_to_track_map[i] = best_track_id
            else:
                # 매칭되는 트랙이 없으면 새 트랙 생성
                self.tracks[self.next_track_id] = Track(detection, colors[i])
                self.get_logger().info(f"새 트랙 생성: {self.next_track_id}")
                # 원본 인덱스와 새 트랙 매핑 저장
                self.original_detection_to_track_map[i] = self.next_track_id
                self.next_track_id += 1

        # 매칭되지 않은 트랙은 검출 누락 처리
        for track_id, track in list(self.tracks.items()):
            if track_id not in assigned_tracks:
                track.missed_detections += 1
            if track.missed_detections > self.max_missed_detections:
                self.get_logger().info(f"트랙 종료: {track_id}")
                del self.tracks[track_id]

        # 필터링된 결과를 담은 메시지를 생성하여 발행
        filtered_msg = ModifiedFloat32MultiArray()
        # 헤더를 원본 메시지와 동일하게 사용 (timestamp 포함)
        filtered_msg.header = msg.header
        # 레이아웃도 원본 메시지와 동일하게 유지
        filtered_msg.layout = msg.layout
        
        # 원본 메시지와 동일한 순서로 클래스 이름 유지
        filtered_msg.class_names = msg.class_names.copy()
        
        # 필터링된 데이터 생성 (원본 메시지와 같은 인덱스 순서 유지)
        filtered_data = []
        for i in range(num_detections):
            if i in self.original_detection_to_track_map:
                track_id = self.original_detection_to_track_map[i]
                if track_id in self.tracks:  # 트랙이 여전히 유효한지 확인
                    pos = self.tracks[track_id].get_predicted_position()
                    filtered_data.extend(pos.tolist())
                else:
                    # 트랙이 삭제된 경우 원본 데이터 사용
                    filtered_data.extend([msg.data[2*i], msg.data[2*i+1]])
            else:
                # 매핑이 없는 경우 원본 데이터 사용
                filtered_data.extend([msg.data[2*i], msg.data[2*i+1]])
        
        filtered_msg.data = filtered_data

        # 필터링된 메시지를 발행
        self.publisher_.publish(filtered_msg)

        # 디버그용: 각 트랙의 상태 로깅 개선
        for track_id, track in self.tracks.items():
            pos = track.get_predicted_position()
            color = track.get_smoothed_color()
            velocity = track.state[2:4]  # vx, vy
            speed = np.linalg.norm(velocity)
            
            # 더 많은 정보를 포함한 디버그 메시지
            self.get_logger().debug(
                f"트랙 {track_id}: 위치 {pos}, 속도 {speed:.2f}m/s, "
                f"색상 {color}, 누락 {track.missed_detections}/{self.max_missed_detections}"
            )

def main(args=None):
    rclpy.init(args=args)
    cone_tracker = ConeTracker()
    rclpy.spin(cone_tracker)
    cone_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

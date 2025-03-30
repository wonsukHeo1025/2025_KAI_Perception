2025 KAI Perception

2025년 건국대학교 자작자동차 동아리 Team K.A.I. 자율주행팀 인지파트에서 수행하는 카메라-Lidar 센서 퓨전 프로젝트에 관한 내용을 담고 있습니다.

주요 기능

센서 데이터 통합 처리

실시간 객체 탐지 및 추적

효율적인 센서 데이터 융합 알고리즘

모듈화된 소프트웨어 아키텍처

폴더 구조 및 설명

src/

cam_yolo/: yolo 모델 테스트용

cone_detection/: 콘 객체의 정확한 인지 및 위치 파악 알고리즘 구현

cone_projection/: 센서 퓨전 결과를 시각화

custom_interface/: 커스텀 메시지

hungarian_association/: 헝가리안 알고리즘을 이용한 센서 간 객체 매칭 및 데이터 연관성 분석

ouster-ros/: ROS2 Ouster Lidar 패키지

ros2_camera_lidar_fusion/: 카메라 내부 캘리브레이션, 카메라-Lidar 외부 캘리브레이션을 위한 패키지

usb_cam/: ROS2 usb_cam 패키지

yolo_ros/: YOLO 모델로 디텍션한 객체 정보를 ROS2로 퍼블리시하는 패키지

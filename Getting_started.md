- 라이다 드라이버
```
ros2 launch ouster_ros driver.launch.py params_file:='/home/kai/KAI_ws/src/ouster-ros/ouster-ros/config/driver_params.yaml'
```

- 카메라 드라이버
```
usbcam1
```
```
usbcam2
```

- YOLO 듀얼카메라
```
ros2 run yolo_ros yolo_dual_camera_node
```

- 라이다 포인트 클라우드 보간
```
ros2 launch prism prism.launch.py
```

- 라이다 콘 디텍션
```
ros2 launch cone_detection cone_detection_launch.py
```

- 카메라 라이다 퓨전
```
ros2 launch calico calico_full.launch.py
```

- tf_static
```
ros2 launch gps_imu_fusion tf_static.launch.py
```

- GPS 노드 키기 (ttyinfo.sh 로 /dev/ttyACM* 번호 체크)
```
ros2 launch ublox_gps ublox_gps_node-launch.py
```

- NTRIP RTK 신호 수신 클라이언트 (인터넷 연결 확인 후 패킷 받아와질때까지 명령어 무한 반복 실행)
```
ros2 launch ntrip_client ntrip_client_launch.py
```

- NTRIP 신호 Ublox 변환 노드
```
ros2 run fix2nmea fix2nmea
```

- IMU 드라이버
```
ros2 launch myahrs_ros2_driver myahrs_ros2_driver.launch.py 
```

- GPS + IMU EKF 퓨전
```
ros2 launch gps_imu_fusion ekf_fusion.launch.py 
```

- GPS 속도, GPS+IMU 퓨전 속도 토픽 출력
```
ros2 run gps_imu_fusion velocity_magnitude_node.py
```


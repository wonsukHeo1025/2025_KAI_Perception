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

- GPS + IMU EKF 퓨전
```
ros2 launch gps_imu_fusion ekf_fusion.launch.py 
```

- GPS 속도, GPS+IMU 퓨전 속도 토픽 출력
```
ros2 run gps_imu_fusion velocity_magnitude_node.py
```


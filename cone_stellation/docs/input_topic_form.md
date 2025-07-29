❯ ros2 topic echo /ublox_gps_node/fix --once
header:
  stamp:
    sec: 1749793040
    nanosec: 875447638
  frame_id: gps
status:
  status: 2
  service: 15
latitude: 37.5413753
longitude: 127.0779785
altitude: 39.5
position_covariance:
- 0.00019600000000000002
- 0.0
- 0.0
- 0.0
- 0.00019600000000000002
- 0.0
- 0.0
- 0.0
- 0.0001
position_covariance_type: 2
---
❯ ros2 topic hz /ublox_gps_node/fix
average rate: 8.011
	min: 0.120s max: 0.131s std dev: 0.00296s window: 9
average rate: 7.998
	min: 0.119s max: 0.134s std dev: 0.00343s window: 17

❯ ros2 topic echo /ublox_gps_node/fix_velocity --once
header:
  stamp:
    sec: 1749793044
    nanosec: 447046
  frame_id: gps
twist:
  twist:
    linear:
      x: -0.852
      y: 0.105
      z: 0.08700000000000001
    angular:
      x: 0.0
      y: 0.0
      z: 0.0
  covariance:
  - 0.005929
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.005929
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.005929
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - -1.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
---

❯ ros2 topic echo /ouster/imu --once
header:
  stamp:
    sec: 1749793067
    nanosec: 9071919
  frame_id: os_imu
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
orientation_covariance:
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
angular_velocity:
  x: 0.03768372942462122
  y: -0.0038615835806148956
  z: 0.09001484484467824
angular_velocity_covariance:
- 0.0006
- 0.0
- 0.0
- 0.0
- 0.0006
- 0.0
- 0.0
- 0.0
- 0.0006
linear_acceleration:
  x: 1.4939818359375
  y: 0.1580173095703125
  z: 9.540893615722656
linear_acceleration_covariance:
- 0.01
- 0.0
- 0.0
- 0.0
- 0.01
- 0.0
- 0.0
- 0.0
- 0.01
---

❯ ros2 topic hz /ouster/imu
average rate: 99.980
	min: 0.009s max: 0.011s std dev: 0.00021s window: 102
average rate: 99.982
	min: 0.009s max: 0.011s std dev: 0.00024s window: 202

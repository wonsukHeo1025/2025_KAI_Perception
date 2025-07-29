# Sensor Simulation Documentation

## Overview

ConeSTELLATION includes comprehensive sensor simulation capabilities for testing and development. The enhanced simulators provide realistic IMU and GPS data with configurable noise models, enabling thorough testing of the EKF fusion and SLAM system without requiring real hardware.

## IMU Simulation

### Features

The IMU simulator (`sensor_simulator_enhanced.py`) implements a high-fidelity model including:

1. **Allan Variance Noise Model**
   - Angle/velocity random walk
   - Bias instability
   - White noise density

2. **Temperature Effects**
   - Temperature-dependent bias drift
   - Thermal time constants
   - Reference temperature calibration

3. **Scale Factor Errors**
   - Gyroscope scale factor (ppm)
   - Accelerometer scale factor (ppm)
   - Non-linearity effects

4. **Axis Misalignment**
   - Cross-axis sensitivity
   - Mounting misalignment errors
   - G-sensitivity (gyro error due to acceleration)

### IMU Configuration

```python
@dataclass
class EnhancedImuConfig:
    # Allan variance parameters
    gyro_noise_density: float = 0.005      # rad/s/√Hz
    gyro_bias_stability: float = 0.1       # rad/s
    gyro_random_walk: float = 0.00001      # rad/s²/√Hz
    
    accel_noise_density: float = 0.01      # m/s²/√Hz
    accel_bias_stability: float = 0.01     # m/s²
    accel_random_walk: float = 0.0001      # m/s³/√Hz
    
    # Temperature effects
    temperature_reference: float = 25.0     # °C
    gyro_temp_coefficient: float = 0.01     # rad/s/°C
    accel_temp_coefficient: float = 0.001   # m/s²/°C
    temperature_time_constant: float = 300  # seconds
    
    # Scale factor errors (parts per million)
    gyro_scale_factor_ppm: float = 500     # ppm
    accel_scale_factor_ppm: float = 1000   # ppm
    
    # Axis misalignment (radians)
    gyro_misalignment_rad: float = 0.001   # ~0.057 degrees
    accel_misalignment_rad: float = 0.002  # ~0.115 degrees
    
    # G-sensitivity
    gyro_g_sensitivity: float = 0.0001     # rad/s/g
```

### IMU Noise Generation

The simulator generates realistic IMU measurements using:

```python
def generate_imu_measurement(true_state, dt):
    # 1. Apply scale factor errors
    scaled_gyro = true_angular_vel * (1 + scale_factor_error)
    scaled_accel = true_linear_accel * (1 + scale_factor_error)
    
    # 2. Add axis misalignment
    misaligned_gyro = misalignment_matrix @ scaled_gyro
    misaligned_accel = misalignment_matrix @ scaled_accel
    
    # 3. Add temperature-dependent bias
    temp_bias = (current_temp - ref_temp) * temp_coefficient
    
    # 4. Add bias instability (random walk)
    bias += bias_random_walk * sqrt(dt) * randn()
    
    # 5. Add white noise
    noise = noise_density * sqrt(1/dt) * randn()
    
    # 6. Add g-sensitivity (gyro only)
    g_error = g_sensitivity * acceleration_magnitude
    
    return measurement + bias + temp_bias + noise + g_error
```

## GPS/RTK Simulation

### Features

The GPS simulator provides realistic RTK GPS behavior including:

1. **RTK Status Modes**
   - RTK Fix (2cm accuracy)
   - RTK Float (30cm accuracy)
   - Single/DGPS (2m accuracy)
   - No Fix (10m accuracy)

2. **Status Transitions**
   - Probabilistic fix loss/recovery
   - Realistic transition times
   - Environment-dependent behavior

3. **Error Sources**
   - Dilution of Precision (DOP)
   - Multipath effects
   - Atmospheric delays
   - Clock errors

4. **Coordinate Systems**
   - WGS84 lat/lon/alt output
   - UTM local frame conversion
   - Proper datum handling

### GPS Configuration

```python
@dataclass
class EnhancedGpsConfig:
    # RTK mode parameters
    rtk_mode: str = "rtk_fix"  # rtk_fix, rtk_float, single, no_fix
    
    # Fix mode noise (1-sigma)
    rtk_fix_noise_h: float = 0.02          # m - horizontal
    rtk_fix_noise_v: float = 0.04          # m - vertical
    rtk_float_noise_h: float = 0.3         # m
    rtk_float_noise_v: float = 0.5         # m
    single_noise_h: float = 2.0            # m
    single_noise_v: float = 5.0            # m
    no_fix_noise_h: float = 10.0           # m
    no_fix_noise_v: float = 15.0           # m
    
    # RTK status transition probabilities
    fix_loss_probability: float = 0.001     # per second
    float_to_fix_probability: float = 0.1   # per second
    single_to_float_probability: float = 0.05 # per second
    
    # DOP (Dilution of Precision) effects
    hdop_min: float = 0.8
    hdop_max: float = 2.0
    vdop_min: float = 1.0
    vdop_max: float = 3.0
    
    # Multipath parameters
    multipath_amplitude: float = 0.5        # meters
    multipath_frequency: float = 0.1        # Hz
    
    # Update rate
    update_rate: float = 10.0               # Hz
```

### RTK Status Simulation

```python
def update_rtk_status(current_status, dt):
    """Probabilistic RTK status transitions"""
    if current_status == RtkStatus.FIX:
        if random() < fix_loss_probability * dt:
            return RtkStatus.FLOAT
    
    elif current_status == RtkStatus.FLOAT:
        if random() < float_to_fix_probability * dt:
            return RtkStatus.FIX
        elif random() < fix_loss_probability * dt:
            return RtkStatus.SINGLE
    
    elif current_status == RtkStatus.SINGLE:
        if random() < single_to_float_probability * dt:
            return RtkStatus.FLOAT
    
    return current_status
```

### GPS Error Modeling

```python
def generate_gps_measurement(true_position_utm):
    # 1. Get noise based on RTK status
    noise_h, noise_v = get_noise_for_status(rtk_status)
    
    # 2. Apply DOP scaling
    hdop = random.uniform(hdop_min, hdop_max)
    vdop = random.uniform(vdop_min, vdop_max)
    
    # 3. Add multipath effects
    multipath = multipath_amplitude * sin(2*pi*multipath_frequency*t)
    
    # 4. Generate position error
    error_north = noise_h * hdop * randn() + multipath
    error_east = noise_h * hdop * randn() + multipath
    error_up = noise_v * vdop * randn()
    
    # 5. Convert to lat/lon/alt
    measured_utm = true_utm + [error_east, error_north, error_up]
    lat, lon = utm.to_latlon(measured_utm[0], measured_utm[1], 
                             zone_number, zone_letter)
    
    return NavSatFix(latitude=lat, longitude=lon, altitude=alt,
                     status=rtk_status_to_ros(rtk_status))
```

## Motion Profiles

### Available Test Paths

The simulator supports multiple motion profiles for comprehensive testing:

1. **Straight Line**
   - Simple forward motion
   - Constant velocity
   - Emergency braking test

2. **Circular Path**
   - Constant radius turn
   - Tests centripetal acceleration
   - Continuous heading change

3. **Figure-8 Path**
   - Complex trajectory
   - Varying curvature
   - Direction changes

4. **Formula Student Track**
   - Realistic race track
   - High-speed sections
   - Tight corners and chicanes

### Motion Profile Configuration

```python
class MotionProfile:
    def __init__(self, profile_type="figure8"):
        self.profiles = {
            "straight": self.straight_motion,
            "circular": self.circular_motion,
            "figure8": self.figure8_motion,
            "formula_student": self.fs_track_motion
        }
        
    def straight_motion(self, t):
        # Constant velocity with acceleration/deceleration phases
        if t < 2.0:  # Acceleration
            v = 5.0 * t
            x = 2.5 * t**2
        elif t < 10.0:  # Constant
            v = 10.0
            x = 20.0 + 10.0 * (t - 2.0)
        else:  # Deceleration
            v = max(0, 10.0 - 5.0 * (t - 10.0))
            x = 100.0 + 10.0*(t-10.0) - 2.5*(t-10.0)**2
        
        return np.array([x, 0, 0]), np.array([v, 0, 0])
    
    def circular_motion(self, t):
        # Circular path with radius R
        R = 20.0  # meters
        omega = 0.5  # rad/s
        
        theta = omega * t
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        
        vx = -R * omega * np.sin(theta)
        vy = R * omega * np.cos(theta)
        
        return np.array([x, y, 0]), np.array([vx, vy, 0])
    
    def figure8_motion(self, t):
        # Figure-8 using Lissajous curve
        A = 25.0  # amplitude
        omega = 0.3  # frequency
        
        x = A * np.sin(omega * t)
        y = A * np.sin(2 * omega * t) / 2
        
        vx = A * omega * np.cos(omega * t)
        vy = A * omega * np.cos(2 * omega * t)
        
        return np.array([x, y, 0]), np.array([vx, vy, 0])
```

## Integration with ROS2

### Publishing Simulated Data

```python
class SensorSimulatorNode(Node):
    def __init__(self):
        # Publishers
        self.imu_pub = self.create_publisher(Imu, '/ouster/imu', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/ublox_gps_node/fix', 10)
        self.gps_vel_pub = self.create_publisher(
            TwistWithCovarianceStamped, '/ublox_gps_node/fix_velocity', 10)
        
        # Timer for 100Hz IMU, 10Hz GPS
        self.imu_timer = self.create_timer(0.01, self.publish_imu)
        self.gps_timer = self.create_timer(0.1, self.publish_gps)
    
    def publish_imu(self):
        # Get true state from motion profile
        true_state = self.motion_controller.get_state(self.get_clock().now())
        
        # Generate IMU measurement
        imu_msg = self.imu_simulator.generate_measurement(true_state)
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "imu_link"
        
        self.imu_pub.publish(imu_msg)
    
    def publish_gps(self):
        # Update RTK status
        self.gps_simulator.update_rtk_status()
        
        # Generate GPS measurement
        gps_msg = self.gps_simulator.generate_measurement(true_state)
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = "gps_link"
        
        self.gps_pub.publish(gps_msg)
```

### Launch Configuration

```python
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cone_stellation',
            executable='imu_gps_publishers.py',
            name='sensor_simulator',
            parameters=[{
                'motion_profile': 'figure8',
                'imu_config': {
                    'gyro_noise_density': 0.005,
                    'accel_noise_density': 0.01,
                    'temperature_effects': True
                },
                'gps_config': {
                    'rtk_mode': 'rtk_fix',
                    'enable_multipath': True,
                    'transition_probabilities': {
                        'fix_loss': 0.001,
                        'float_to_fix': 0.1
                    }
                }
            }]
        )
    ])
```

## Testing and Validation

### Ground Truth Comparison

The simulator maintains ground truth data for validation:

```python
class GroundTruthRecorder:
    def __init__(self):
        self.true_trajectory = []
        self.true_velocities = []
        self.true_accelerations = []
        
    def record_state(self, timestamp, state):
        self.true_trajectory.append({
            'time': timestamp,
            'position': state.position,
            'orientation': state.orientation,
            'velocity': state.linear_velocity,
            'angular_velocity': state.angular_velocity
        })
    
    def save_to_file(self, filename):
        # Save as CSV for analysis
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                'vx', 'vy', 'vz', 'wx', 'wy', 'wz'
            ])
            writer.writeheader()
            # Write data...
```

### Noise Model Validation

Tools for validating sensor noise characteristics:

```python
def validate_allan_variance(imu_data, sampling_rate):
    """Compute Allan variance to verify noise parameters"""
    # Implementation of Allan variance computation
    tau, allan_dev = compute_allan_variance(imu_data)
    
    # Extract noise parameters
    angle_random_walk = extract_arw(tau, allan_dev)
    bias_instability = extract_bias_instability(tau, allan_dev)
    
    return {
        'angle_random_walk': angle_random_walk,
        'bias_instability': bias_instability,
        'matches_config': validate_against_config(...)
    }
```

## Usage Examples

### Basic Sensor Simulation

```bash
# Launch sensor simulators with figure-8 motion
ros2 launch cone_stellation imu_gps_ekf_launch.py motion_profile:=figure8

# Launch with custom noise parameters
ros2 launch cone_stellation imu_gps_ekf_launch.py \
    gyro_noise:=0.01 \
    gps_mode:=rtk_float \
    multipath:=true
```

### Testing Different Scenarios

```python
# Test RTK fix loss scenario
simulator.set_rtk_loss_probability(0.1)  # 10% per second
simulator.run_for_duration(60.0)  # Run for 60 seconds

# Test high temperature drift
simulator.set_temperature_profile(lambda t: 25.0 + 20.0 * sin(0.1 * t))
simulator.run_for_duration(300.0)  # 5 minutes

# Test multipath in urban environment
simulator.set_multipath_amplitude(2.0)  # 2 meter multipath
simulator.set_multipath_frequency(0.5)  # 0.5 Hz oscillation
```

## Performance Considerations

- IMU runs at 100Hz with minimal computational overhead
- GPS runs at 10Hz with UTM conversion
- Motion profiles are pre-computed for efficiency
- Noise generation uses efficient random number generators
- Ground truth recording has negligible impact

## Future Enhancements

1. **Additional Sensors**
   - Wheel odometry simulation
   - Visual odometry error models
   - Magnetometer with hard/soft iron effects

2. **Environmental Effects**
   - GPS signal occlusion modeling
   - IMU vibration effects
   - Temperature profiles from track conditions

3. **Advanced Motion Profiles**
   - Recorded real race data playback
   - Dynamic obstacle avoidance scenarios
   - Multi-vehicle interactions
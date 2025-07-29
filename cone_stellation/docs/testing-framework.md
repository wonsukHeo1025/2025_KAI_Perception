# Testing Framework Documentation

## Overview

ConeSTELLATION includes a comprehensive testing framework with predefined motion profiles, ground truth tracking, and performance evaluation tools. This enables systematic validation of the SLAM system under various conditions without requiring physical hardware or real-world data.

## Motion Profiles

### Available Test Paths

#### 1. Straight Line Path
Simple forward motion for basic validation and emergency braking scenarios.

**Parameters:**
- Length: 150 meters
- Max velocity: 10 m/s
- Acceleration: 5 m/s²
- Test duration: ~15 seconds

**Use Cases:**
- Initial system validation
- Longitudinal dynamics testing
- Emergency braking scenarios
- Sensor alignment verification

```python
def straight_motion(self, t):
    if t < 2.0:  # Acceleration phase
        v = 5.0 * t
        x = 2.5 * t**2
    elif t < 10.0:  # Constant velocity
        v = 10.0
        x = 20.0 + 10.0 * (t - 2.0)
    else:  # Deceleration phase
        v = max(0, 10.0 - 5.0 * (t - 10.0))
        x = 100.0 + 10.0*(t-10.0) - 2.5*(t-10.0)**2
    
    return position, velocity, acceleration
```

#### 2. Circular Path
Constant radius circular motion for testing centripetal acceleration handling.

**Parameters:**
- Radius: 20 meters
- Angular velocity: 0.5 rad/s
- Linear velocity: 10 m/s
- Lateral acceleration: 5 m/s²

**Use Cases:**
- IMU gyroscope validation
- Continuous heading change
- Lateral dynamics testing
- GPS/IMU fusion under rotation

```python
def circular_motion(self, t):
    R = 20.0  # radius in meters
    omega = 0.5  # angular velocity in rad/s
    
    theta = omega * t
    position = [R * cos(theta), R * sin(theta), 0]
    velocity = [-R * omega * sin(theta), R * omega * cos(theta), 0]
    acceleration = [-R * omega² * cos(theta), -R * omega² * sin(theta), 0]
    
    return position, velocity, acceleration
```

#### 3. Figure-8 Path
Complex trajectory using Lissajous curves for comprehensive testing.

**Parameters:**
- Amplitude: 25 meters
- Frequency: 0.3 rad/s
- Max velocity: 15 m/s
- Max acceleration: 7 m/s²

**Use Cases:**
- Complex trajectory tracking
- Direction reversal handling
- Variable curvature paths
- Full system stress testing

```python
def figure8_motion(self, t):
    A = 25.0  # amplitude
    omega = 0.3  # base frequency
    
    # Lissajous curve
    x = A * sin(omega * t)
    y = A * sin(2 * omega * t) / 2
    
    # Velocities
    vx = A * omega * cos(omega * t)
    vy = A * omega * cos(2 * omega * t)
    
    # Accelerations
    ax = -A * omega² * sin(omega * t)
    ay = -2 * A * omega² * sin(2 * omega * t)
    
    return position, velocity, acceleration
```

#### 4. Formula Student Track
Realistic racing track based on FS competition layouts.

**Parameters:**
- Track length: ~300 meters
- Max velocity: 20 m/s
- Mix of straights and corners
- Chicane sections

**Features:**
- Long straights (80m)
- Tight corners (R=5m)
- High-speed sections
- Technical chicanes
- Realistic cone placement

```python
class FormulaStudentTrack:
    def __init__(self):
        # Define track waypoints
        self.waypoints = [
            (35.0, 12.5),   # Start/finish straight
            (88.0, 12.5),   # End of straight
            (94.36, 3.22),  # Tight right corner
            (104.72, 5.28), # Corner exit
            (106.78, 15.64),# Left sweep
            (98.00, 21.50), # Chicane entry
            # ... additional waypoints
        ]
        
        # Generate smooth spline through waypoints
        self.trajectory = self.generate_spline_trajectory()
```

### Motion Profile Configuration

```yaml
# motion_profiles.yaml
motion_controller:
  profile_type: "figure8"  # straight, circular, figure8, formula_student
  
  velocity_limits:
    max_velocity: 20.0      # m/s
    max_acceleration: 10.0  # m/s²
    max_jerk: 50.0         # m/s³
    
  safety_limits:
    max_lateral_acceleration: 8.0  # m/s²
    max_yaw_rate: 2.0             # rad/s
    
  trajectory_smoothing:
    spline_degree: 3
    smoothing_factor: 0.1
```

## Ground Truth System

### Architecture

The ground truth system maintains perfect knowledge of:
1. Vehicle true pose and velocity
2. Cone true positions
3. Sensor measurement errors
4. System timing information

```python
class GroundTruthManager:
    def __init__(self):
        self.true_vehicle_states = []
        self.true_cone_positions = {}
        self.measurement_errors = []
        self.timing_statistics = []
        
    def record_vehicle_state(self, timestamp, state):
        """Record true vehicle pose and dynamics"""
        self.true_vehicle_states.append({
            'timestamp': timestamp,
            'position': state.position,
            'orientation': state.orientation,
            'velocity': state.linear_velocity,
            'angular_velocity': state.angular_velocity,
            'acceleration': state.linear_acceleration
        })
    
    def record_measurement_error(self, timestamp, sensor_type, error):
        """Track sensor measurement errors"""
        self.measurement_errors.append({
            'timestamp': timestamp,
            'sensor': sensor_type,
            'error': error,
            'statistics': self.compute_error_statistics(error)
        })
```

### Cone Ground Truth

```python
class ConeGroundTruth:
    def __init__(self, track_generator):
        self.true_cones = track_generator.generate_cones()
        
    def get_visible_cones(self, vehicle_pose, sensor_config):
        """Get cones visible from current pose"""
        visible = []
        for cone in self.true_cones:
            # Check range
            distance = np.linalg.norm(cone.position - vehicle_pose.position)
            if distance > sensor_config.max_range:
                continue
                
            # Check field of view
            angle = self.compute_bearing(vehicle_pose, cone)
            if abs(angle) > sensor_config.fov / 2:
                continue
                
            # Check occlusion (simplified)
            if not self.is_occluded(vehicle_pose, cone):
                visible.append(cone)
                
        return visible
```

### Error Metrics

```python
class ErrorMetrics:
    def compute_pose_error(self, true_pose, estimated_pose):
        """Compute pose estimation error"""
        # Position error
        position_error = np.linalg.norm(
            true_pose.position - estimated_pose.position)
        
        # Orientation error (using quaternion metric)
        q_true = quaternion_from_matrix(true_pose.rotation)
        q_est = quaternion_from_matrix(estimated_pose.rotation)
        orientation_error = quaternion_distance(q_true, q_est)
        
        # Compute APE (Absolute Pose Error)
        ape = {
            'position': position_error,
            'orientation': orientation_error,
            'combined': np.sqrt(position_error**2 + orientation_error**2)
        }
        
        return ape
    
    def compute_trajectory_metrics(self, true_trajectory, estimated_trajectory):
        """Compute trajectory-level metrics"""
        # Align trajectories temporally
        aligned_true, aligned_est = self.align_trajectories(
            true_trajectory, estimated_trajectory)
        
        # Compute metrics
        metrics = {
            'ate': self.absolute_trajectory_error(aligned_true, aligned_est),
            'rpe': self.relative_pose_error(aligned_true, aligned_est),
            'drift': self.compute_drift_rate(aligned_true, aligned_est),
            'completeness': len(aligned_est) / len(aligned_true)
        }
        
        return metrics
```

## Performance Evaluation

### Real-time Metrics

```python
class PerformanceMonitor:
    def __init__(self):
        self.timing_buffer = deque(maxlen=1000)
        self.cpu_usage = []
        self.memory_usage = []
        
    def record_iteration(self, start_time, end_time, stage):
        """Record timing for SLAM iteration"""
        self.timing_buffer.append({
            'stage': stage,
            'duration': end_time - start_time,
            'timestamp': start_time
        })
    
    def get_statistics(self):
        """Compute performance statistics"""
        if not self.timing_buffer:
            return {}
            
        durations = [t['duration'] for t in self.timing_buffer]
        return {
            'mean_ms': np.mean(durations) * 1000,
            'std_ms': np.std(durations) * 1000,
            'max_ms': np.max(durations) * 1000,
            'p95_ms': np.percentile(durations, 95) * 1000,
            'frequency_hz': 1.0 / np.mean(durations)
        }
```

### Evaluation Pipeline

```python
class SLAMEvaluator:
    def __init__(self, ground_truth_manager):
        self.ground_truth = ground_truth_manager
        self.results = {}
        
    def evaluate_run(self, slam_output):
        """Comprehensive evaluation of SLAM run"""
        # Trajectory accuracy
        self.results['trajectory'] = self.evaluate_trajectory(
            self.ground_truth.true_vehicle_states,
            slam_output.estimated_poses)
        
        # Mapping accuracy
        self.results['mapping'] = self.evaluate_mapping(
            self.ground_truth.true_cone_positions,
            slam_output.estimated_landmarks)
        
        # Timing performance
        self.results['performance'] = self.evaluate_performance(
            slam_output.timing_data)
        
        # Consistency checks
        self.results['consistency'] = self.check_consistency(
            slam_output)
        
        return self.results
```

## Testing Workflows

### Automated Test Suite

```python
class AutomatedTestSuite:
    def __init__(self):
        self.test_configurations = [
            {
                'name': 'basic_straight',
                'motion': 'straight',
                'duration': 20.0,
                'sensors': ['imu', 'gps'],
                'expected_drift': 0.1  # meters
            },
            {
                'name': 'circular_imu_only',
                'motion': 'circular',
                'duration': 60.0,
                'sensors': ['imu'],
                'expected_drift': 5.0
            },
            {
                'name': 'figure8_full_sensors',
                'motion': 'figure8',
                'duration': 120.0,
                'sensors': ['imu', 'gps', 'cones'],
                'expected_drift': 0.5
            }
        ]
    
    def run_all_tests(self):
        """Execute complete test suite"""
        results = []
        for config in self.test_configurations:
            result = self.run_single_test(config)
            results.append(result)
            
            # Check pass/fail
            if result['drift'] > config['expected_drift']:
                print(f"FAILED: {config['name']} - "
                      f"Drift {result['drift']:.2f}m exceeds "
                      f"limit {config['expected_drift']}m")
        
        return results
```

### Launch Testing

```bash
# Run specific motion profile
ros2 launch cone_stellation test_slam_launch.py \
    motion_profile:=figure8 \
    test_duration:=120.0 \
    enable_ground_truth:=true

# Run with specific sensor configuration
ros2 launch cone_stellation test_slam_launch.py \
    motion_profile:=circular \
    disable_gps:=true \
    imu_noise_scale:=2.0

# Run automated test suite
ros2 run cone_stellation run_test_suite.py \
    --output-dir ./test_results \
    --generate-plots
```

### Visualization Tools

```python
class TestVisualizer:
    def plot_trajectory_comparison(self, true_traj, est_traj):
        """Plot true vs estimated trajectory"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY trajectory
        axes[0,0].plot(true_traj.x, true_traj.y, 'g-', label='True')
        axes[0,0].plot(est_traj.x, est_traj.y, 'b--', label='Estimated')
        axes[0,0].set_xlabel('X (m)')
        axes[0,0].set_ylabel('Y (m)')
        axes[0,0].legend()
        axes[0,0].set_title('Trajectory Comparison')
        
        # Error over time
        errors = self.compute_errors(true_traj, est_traj)
        axes[0,1].plot(errors.time, errors.position, 'r-')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Position Error (m)')
        axes[0,1].set_title('Position Error Evolution')
        
        # Heading comparison
        axes[1,0].plot(true_traj.time, true_traj.yaw, 'g-', label='True')
        axes[1,0].plot(est_traj.time, est_traj.yaw, 'b--', label='Estimated')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Yaw (rad)')
        axes[1,0].legend()
        
        # Error distribution
        axes[1,1].hist(errors.position, bins=50, alpha=0.7)
        axes[1,1].set_xlabel('Position Error (m)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Error Distribution')
        
        plt.tight_layout()
        return fig
```

## Batch Processing

### Parameter Sweep Testing

```python
class ParameterSweepTester:
    def __init__(self):
        self.parameter_grid = {
            'imu_noise': [0.001, 0.005, 0.01, 0.05],
            'gps_accuracy': ['rtk_fix', 'rtk_float', 'single'],
            'cone_detection_range': [10.0, 20.0, 30.0],
            'optimization_rate': [10, 20, 30]  # Hz
        }
    
    def run_parameter_sweep(self):
        """Test all parameter combinations"""
        from itertools import product
        
        results = []
        for params in product(*self.parameter_grid.values()):
            config = dict(zip(self.parameter_grid.keys(), params))
            
            # Run test with configuration
            result = self.run_test_with_config(config)
            result['parameters'] = config
            results.append(result)
        
        # Analyze results
        self.analyze_parameter_sensitivity(results)
        return results
```

### Monte Carlo Testing

```python
class MonteCarloTester:
    def __init__(self, num_runs=100):
        self.num_runs = num_runs
        
    def run_monte_carlo(self, base_config):
        """Run multiple trials with random variations"""
        results = []
        
        for i in range(self.num_runs):
            # Add random variations
            config = self.add_random_variations(base_config)
            
            # Run test
            result = self.run_single_test(config)
            results.append(result)
        
        # Compute statistics
        statistics = {
            'mean_drift': np.mean([r['drift'] for r in results]),
            'std_drift': np.std([r['drift'] for r in results]),
            'success_rate': sum(r['success'] for r in results) / self.num_runs,
            'percentiles': {
                'p50': np.percentile([r['drift'] for r in results], 50),
                'p95': np.percentile([r['drift'] for r in results], 95),
                'p99': np.percentile([r['drift'] for r in results], 99)
            }
        }
        
        return statistics
```

## Data Recording and Analysis

### Test Data Recording

```python
class TestDataRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def record_test_run(self, test_name, data):
        """Save all test data for analysis"""
        # Save trajectories
        np.savetxt(f"{self.output_dir}/{test_name}_true_traj.csv",
                   data['true_trajectory'])
        np.savetxt(f"{self.output_dir}/{test_name}_est_traj.csv",
                   data['estimated_trajectory'])
        
        # Save landmarks
        with open(f"{self.output_dir}/{test_name}_landmarks.json", 'w') as f:
            json.dump(data['landmarks'], f, indent=2)
        
        # Save metrics
        with open(f"{self.output_dir}/{test_name}_metrics.json", 'w') as f:
            json.dump(data['metrics'], f, indent=2)
        
        # Save timing data
        pd.DataFrame(data['timing']).to_csv(
            f"{self.output_dir}/{test_name}_timing.csv")
```

### Post-Processing Tools

```bash
# Generate test report
ros2 run cone_stellation generate_test_report.py \
    --data-dir ./test_results \
    --output report.pdf

# Compare multiple runs
ros2 run cone_stellation compare_runs.py \
    --baseline ./test_results/baseline \
    --comparison ./test_results/new_version \
    --metrics ate,rpe,computation_time

# Export for external analysis
ros2 run cone_stellation export_to_evo.py \
    --input ./test_results \
    --output ./evo_format
```
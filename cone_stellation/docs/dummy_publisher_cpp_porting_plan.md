# ConeSTELLATION C++ Non-Invasive Porting Plan

## Overview
This document outlines a **non-invasive** plan for porting the ConeSTELLATION Python implementation to C++ while maintaining complete independence from the existing Python codebase. This approach ensures we can always revert to the Python implementation if needed.

## 1. Non-Invasive Implementation Strategy

### Key Principles:
1. **Complete Independence**: C++ implementation will not modify or depend on Python code
2. **Parallel Execution**: Both Python and C++ versions can run simultaneously
3. **Different Node Names**: C++ nodes will have `_cpp` suffix to avoid conflicts
4. **Separate Branch**: All development in dedicated `feature/cpp-porting` branch
5. **Easy Rollback**: Simple branch switch to revert to Python-only version

### Topic Publishing Strategy:
- **IMPORTANT**: C++ nodes will publish to the **EXACT SAME** topics as Python nodes
- For imu_gps_publishers replacement:
  - `/ouster/imu`
  - `/ublox_gps_node/fix`
  - `/ublox_gps_node/fix_velocity`
- For dummy_publisher_node replacement (simulation):
  - `/ouster/imu_sim`
  - `/ublox_gps_node/fix_sim`
  - `/ublox_gps_node/fix_velocity_sim`
  - `/odom_sim`
  - `/fused_sorted_cones_ukf_sim`
- **Strategy**: Use launch file arguments to switch between Python and C++ implementations

## 2. Module Analysis and Porting Strategy

### 2.1 cone_definitions.py → cone_definitions.hpp/cpp
- **Purpose**: Defines ground truth cone positions for two scenarios
- **Key features**:
  - Scenario 1: Straight track with AEB zone (blue/yellow/red cones)
  - Scenario 2: Complex Formula Student track with curves
- **C++ considerations**:
  - Use `std::map<int, Cone>` for cone storage
  - Create `Cone` struct with position (Eigen::Vector2d) and type (enum)
  - Implement arc/line generation functions using Eigen

### 2.2 motion_controller.py → motion_controller.hpp/cpp
- **Purpose**: Simulates vehicle motion along predefined trajectories
- **Key features**:
  - Continuous spline-based trajectories
  - Two motion scenarios (straight track, Formula Student circuit)
  - Arc-length parameterization for smooth motion
- **C++ dependencies**:
  - Replace scipy.interpolate with:
    - Option 1: Eigen::Spline (limited functionality)
    - Option 2: GSL (GNU Scientific Library) for spline interpolation
    - Option 3: Custom cubic spline implementation
  - Use `std::vector<Eigen::Vector2d>` for waypoints

### 2.3 sensor_simulator.py → sensor_simulator.hpp/cpp
- **Purpose**: Simulates IMU, GPS, and odometry with realistic noise
- **Key features**:
  - Allan variance-based IMU noise model
  - RTK GPS simulation with different fix modes
  - Odometry drift simulation
- **C++ considerations**:
  - Use `std::random` for noise generation
  - Implement Allan variance models
  - Create separate classes for each sensor type

### 2.4 sensor_simulator_enhanced.py → enhanced_sensor_simulator.hpp/cpp
- **Purpose**: Advanced sensor simulation with temperature effects and misalignment
- **Key features**:
  - Temperature-dependent IMU drift
  - Scale factor errors and axis misalignment
  - RTK status transitions
  - UTM/WGS84 conversion
- **C++ dependencies**:
  - GeographicLib for UTM conversions (or custom implementation)
  - More complex noise models with state tracking

### 2.5 dummy_publisher_node.py → dummy_publisher_node.cpp
- **Purpose**: Main ROS2 node publishing all simulated data
- **Key features**:
  - Manages all sensor simulators
  - Publishes cones, IMU, GPS, odometry
  - Handles TF transforms
  - Visualization markers
- **C++ implementation**:
  - Use `rclcpp::Node` base class
  - Multiple publishers and timers
  - Parameter handling with YAML config

### 2.6 imu_gps_publishers.py → imu_gps_publishers.cpp
- **Purpose**: Standalone IMU/GPS publisher for testing
- **Key features**:
  - Various motion profiles (circular, figure-8, straight)
  - Realistic sensor data matching real hardware format
- **C++ considerations**:
  - Simpler than dummy_publisher_node
  - Good starting point for testing

### 2.7 gps_to_cartesian.py → gps_to_cartesian.cpp
- **Purpose**: Converts GPS lat/lon to local Cartesian coordinates
- **Key features**:
  - UTM conversion
  - Publishing odometry from GPS
  - TF broadcasting
- **C++ dependencies**:
  - GeographicLib or custom UTM implementation

## 3. ROS2 Topics and Message Types

### Input Topics:
- `/fused_sorted_cones_ukf_sim` (custom_interface/TrackedConeArray)
- `/odom` (nav_msgs/Odometry)
- `/ouster/imu` (sensor_msgs/Imu)
- `/ublox_gps_node/fix` (sensor_msgs/NavSatFix)
- `/ublox_gps_node/fix_velocity` (geometry_msgs/TwistWithCovarianceStamped)

### Output Topics:
- `/slam/landmarks` (visualization_msgs/MarkerArray)
- `/slam/factor_graph` (visualization_msgs/MarkerArray)
- `/slam/pose` (geometry_msgs/PoseStamped)
- `/slam/path` (nav_msgs/Path)
- `/gps/odometry` (nav_msgs/Odometry)

## 4. Key Dependencies to Handle

1. **NumPy → Eigen**:
   - Arrays → Eigen::VectorXd, Eigen::MatrixXd
   - Mathematical operations are similar

2. **scipy.interpolate → Spline library**:
   - Most critical dependency
   - Options: GSL, Eigen::Spline, or custom implementation

3. **utm → GeographicLib**:
   - For GPS coordinate conversions
   - Well-maintained C++ library

4. **Python-specific features**:
   - List comprehensions → std::transform or loops
   - Dictionaries → std::unordered_map
   - Dynamic typing → explicit types
   - Global variables → class members or namespaces

## 5. Implementation Priority

### Phase 1 - Core Data Structures
- Cone struct/class
- VehicleState struct
- Sensor noise configurations
- Basic mathematical utilities

### Phase 2 - Sensor Simulators
- Basic sensor_simulator (IMU, GPS, Odometry classes)
- Noise models implementation
- State tracking for drift

### Phase 3 - Motion Control
- Trajectory generation
- Spline interpolation
- Motion scenarios

### Phase 4 - ROS2 Integration
- dummy_publisher_node
- Parameter handling
- Timer callbacks
- TF broadcasting

### Phase 5 - Enhanced Features
- Enhanced sensor simulator
- GPS to Cartesian converter
- Visualization utilities

## 6. Testing Strategy

### Unit Tests
- Sensor noise models
- Coordinate conversions
- Trajectory generation

### Integration Tests
- Compare C++ output with Python output
- Use same random seeds for verification
- Check ROS2 message compatibility

### System Tests
- Full simulation with SLAM system
- Performance benchmarking
- Real-time capability verification

## 7. Build System Configuration

```cmake
# CMakeLists.txt additions
find_package(Eigen3 REQUIRED)
find_package(GeographicLib REQUIRED)
find_package(GSL)  # Optional for splines

# Add libraries
add_library(cone_stellation_simulation
  src/simulation/cone_definitions.cpp
  src/simulation/motion_controller.cpp
  src/simulation/sensor_simulator.cpp
  # ...
)

# Add executables
add_executable(dummy_publisher_node
  src/nodes/dummy_publisher_node.cpp
)
```

## 8. Specific Challenges and Solutions

### Spline Interpolation
- **Challenge**: No direct scipy equivalent
- **Solution**: Implement cubic spline class or use GSL

### Dynamic Parameters
- **Challenge**: Python's flexible parameter handling
- **Solution**: Use ROS2 parameter callbacks and structured configs

### Random Number Generation
- **Challenge**: Matching Python's numpy.random behavior
- **Solution**: Use same algorithms (e.g., Mersenne Twister) with same seeds

### Visualization
- **Challenge**: Complex marker generation
- **Solution**: Create utility functions for common marker types

## 9. Non-Invasive File Structure

```
cone_stellation/
├── scripts/                     # UNCHANGED - Python files remain here
│   ├── cone_definitions.py
│   ├── dummy_publisher_node.py
│   └── ...
├── include/cone_stellation/     # NEW - C++ headers
│   └── simulation/
│       ├── cone_definitions.hpp
│       ├── motion_controller.hpp
│       ├── sensor_simulator.hpp
│       ├── enhanced_sensor_simulator.hpp
│       ├── spline.hpp
│       ├── noise_models.hpp
│       └── coordinate_converter.hpp
├── src/                         # NEW - C++ implementation
│   └── simulation/
│       ├── cone_definitions.cpp
│       ├── motion_controller.cpp
│       ├── sensor_simulator.cpp
│       ├── enhanced_sensor_simulator.cpp
│       ├── dummy_publisher_node.cpp      # Node file with _node suffix
│       ├── imu_gps_publishers_node.cpp   # Node file with _node suffix
│       └── gps_to_cartesian_node.cpp     # Node file with _node suffix
├── test/                        # NEW - C++ tests
│   ├── unit/
│   ├── integration/
│   └── comparison/              # Compare Python vs C++ outputs
└── launch/                      # MODIFIED - Add C++ launch files
    ├── existing_python_launches.py
    ├── dummy_publisher_cpp.launch.py     # NEW
    └── comparison_test.launch.py         # NEW - Run both versions
```

## 10. Git Branch Strategy

### Branch Creation:
```bash
git checkout -b feature/cpp-porting
```

### Development Rules:
1. All C++ code commits to `feature/cpp-porting` only
2. No modifications to Python files in scripts/
3. Regular rebasing from main to stay current
4. Clear commit messages: "[CPP] Added sensor_simulator implementation"

### Rollback Strategy:
```bash
# Simply switch back to main if issues arise
git checkout main
# Python implementation remains untouched
```

## 11. CMakeLists.txt Modifications

```cmake
# Add C++ compilation ONLY if explicitly enabled
option(BUILD_CPP_NODES "Build C++ version of nodes" OFF)

if(BUILD_CPP_NODES)
  find_package(Eigen3 REQUIRED)
  find_package(GeographicLib REQUIRED)
  
  # C++ library
  add_library(${PROJECT_NAME}_cpp
    src/simulation/cone_definitions.cpp
    src/simulation/motion_controller.cpp
    src/simulation/sensor_simulator.cpp
  )
  
  # C++ nodes with _cpp suffix
  add_executable(dummy_publisher_cpp_node
    src/simulation/dummy_publisher_node.cpp
  )
  
  # Install with different names
  install(TARGETS dummy_publisher_cpp_node
    DESTINATION lib/${PROJECT_NAME}
  )
endif()
```

## 12. Launch File Strategy

### Modified approach - Use arguments to switch implementations:
```python
# imu_gps_ekf_launch.py - MODIFIED to support both
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_cpp = LaunchConfiguration('use_cpp', default='false')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_cpp',
            default_value='false',
            description='Use C++ implementation instead of Python'
        ),
        
        # Conditional node selection
        Node(
            package='cone_stellation',
            executable=PythonExpression([
                "'imu_gps_publishers_cpp_node' if '", use_cpp, "' == 'true' else 'imu_gps_publishers.py'"
            ]),
            name='imu_gps_publisher',
            output='screen'
        )
    ])
```

### Usage:
```bash
# Run Python version (default)
ros2 launch cone_stellation imu_gps_ekf_launch.py

# Run C++ version
ros2 launch cone_stellation imu_gps_ekf_launch.py use_cpp:=true
```

## 13. Testing Strategy for Non-Invasive Implementation

### Phase 1 - Isolated C++ Testing:
1. Build only C++ components: `colcon build --cmake-args -DBUILD_CPP_NODES=ON`
2. Run C++ nodes independently
3. Verify basic functionality

### Phase 2 - Side-by-Side Comparison:
1. Run both Python and C++ versions simultaneously
2. Record topics from both: `ros2 bag record /dummy/* /dummy_cpp/*`
3. Compare outputs with analysis scripts

### Phase 3 - Performance Benchmarking:
1. CPU/Memory usage comparison
2. Message publishing frequency stability
3. Latency measurements

## 14. Implementation Priority (Revised)

### Milestone 1 - Minimal Viable Node:
1. Basic dummy_publisher_cpp_node that publishes constant data
2. Verify CMake build and installation
3. Confirm no interference with Python nodes

### Milestone 2 - Core Functionality:
1. Port cone_definitions (static data)
2. Basic sensor_simulator (without splines)
3. Simple circular motion for testing

### Milestone 3 - Full Feature Parity:
1. Spline-based motion controller
2. Enhanced sensor simulator
3. All noise models

### Milestone 4 - Integration & Optimization:
1. Parameter server integration
2. Performance optimization
3. Complete test coverage

## 15. Specific Non-Invasive Guidelines

### DO:
- ✅ Keep all C++ code in separate directories
- ✅ Use exact same topic names as Python nodes
- ✅ Make C++ building optional via CMake flag
- ✅ Add launch argument to switch between implementations
- ✅ Document differences between implementations
- ✅ Test both versions produce identical outputs

### DON'T:
- ❌ Modify any existing Python files
- ❌ Change topic names or message formats
- ❌ Alter package.xml dependencies for Python nodes
- ❌ Mix C++ and Python code in same files
- ❌ Create dependencies between C++ and Python implementations
- ❌ Use different topic namespaces (they must be identical!)

This non-invasive approach ensures zero risk to the existing system while allowing progressive C++ implementation and testing.
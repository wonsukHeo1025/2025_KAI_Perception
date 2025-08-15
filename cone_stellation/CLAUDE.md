# Path: /home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/CLAUDE.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Global Task Execution Guidelines
- Before executing any command for any task, first scan the entire directory structure of the workspace or the target package and the list of existing files.
- Understand the functionality of existing files. When a new task is requested, refrain from creating new files unless absolutely necessary, i.e., only create new files if adding to or modifying existing ones would compromise modularity.
- In whichever directory you are working, update the CLAUDE.md file to reflect the current status ALWAYS, WHENEVER YOU DO ANYTHING. It does not need to be overly specific.
- Furthermore, just as you continuously update markdown files, if the user raises an issue or if you discover and correct an error during analysis, briefly summarize the debugged items, the problem, the solution, and the result (e.g., resolved, new issue occurred). Continuously append this summary with a timestamp to a debug_log.md file in the package path. To save input tokens, do not read the entire file; just append new entries to the end when updates are needed.
- Think.
- Think and conduct all tasks in English, and use Korean only after all the tasks are done and add Korean summaries, only then you can use Korean in order to save tokens. With that being said, you can utilize more marginal token for thinking and fluent coding. 
- When building with ROS2, run colcon build from the workspace root, and use the --symlink-install option whenever possible.
- In whichever directory you are working, update the CLAUDE.md file to reflect the current status. It does not need to be overly specific.
- When the user issues a command related to activating Gemini MCP, if the request pertains to code content, the relevant parts of the code must be provided directly to Gemini as part of the prompt. This is because the Gemini connected to the Claude code's MCP cannot utilize its MCP to read files on its own.

---

## Local Task Execution Guidelines
- GLIM_ws/ directory should be what you are gonna refer to whenever you develop. Based on our specific situation, you will proceed with coding, striving to modify the GLIM_ws codebase as much as possible.
- To ensure this, the actual sensor topic inputs are specified in input_topic_structure.md. Testing the SLAM logic should be done by creating test logs, not by creating separate test files or launch files. 
- The simulator in cc_slam_sym functions by generating ground truth cone and odometry data, as well as sensor measurements. Therefore, the SLAM logic must operate independently of it. This means the SLAM logic must behave identically regardless of the input source, whether it's the dummy publisher simulator, a rosbag, or live sensor data.
- All documents, such as prd.md file should be placed in docs/ folder, except readme.md and claude.md.  

## Package Overview

ConeSTELLATION (Cone-based STructural ELement Layout for Autonomous NavigaTION) is a cone-based Graph SLAM system inspired by GLIM's modular architecture. It processes cone detection instances from LiDAR clustering with YOLO-based color classification instead of raw point clouds.

## Current Status

- **Created**: 2025-07-18
- **Updated**: 2025-07-28 (IMU-GPS EKF fusion implemented)
- **Status**: SLAM Working Well with Inter-landmark Factors AND Loop Closure! Full SLAM pipeline operational.
- **Architecture**: Based on GLIM's proven modular design with novel inter-landmark factors
- **Key Decision**: Use external IMU+GPS odometry, SLAM for mapping only (like GLIM)
- **Active Development**: IMU-GPS EKF fusion ready for testing
- **Latest Updates**: 
  - ✅ Data association working excellently with minimal overlapping landmarks
  - ✅ Noise filtering successfully blocks false positives/negatives
  - ✅ Factor graph properly constructed with pose nodes and observation edges
  - ✅ Real-time backend optimization smoothly adjusting poses and map
  - ✅ Track ID properly utilized in data association
  - ✅ Clean visualization without orphan nodes
  - ✅ Odometry architecture decision made (IMU+GPS for control, SLAM for correction)
  - ✅ Drift correction implemented! map->odom transform now updates based on SLAM optimization
  - ✅ DriftCorrectionManager with pose interpolation (GLIM-inspired)
  - ✅ Inter-landmark factors NOW WORKING! Co-observation tracking fixed
  - ✅ Circular track shapes better maintained with inter-landmark constraints
  - ✅ 하이브리드 아키텍처 결정: 외부 EKF (100Hz) + SLAM 맵핑 (20Hz)
  - ✅ map->odom 동적 보정의 작동 원리 이해 완료
  - ⏸️ Fixed-lag smoother POSTPONED (랜드마크 SLAM에 부적합)
  - ✅ Loop closure ENHANCED for sparse environments! (2025-07-22)
    - Constellation-based recognition (reduced to 3+ cones)
    - Path-based loop detection with curvature profiles
    - Geometric feature detection (turns, transitions, chicanes)
    - Combined scoring: 30% cones, 30% path, 40% features
    - Purple visualization for loop closure factors
  - ✅ Loop closure segfault FIXED! (2025-07-23)
    - Missing LoopClosureDetector function implementations added
    - Full RANSAC-based geometric verification implemented
    - Deadlock fixed in find_candidates() function
    - Build successful, ready for testing
  - ✅ RViz performance optimized! (2025-07-23)
    - Added marker lifetimes to factor graph visualization
    - Implemented sliding window: shows most recent N factors of each type
    - Limits: 100 obs, 200 odom, 50 inter, 20 loop factors
    - Only delete markers every 30s instead of every frame
    - .gitignore updated to exclude build artifacts from package
  - ✅ Inter-landmark factors IMPROVED! (2025-07-23)
    - Implemented clustering algorithm to group co-observed landmarks
    - Factors now created between cluster representatives instead of all pairs
    - Adaptive noise model based on distance
    - Stricter parameters: min_covisibility=3, min_distance=1.5m
    - Max 15 factors per frame to avoid over-constraining
  - ✅ False positive filtering ENHANCED! (2025-07-23)
    - Added outlier rejection to tentative landmarks (2 std dev threshold)
    - Tightened noise models: observation=0.3, inter-landmark=0.05
    - Reduced max association distance to 1.5m
    - Increased color confidence requirement to 0.8
    - Tentative landmarks now reject observations too far from mean
  - ✅ System stabilized (2025-07-23)
    - Data association improved with strict color matching
    - Tentative landmark parameters tightened
    - Complex clustering removed for stability
    - Loop closure temporarily disabled
  - ✅ IMU-GPS EKF Fusion Implemented! (2025-07-28)
    - Created realistic IMU/GPS publishers matching exact topic formats
    - GPS to local Cartesian converter with UTM transformation
    - robot_localization EKF configuration for 100Hz fusion
    - Launch file for complete system integration
    - Ready for testing with multiple motion profiles
    - EKF-only launch ready for real bag file data with proper coordinate system
    - Added GPS converter node with Konkuk University coordinates (37.540091°N, 127.076555°E)
    - Complete TF tree structure: map → odom → base_link → sensors
  - ✅ Documentation Consolidated (2025-07-28)
    - Created comprehensive PRD.md (Product Requirements Document)
    - Consolidated DEVELOPMENT_PLAN.md with all technical details
    - Removed redundant documentation files
    - Maintained debug_log.md for incremental issue tracking
  - ⏳ 다음 목표: 
    - Test IMU-GPS EKF fusion with SLAM integration
    - GTSAM IMU preintegration factors (future enhancement)
    - Sparse rigid body inter-landmark constraints
    - Stable loop closure reimplementation

## Project Structure (Current)

```
cone_stellation/
├── include/cone_stellation/    # Public headers (header-only design)
│   ├── common/                 # Core data structures (cone.hpp, estimation_frame.hpp, tentative_landmark.hpp)
│   ├── factors/                # Custom GTSAM factors (cone_observation_factor.hpp, inter_landmark_factors.hpp)
│   ├── mapping/                # SLAM mapping modules (cone_mapping.hpp, cone_mapping_safe.hpp, data_association.hpp, loop_closure_detector.hpp, simple_cone_mapping.hpp)
│   ├── odometry/               # Odometry estimation modules (cone_odometry_base.hpp, cone_odometry_2d.hpp, async_cone_odometry.hpp)
│   ├── preprocessing/          # Cone data preprocessing (cone_preprocessor.hpp)
│   ├── util/                   # ROS2 utilities (ros_utils.hpp, drift_correction_manager.hpp)
│   └── viewer/                 # Visualization components (multiple viewer modules)
├── src/                        # Implementation files
│   ├── cone_stellation/        # Main source directory
│   │   ├── ros/                # ROS2 node implementation (cone_slam_node.cpp)
│   │   ├── factors/            # Factor implementations (inter_landmark_factors.cpp)
│   │   └── preprocessing/      # Preprocessing implementations (cone_preprocessor.cpp)
│   └── loop_closure_detector.cpp # Loop closure implementation
├── config/                     # YAML configuration files
│   ├── slam_config.yaml       # SLAM parameters
│   ├── dummy_publisher_config.yaml # Simulation parameters
│   └── cone_slam.rviz         # RViz visualization config
├── scripts/                    # Python simulation scripts
│   ├── tests/                  # Test scripts directory
│   │   ├── check_topics.py     # Topic monitoring
│   │   ├── test_dummy_publisher.py # Basic testing
│   │   ├── test_imu_gps_fusion.py # IMU-GPS fusion testing
│   │   ├── test_loop_closure.py # Loop closure testing
│   │   └── test_slam_only.py   # SLAM-only testing
│   ├── dummy_publisher_node.py # Dummy cone publisher for testing
│   ├── sensor_simulator.py     # Basic sensor simulation
│   ├── sensor_simulator_enhanced.py # Enhanced sensor models
│   ├── motion_controller.py    # Vehicle motion control
│   ├── cone_definitions.py     # Cone type definitions
│   └── imu_gps_publishers.py   # IMU/GPS data publishers
├── launch/                     # ROS2 launch files
│   ├── slam_only_launch.py    # SLAM-only launch
│   ├── dummy_publisher_launch.py # Simulation launch
│   └── test_slam_launch.py    # Combined test launch
├── docs/                       # Documentation
│   ├── DEVELOPMENT_PLAN.md    # Development roadmap
│   ├── PRD.md                 # Product requirements
│   ├── debug_log.md           # Debug history
│   └── (other docs)           # Various technical documents
└── rqt_log/                   # RQT logging data
    └── logging.csv            # Log data file
```

## Key Design Decisions

1. **Modular Architecture**: Following GLIM's plugin-based system
2. **Data Flow**: Cone Detection → Preprocessing → Association → Mapping → Optimization
3. **Base Classes**: Abstract interfaces for each module type
4. **Configuration**: JSON-based hierarchical configuration
5. **Threading**: Asynchronous wrappers for real-time performance
6. **Tentative Landmark System**: Observation buffering before landmark creation
   - Prevents false positives from noise
   - Ensures landmarks are well-constrained
   - Color voting for robust classification
   - Track ID hysteresis for occlusion handling

## Development Phases

### Completed Phases ✅
1. **Basic Infrastructure**: Core data structures, simulation, visualization
2. **Core Modules**: ConePreprocessor, ConeMapping with ISAM2
3. **GTSAM Integration**: Custom factors (inter-landmark, observation)
4. **ROS2 Integration**: Basic node, TF, config, visualization
5. **Tentative Landmarks**: Observation buffering, color voting, promotion criteria

### Current Development (Detailed in docs/)
- **Completed**: Drift correction (map->odom tf calculation) ✅
- **In Progress**: Loop closure implementation based on GLIM's approach
  - GLIM uses implicit loop detection via proximity and overlap
  - Manual loop closure also available via interactive GUI
  - Uses FPFH features and global registration for initial alignment
- **Next Priority**: Odometry/mapping separation for 100Hz vehicle control
- **Phase 3**: Enable and tune inter-landmark factors
- **Phase 4**: IMU/GPS integration for robust multi-sensor fusion
- **Phase 5**: Loop closure with cone constellations
- **Phase 6**: ~~Fixed-lag smoother~~ POSTPONED - focus on loop closure instead

### GLIM Features Integration
See `glim_features_integration.md` for detailed roadmap including:
- Multi-threading architecture
- Memory management strategies
- Robust optimization techniques
- Configuration management
- Serialization and recovery

### Implementation Status
See `implementation_status.md` for:
- ✅ Already implemented features
- ❌ Not yet implemented features
- Performance targets and metrics
- Testing requirements

## Dependencies

- **Required**: Eigen, GTSAM, gtsam_points, spdlog, Boost
- **ROS2**: geometry_msgs, nav_msgs, sensor_msgs, tf2
- **Optional**: OpenCV (for visualization), CUDA (future GPU acceleration)

## Key Differences from cc_slam_sym

1. **No SymForce**: Direct GTSAM integration without intermediate code generation
2. **Novel Inter-landmark Factors**: Key innovation for sparse cone observations
   - ConeDistanceFactor: Maintains relative distances between co-observed cones
   - ConeLineFactor: Enforces collinearity for cones on track boundaries
   - Pattern-based factors: Leverages geometric patterns in cone layouts
3. **Header-only Design**: Following GLIM's approach for flexibility
4. **Proven Architecture**: Based on GLIM's successful design patterns

## References

- GLIM architecture: `/home/user1/ROS2_Workspace/GLIM_ws/src/glim/`
- Development plan: `DEVELOPMENT_PLAN.md`

---

## 2025-08-08 — Status Note
- Performed cone mapping robustness analysis under sensor shake; created `docs/cone_mapping_robustness_analysis.md` with prioritized fixes (gating by motion, remove early direct landmark creation, adaptive per-observation noise with robust loss, stronger association, inter-landmark factors from same-frame geometry, simple smoothing in preprocessing).
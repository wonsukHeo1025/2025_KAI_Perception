# include/ Directory

This directory contains all public headers for the cone_stellation SLAM system, following a header-only design pattern inspired by GLIM.

## Structure

- **cone_stellation/**: Main namespace directory containing all headers
  - **common/**: Core data structures (Cone, EstimationFrame, etc.)
  - **factors/**: GTSAM custom factors for cone SLAM
  - **mapping/**: Mapping and optimization modules
  - **odometry/**: Odometry estimation modules (TO BE IMPLEMENTED)
  - **preprocessing/**: Cone data preprocessing and filtering
  - **util/**: ROS2 utilities and helpers
  - **viewer/**: Visualization components (TO BE IMPLEMENTED)

## Design Philosophy

Following GLIM's approach, most implementations are header-only for:
- Template flexibility
- Easier integration
- No need for complex linking
- Inline optimization opportunities

## Current Status (2025-07-20)

### Recent Updates
- **cone.hpp**: ✅ FIXED - Co-observation tracking now properly counts observations
  - Added co_observation_counts_ map for actual counting
  - Fixed bug where co_observation_count() only returned 0 or 1
- **mapping/**: ✅ Inter-landmark factors NOW WORKING!
  - Distance factors created between co-observed landmarks
  - Helps maintain track shape (especially curves)
  - Visualization shows red lines for inter-landmark constraints
- **data_association.hpp**: Basic data association module
  - Nearest neighbor matching with color constraints
  - Track ID support integrated
- **odometry/**: Cone-based odometry modules
  - cone_odometry_2d.hpp: 2D implementation using GTSAM
  - async_cone_odometry.hpp: Asynchronous wrapper
- **viewer/**: Separated visualization modules following GLIM architecture
- **util/drift_correction_manager.hpp**: Calculates map->odom transform

### Current Architecture
- Cone observations → Odometry estimation → Mapping with inter-landmark factors
- Drift correction properly calculates map->odom transform
- Real-time optimization with ISAM2
- Track ID utilized for robust data association

### Next Steps
- Implement pattern detection (line, curve, parallel lines)
- Add IMU/GPS integration for high-rate odometry
- Implement loop closure detection
- Performance optimization for larger maps
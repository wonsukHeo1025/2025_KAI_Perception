
# Simple SLAM Project

## Preliminaries

This document outlines the requirements and design for the LiDAR-IMU Odometry (LIO) system development.

### 1. Goals

*   Fuse the high-frequency dynamic response of IMU with the low-drift characteristics of LiDAR (using cone landmarks) to achieve accurate and robust 6-DoF Odometry estimation.
*   Utilize existing cone detection/tracking modules (`cone_detection_node`, `cone_tracker_ukf`). Leverage tracked cone landmarks with known associations (via Track IDs) for efficient LiDAR updates.
*   Mitigate the cumulative drift issue inherent in IMU-only Odometry.
*   Design an extensible architecture (considering future fusion with GPS, etc.).

### 2. Key Input/Output Topics and Data (Revised)

*   **Input:**
    *   `/ouster/imu` ([sensor_msgs/msg/Imu](https://docs.ros2.org/foxy/api/sensor_msgs/msg/Imu.html)): 6-axis IMU data (angular velocity, linear acceleration). The `orientation` field is unusable (`covariance == -1.0`). `frame_id: os_imu`.
    *   `/fused_sorted_cones_ukf` (**[custom_interface/msg/TrackedConeArray]**): Array of tracked and filtered cone information from the UKF tracker. Each cone includes its **Track ID**, 3D position, and estimated color. (Requires prerequisite implementation of new message types and modification of `cone_tracker_ukf`).
      ```protobuf
      # custom_interface/msg/TrackedCone.msg
      int32 track_id          # Unique identifier for the tracked cone
      geometry_msgs/Point position # 3D position in os_sensor frame (x, y, z - float64)
      string color            # Estimated color name (e.g., "Blue cone", "Unknown")

      # custom_interface/msg/TrackedConeArray.msg
      std_msgs/Header header     # Timestamp and frame_id (os_sensor)
      custom_interface/msg/TrackedCone[] cones # Array of tracked cones
      ```
*   **(Reference) Raw Cone Detection:**
    *   `/sorted_cones_time` ([custom_interface/msg/ModifiedFloat32MultiArray](https://github.com/user/custom_interface)): Cone coordinates detected by `cone_detection_node` (`os_sensor` frame). Not directly used by the LIO node but is input to the tracker.
*   **Output (LIO_Node):**
    *   `/odom_lio` ([nav_msgs/msg/Odometry](https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html)): Estimated Odometry (position/orientation/velocity of `os_sensor` relative to `odom` frame).
    *   `/tf`: Publishes the `odom` -> `os_sensor` transform and the `map` -> `odom` transform.
    *   `/mapped_cones_markers` ([visualization_msgs/msg/MarkerArray](https://docs.ros2.org/foxy/api/visualization_msgs/msg/MarkerArray.html)): Visualization of mapped cone landmarks (`map` frame).

### 3. TF Tree Structure

```
map (World Fixed)
└── odom (Drift is possible)
    └── os_sensor (Target frame estimated by LIO)
        ├── os_lidar (Static TF)
        └── os_imu (Static TF)
```

*   `os_sensor` -> `os_lidar` / `os_imu`: Static transforms. Utilize values from existing nodes (`imu_odometry_node`, `cone_mapper_node`).
*   `odom` -> `os_sensor`: The core transform estimated and published in real-time by the LIO node.
*   `map` -> `odom`: Transform updated during drift correction. Initialized to Identity.

### 4. Core Requirements (Revised)

*   **Data Association:** Perform **direct and robust** association between observed cones and internal map landmarks using the **`track_id`** provided in the `/fused_sorted_cones_ukf` (**`TrackedConeArray`**) message.
*   **IMU Processing:** Account for the lack of `orientation` in the input IMU data during LIO filter design (requires an explicit initialization phase).
*   **Kalman Filter Based:** Implement LIO estimation using a UKF. The state vector must include at least `[pose, velocity, imu_biases]`.
*   **Message Type Prerequisite:** The definition and implementation of `TrackedCone.msg` and `TrackedConeArray.msg` in the `custom_interface` package, and the modification of `cone_tracker_ukf` to use this new type, are **mandatory prerequisites** for LIO development.

---

## Overall Design Flow Summary (Final - TrackID Based)

This section summarizes the final phased development approach for the Cone-LIO system, assuming Track IDs are available via the new message types.

0.  **[Step 0] Prerequisite: Custom Message Types:** Define and implement `TrackedCone.msg` and `TrackedConeArray.msg` in the `custom_interface` package. Modify `cone_tracker_ukf` to publish this new type. *Key: Enabling Robust Association.*
1.  **[Step 1] Basic Node Setup & Sync:** Establish the LIO node structure with mandatory time synchronization (`message_filters`) for `/ouster/imu` and the *new* `/fused_sorted_cones_ukf` (`TrackedConeArray`) topics. *Key: Foundation & Sync.*
2.  **[Step 2] Static TF Publishing:** Publish fixed sensor geometry. *Key: Sensor Geometry.*
3.  **[Step 3] KF Init & Explicit State Init:** Define state/covariance and implement an explicit initialization phase (gravity, orientation, biases) with quality verification. *Key: Filter State & Stable Start.*
4.  **[Step 4] IMU Prediction (Stabilized):** Implement KF prediction with robust gravity/bias compensation. Publish predicted TF. *Key: Stabilized Motion Propagation.*
5.  **[Step 5] Robust Map Management:** Implement map structure with criteria for adding/refining landmarks (using Track ID). *Key: Reliable Landmark Storage.*
6.  **[Step 6] LiDAR Update - Association (TrackID Based):** Implement data association by directly matching `track_id` from observations to map landmarks. **Proceed only if >= 2-3 known landmarks are observed.** *Key: Simple & Robust Matching.*
7.  **[Step 7] LiDAR Update - SVD Pose & Quality Check:** Calculate relative pose via SVD on associated landmarks and **verify SVD result stability** (geometric configuration). *Key: Geometric Measurement & Stability Check.*
8.  **[Step 8] LiDAR Update - KF Correction & Drift Comp:** Correct KF state using SVD pose (if stable) and explicitly compute/apply `map` -> `odom` drift correction (smoothed). *Key: State Correction & Drift Compensation.*
9.  **[Step 9] Results Publishing & Refined Map Update:** Publish corrected TFs, Odometry, and map markers. Update map based on high-quality (stable SVD) measurements. *Key: Final Output & Refined Map.*
10. **[Step 10] Integration & Tuning:** Integrate, test, and tune, focusing on initialization, SVD stability, drift correction, and map quality. *Key: System Validation & Optimization.*

## Detailed Development Plan (Final - TrackID Based)

This section provides the final step-by-step guide, assuming Track IDs are available via new message types and adhering to KISS principles.

### **[Step 0] Prerequisite: Implement Custom Message Types**

*   **Action:** In the `custom_interface` package:
    *   Create `msg/TrackedCone.msg` (fields: `int32 track_id`, `geometry_msgs/Point position`, `string color`).
    *   Create `msg/TrackedConeArray.msg` (fields: `std_msgs/Header header`, `custom_interface/msg/TrackedCone[] cones`).
    *   Modify `CMakeLists.txt` and `package.xml` to include these new message files and depend on `geometry_msgs`.
*   **Action:** Build the `custom_interface` package (`colcon build --packages-select custom_interface`).
*   **Action:** Modify `kalman_filtering.py` (`cone_tracker_ukf` node):
    *   Change the publisher for `/fused_sorted_cones_ukf` to use `TrackedConeArray`.
    *   In `listener_callback`, create a `TrackedConeArray` message. Populate the `cones` array by iterating through `self.tracks`, filling `track_id`, `position` (from `get_predicted_position_xyz()`), and `color` (from `get_smoothed_color()`) for each track.
*   **Goal:** Ensure the `cone_tracker_ukf` node publishes tracked cones with explicit Track IDs using the new message types. **This step must be completed before starting LIO node development.**

### [Step 1] `LIO_Node` Basic Skeleton and Environment Setup

*   **Action:** Create `simple_slam/lio_node.py`.
*   **Action:** Update dependencies (`package.xml`, `setup.py` - ensure `custom_interface` is included).
*   **Action:** Define `LIO_Node` class.
*   **Action:** Implement `__init__`:
    *   Declare parameters.
    *   **Mandatory:** Use `message_filters.ApproximateTimeSynchronizer` to subscribe to `/ouster/imu` (Imu) and `/fused_sorted_cones_ukf` (**TrackedConeArray**). Register a single synchronized callback.
    *   Create Publishers (`/odom_lio`, `/mapped_cones_markers`).
    *   Setup TF broadcasters/listeners.
*   **Goal:** Establish LIO node structure with guaranteed input synchronization using the new message type.

### [Step 2] Static Transform Publishing

*   **Action:** In `LIO_Node.__init__`, retrieve static transform parameters (`os_sensor` to `os_imu`, `os_sensor` to `os_lidar`).
*   **Action:** Convert parameters to `geometry_msgs/msg/TransformStamped` messages.
*   **Action:** Use `StaticTransformBroadcaster.sendTransform()` to publish these transforms once at node startup.
*   **Goal:** Ensure the fixed geometric relationship between sensors is available in the TF tree.

### [Step 3] Kalman Filter State and Initialization

*   **Action:** Define the state vector `x` (e.g., pose, velocity, biases) and covariance matrix `P`. Define process noise `Q` and measurement noise `R` matrices.
*   **Action:** Implement an **explicit initialization phase** with **quality checks**:
    *   Add a node state (e.g., `INITIALIZING`, `RUNNING`).
    *   During `INITIALIZING` (assuming static start): Estimate initial gravity vector, biases, and orientation. Initialize `self.x` and `self.P`.
    *   **Verify initialization quality** (e.g., check gravity magnitude, bias stability). Log warning or pause if checks fail (avoid complex dynamic init for KISS).
    *   Transition to `RUNNING` after successful verification.
*   **Action:** Initialize `self.map_to_odom_transform` as identity and publish.
*   **Action:** Configure UKF implementation (e.g., `filterpy`).
*   **Goal:** Set up KF variables and ensure a **verified stable initial state**.

### [Step 4] IMU Prediction Step Implementation

*   **Action:** Implement the `imu_callback` or logic within the synchronized callback (triggered only in `RUNNING` state).
*   **Action:** Calculate `dt` accurately. Extract `omega_imu` and `accel_imu`.
*   **Action:** Implement bias correction using current state `self.x`.
*   **Action:** Implement **stabilized gravity compensation** using the verified initial gravity estimate and current predicted/corrected orientation.
*   **Action:** Define the UKF state transition function `fx(...)` for state propagation.
*   **Action:** Call `self.ukf.predict(...)`.
*   **Action:** Publish the predicted `odom` -> `os_sensor` TF.
*   **Goal:** Propagate state using IMU, relying on a stable gravity estimate and accurate timing.

### [Step 5] Map Management Implementation

*   **Action:** Initialize `self.map_landmarks = {}` where keys are `track_id`. Value could store `{'pos_map': np.array([x, y, z]), 'color': "color_string", ...}`.
*   **Action:** Implement `add_or_update_landmark(self, track_id, position_map, color)`:
    *   If `track_id` exists, update `pos_map` (e.g., simple overwrite or EMA).
    *   If `track_id` is new, add it. Use simple criteria initially (e.g., add on first sight).
*   **Goal:** Manage landmarks using the reliable `track_id` as the key. Keep update logic simple initially.

### [Step 6] LiDAR Update Step - Association (TrackID Based)

*   **Action:** Triggered by the synchronized callback when new `TrackedConeArray` data arrives (only in `RUNNING` state).
*   **Action:** Extract the list of observed cones (`observed_cones`: list of `TrackedCone` objects) from `cones_msg.cones`.
*   **Action:** **SLAM Start/Update Condition:** Check if the number of *known* landmarks among `observed_cones` (i.e., `obs_cone.track_id` exists in `self.map_landmarks`) is `>= MIN_KNOWN_LANDMARKS_FOR_UPDATE` (e.g., 2 or 3). If not, skip LiDAR update (Steps 6-9).
*   **Action:** Perform **direct association using Track ID**:
    *   Initialize `valid_associations = []` (list of `(track_id, observed_cone)`)
    *   For each `obs_cone` in `observed_cones`:
        *   If `obs_cone.track_id` exists as a key in `self.map_landmarks`:
            *   Add `(obs_cone.track_id, obs_cone)` to `valid_associations`.
*   **Goal:** Reliably and simply associate observations using Track IDs. Proceed only if enough *known* landmarks are observed.

### [Step 7] LiDAR Update Step - SVD Pose Calculation

*   **Action:** Check if `len(valid_associations)` >= minimum required for SVD (e.g., 3). If not, skip SVD and subsequent update steps.
*   **Action:** Prepare point sets for SVD and assess stability:
    *   `P_map = [self.map_landmarks[assoc]['pos_map'] for assoc in valid_associations]`
    *   `P_sensor = [np.array([assoc.position.x, assoc.position.y, assoc.position.z]) for assoc in valid_associations]`
    *   Implement SVD calculation to find `T_sensor_map_lidar`.
    *   **Check SVD result stability** (e.g., minimally check point distribution to avoid degenerate cases like collinear points). Store quality flag. Proceed only if stable.
*   **Action:** Invert to get `T_map_sensor_lidar`, extract measured pose `p_map_lidar`, `q_map_lidar`.
*   **Goal:** Compute relative geometric pose from associated points and verify geometric stability.

### [Step 8] LiDAR Update Step - Kalman Filter Correction & Drift Compensation

*   **Action:** **Stabilized Drift Correction Calculation & Application:**
    *   Calculate the discrepancy (`T_error`) between the SVD measurement (`T_map_sensor_meas`) and the filter's prediction (`T_map_sensor_pred`).
    *   **Apply `T_error` to `self.map_to_odom_transform` cautiously only if SVD quality (Step 7) is high.**
    *   Apply the correction smoothly using a **simple low-pass filter or Exponential Moving Average (EMA)** on the components of `T_error`.
*   **Action:** Prepare the measurement vector `z_odom` for the KF by transforming the SVD measurement (`p_map_lidar`, `q_map_lidar`) to the current `odom` frame using the updated `map_to_odom_transform`.
*   **Action:** Define the UKF measurement function `hx(x)` (extracts pose from state). Define measurement noise `R_lidar`.
*   **Action:** Call `self.ukf.update(z_odom, ...)` **only if the SVD quality check passed**.
*   **Goal:** Correct the KF state and stably compensate for drift using only geometrically stable measurements and smoothed updates.

### [Step 9] Results Publishing and Map Update

*   **Action:** Publish the corrected `odom` -> `os_sensor` TF (from corrected `self.x`) and the updated `map` -> `odom` TF (from `self.map_to_odom_transform`).
*   **Action:** Publish the corrected `/odom_lio` message (extracting pose, velocity, covariance from `self.x`, `self.P`).
*   **Action:** Implement map update logic:
    *   For associated cones (`valid_associations`): Call `add_or_update_landmark()` **only if SVD quality (Step 7) was sufficient** for this update cycle. Use the `track_id` and the transformed observed position.
    *   For unassociated cones (`obs_cone` with new `track_id`): Call `add_or_update_landmark()` based on simple criteria (e.g., add new tracks immediately after transforming position to map frame using corrected pose).
*   **Action:** Publish `/mapped_cones_markers` visualization based on `self.map_landmarks`.
*   **Goal:** Publish final results and manage the map based on reliable Track IDs and geometrically stable updates.

### [Step 10] Integration, Testing, and Tuning

*   **Action:** Create a ROS 2 launch file (`lio.launch.py`) to start `LIO_Node` and other relevant nodes. Configure parameters.
*   **Action:** Test with recorded bag files or live sensor data.
*   **Action:** Use RViz to visualize TFs, `/odom_lio`, `/mapped_cones_markers`, and input `/fused_sorted_cones_ukf`.
*   **Action:** Tune parameters, focusing on:
    *   Initialization quality checks.
    *   `MIN_KNOWN_LANDMARKS_FOR_UPDATE` threshold (Step 6).
    *   **SVD quality metrics and thresholds (Step 7).**
    *   Drift correction smoothing factor and update gating logic (Step 8).
    *   KF parameters (`Q`, `R`, `P0`).
    *   Map update criteria (initially simple).
*   **Action:** Evaluate performance (trajectory accuracy, drift, map consistency).
*   **Goal:** Validate the **simplified and robustified (TrackID-based)** LIO system and optimize its performance.

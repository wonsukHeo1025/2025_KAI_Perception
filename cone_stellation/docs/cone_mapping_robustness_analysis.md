# Cone Mapping Robustness Analysis (2025-08-08)

## Context (from current system behavior)
- EKF odometry (/odometry/filtered) fuses IMU+GPS and becomes accurate once the vehicle moves; yaw drifts while stationary (no wheel encoder) then “snaps in” on motion.
- Cone-based SLAM launches fine with bag; mapping looks sensitive to vehicle/lidar shake, leading to inconsistent cone landmarks.

## Key findings from code/config
- Preprocessing (`include/cone_stellation/preprocessing/cone_preprocessor.hpp`)
  - Outlier filter only checks distance and confidence; no temporal smoothing across frames.
  - Pattern detection runs but pattern-based factors are disabled in mapping, so detection has no downstream effect.
  - Tracking assigns incremental IDs but doesn’t do motion gating, smoothing, or track-based filtering.

- Data association
  - Main mapping (`include/cone_stellation/mapping/cone_mapping.hpp`) uses nearest-neighbor with color gate but no covariance/Mahalanobis gating and no explicit track-id preference when associating.
  - A more complete association module exists (`include/cone_stellation/mapping/data_association.hpp`) but is used only by `SimpleConeMapping`, not by `ConeMapping`.

- Landmark creation
  - Early-phase shortcut directly creates landmarks for the first N observations: currently N=30. This bypasses the tentative-landmark buffer and can crystallize noisy detections during platform shake.

- Factor modeling
  - Observation factors use constant noise (`config_.cone_observation_noise`), not scaled by range or per-detection confidence.
  - Inter-landmark distance factors are created based on co-observation counts, but the measured distance is taken from current landmark estimates (map frame), not from same-frame observations. This couples the constraint to transient pose/landmark errors.
  - No robust loss (Huber/Tukey) on observation or inter-landmark factors, so outliers can pull strongly.

- Timing/optimization
  - Keyframes are created at >1.0m or >0.2rad; appropriate to avoid excessive keyframes from shake.
  - Optimization cadence is configured in YAML as 1, and the node reads parameters; ISAM2 params in YAML (relinearize_threshold/skip) are defined, but the node does not explicitly load them into `mapping_config_` (constructor defaults are used). Minor mismatch risk.

- TF and frames
  - Cone observations are transformed to `base_link` before factoring; good. `map->odom` is fixed identity currently (drift correction disabled) to avoid circular dependency; OK.

## Why mapping looks fragile under shake
- Sensor shake perturbs relative cone measurements in the vehicle frame; with constant, relatively tight noise, these perturbations turn into inconsistent landmark updates.
- Early direct landmark creation admits outliers before they are stabilized by repeated observations.
- Association without covariance gating or track-id leverage is more prone to mismatches when observations jitter.
- Inter-landmark factors derived from current map positions can reinforce early errors instead of averaging measurement geometry across co-observations.

## Recommended improvements (prioritized)
1. Gate SLAM updates by motion/quality
   - Do not create keyframes or update mapping while the vehicle is stationary or EKF yaw covariance is high. Use `/odometry/filtered` twist or covariance to require small linear speed > v_min or covariance below threshold.
   - Practical: add a check in `cone_callback()` before building frames.

2. Always use tentative landmarks (remove early direct creation)
   - Remove or drastically reduce the “first 30 landmarks direct creation” path. Promote only through `TentativeLandmark` with current strict thresholds (min_observations=3, min_time_span=0.5s, max_var=0.2, min_color_conf=0.8).
   - Expected effect: fewer spurious landmarks during shake; landmarks stabilize before entering graph.

3. Observation noise adaptation + robust loss
   - Scale per-observation sigma by range and inverse of detection confidence, e.g., sigma = base + k_range*range / conf_scale.
   - Wrap observation and inter-landmark factors in a Huber loss (e.g., k=1.345*sigma) to down-weight outliers.
   - Implement in `add_observation_factor()` and `create_distance_factor()`.

4. Stronger association
   - Prefer track-id matches (explicit negative score like in `DataAssociation`) and fall back to nearest neighbor with color gate.
   - Add Mahalanobis gating using an innovation covariance that includes pose and measurement covariance.
   - Reduce `association.max_association_distance` if needed (start at 1.0–1.2m with track-id preference enabled).

5. Inter-landmark factors from same-frame geometry
   - Compute measured distance using same-frame, sensor-frame observations (or their average across co-observations), not current map positions. Maintain a small buffer of co-observed measured distances per pair and use the median.
   - Increase `min_covisibility_count` to 3 to ensure stability; keep a minimum distance (>=1.0m) and color compatibility if applicable.
   - Apply robust loss to distance factors.

6. Preprocessor smoothing for tracked cones
   - Maintain a short history per track-id and apply exponential smoothing or a sliding median to `obs.position` before passing to mapping.
   - Optionally disable pattern detection to save CPU until pattern factors are re-enabled.

7. Parameter alignment
   - Load ISAM2 params from YAML into `mapping_config_` (relinearize_threshold/skip) to match `config/slam_config.yaml`.
   - Consider setting `optimize_every_n_frames` to 2–3 if CPU is tight under robust loss, otherwise 1 is fine.

## Concrete code touchpoints
- `src/cone_stellation/ros/cone_slam_node.cpp`
  - Add motion/yaw-covariance gate before `should_create_keyframe()` and frame creation.
- `include/cone_stellation/mapping/cone_mapping.hpp`
  - Remove/limit direct landmark creation (search “DIRECT CREATION”).
  - In `associate_with_confirmed_landmark()`: add track-id preference and Mahalanobis gating.
  - In `add_observation_factor()`: compute per-observation noise and wrap with robust loss.
  - In `create_distance_factor()`: use buffered same-frame distances and robust loss; raise `min_covisibility_count`.
- `include/cone_stellation/preprocessing/cone_preprocessor.hpp`
  - Add smoothing for `update_tracking()` positions by track-id; optionally disable pattern detection unless used.
- `config/slam_config.yaml`
  - Tighten `association.max_association_distance` if needed; confirm ISAM2 params are applied; configure robust loss parameters if exposed.

## Expected impact
- Less landmark flicker and drift under platform shake.
- Fewer false landmarks; improved association stability with track-id and covariance gating.
- Inter-landmark constraints reflect real, repeated geometry rather than transient map estimates.

## Next steps (incremental)
1) Implement gates + remove direct creation; test on same bag. 2) Add per-obs adaptive noise + Huber; test. 3) Switch inter-landmark distance to same-frame median; test. 4) Add preprocessor smoothing. 5) Tune association distance and thresholds.

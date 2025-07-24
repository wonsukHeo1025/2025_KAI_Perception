#!/usr/bin/env python3
import numpy as np

# --- Hardcoded Ground Truth Cones ---
_cone_id_counter = 0 
def _add_cone_global(cones_dict, x, y, cone_type, unique_id):
    cones_dict[unique_id] = {'pos': np.array([float(x), float(y)]), 'type': cone_type.lower()}

# --- SCENARIO 1 ---
GROUND_TRUTH_CONES_SCENARIO_1 = {}
s1_current_cone_id = 0 
LANE_WIDTH_S1 = 5.0
BLUE_Y_S1 = LANE_WIDTH_S1 / 2.0  # 2.5
YELLOW_Y_S1 = -LANE_WIDTH_S1 / 2.0 # -2.5
CONE_SPACING_S1 = 2.5

# 0m to 100m: Blue (left) / Yellow (right)
for x_pos_s1 in np.arange(0, 100.0, CONE_SPACING_S1):
    s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, x_pos_s1, BLUE_Y_S1, "blue", s1_current_cone_id)
    s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, x_pos_s1, YELLOW_Y_S1, "yellow", s1_current_cone_id)
s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, 100.0, BLUE_Y_S1, "blue", s1_current_cone_id) # Ensure end cone
s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, 100.0, YELLOW_Y_S1, "yellow", s1_current_cone_id) # Ensure end cone

# 100m to 150m: Red cones for AEB zone
# Left side (Y = +2.5)
for x_pos_s1 in np.arange(100.0 + CONE_SPACING_S1, 150.0, CONE_SPACING_S1): # Start after the 100m mark
    s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, x_pos_s1, BLUE_Y_S1, "red", s1_current_cone_id)
s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, 150.0, BLUE_Y_S1, "red", s1_current_cone_id) # End cone for red lane

# Right side (Y = -2.5)
for x_pos_s1 in np.arange(100.0 + CONE_SPACING_S1, 150.0, CONE_SPACING_S1):
    s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, x_pos_s1, YELLOW_Y_S1, "red", s1_current_cone_id)
s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, 150.0, YELLOW_Y_S1, "red", s1_current_cone_id) # End cone for red lane

# End wall of Red cones at X = 150m
# Denser wall, ensure y_pos does not exceed BLUE_Y_S1 for the last cone if CONE_SPACING_S1/1.5 is not a perfect divisor
y_positions_for_wall = list(np.arange(YELLOW_Y_S1, BLUE_Y_S1, CONE_SPACING_S1/1.5))
if BLUE_Y_S1 not in y_positions_for_wall: # Ensure the blue line extent is included
    y_positions_for_wall.append(BLUE_Y_S1)

for y_pos_s1 in y_positions_for_wall:
    s1_current_cone_id += 1; _add_cone_global(GROUND_TRUTH_CONES_SCENARIO_1, 150.0, y_pos_s1, "red", s1_current_cone_id)


# --- SCENARIO 2 ---
GROUND_TRUTH_CONES_SCENARIO_2 = {}
s2_current_cone_id = 0

def _add_cone_s2(cones_dict, x, y, cone_type_str, segment_prefix, cone_idx_in_segment):
    global s2_current_cone_id # Note: This global might need refactoring if used across modules directly
    s2_current_cone_id += 1
    cones_dict[s2_current_cone_id] = {'pos': np.array([float(x), float(y)]), 'type': cone_type_str.lower()}

def _generate_cones_on_arc_s2(cones_dict, center_x, center_y, radius, start_angle_deg, end_angle_deg,
                             cone_type_str, segment_prefix, angle_step_deg=10):
    if np.isclose(start_angle_deg, end_angle_deg):
        rad = np.radians(start_angle_deg)
        _add_cone_s2(cones_dict, center_x + radius * np.cos(rad), center_y + radius * np.sin(rad), cone_type_str, segment_prefix, 0)
        return
    effective_end_angle_deg = end_angle_deg + 360 if end_angle_deg < start_angle_deg else end_angle_deg
    total_arc_angle = effective_end_angle_deg - start_angle_deg
    num_steps = max(1, int(np.ceil(abs(total_arc_angle) / angle_step_deg))) 
    angles_rad = np.linspace(np.radians(start_angle_deg), np.radians(effective_end_angle_deg), num_steps + 1)
    for idx, rad in enumerate(angles_rad):
        x = center_x + radius * np.cos(rad)
        y = center_y + radius * np.sin(rad)
        _add_cone_s2(cones_dict, x, y, cone_type_str, segment_prefix, idx)

def _generate_cones_on_line_s2(cones_dict, p1, p2, cone_type_str, segment_prefix, distance_step=2.0):
    length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    if np.isclose(length, 0):
        _add_cone_s2(cones_dict, p1[0], p1[1], cone_type_str, segment_prefix, 0)
        return
    num_segments = max(1, int(np.ceil(length / distance_step)))
    for i in range(num_segments + 1):
        ratio = i / num_segments
        x = p1[0] * (1 - ratio) + p2[0] * ratio
        y = p1[1] * (1 - ratio) + p2[1] * ratio
        _add_cone_s2(cones_dict, x, y, cone_type_str, segment_prefix, i)

# --- Scenario 2: 트랙 형상 정의 및 콘 생성 ---
center_R_Y = (98.0, 12.0)
center_L_Y = (12.0, 12.0)
radius_Y = 12.0
y_lower_Y = 10.0
y_upper_Y = 30.0
len_upper_straight_Y = 50.0

delta_y_RCE_Y = y_lower_Y - center_R_Y[1]
P_RCE_Y_dx = -np.sqrt(max(0, radius_Y**2 - delta_y_RCE_Y**2))
P_RCE_Y = (center_R_Y[0] + P_RCE_Y_dx, y_lower_Y)
angle_R_start_Y_deg = np.degrees(np.arctan2(delta_y_RCE_Y, P_RCE_Y_dx))
angle_R_start_Y_deg += 360 if angle_R_start_Y_deg < 0 else 0
angle_R_end_Y_deg = (angle_R_start_Y_deg + 270.0) % 360
P_RCX_Y = (center_R_Y[0] + radius_Y * np.cos(np.radians(angle_R_end_Y_deg)), 
           center_R_Y[1] + radius_Y * np.sin(np.radians(angle_R_end_Y_deg)))
y_intermediate_L_Y = P_RCX_Y[1]
delta_y_LCE_Y = y_intermediate_L_Y - center_L_Y[1]
P_LCE_Y_dx = np.sqrt(max(0, radius_Y**2 - delta_y_LCE_Y**2))
x_LCE_Y = center_L_Y[0] + P_LCE_Y_dx
total_x_span_upper_Y = P_RCX_Y[0] - x_LCE_Y
x_span_diag_Y = total_x_span_upper_Y - len_upper_straight_Y
D1x_Y = x_span_diag_Y / 2.0 if x_span_diag_Y > 0 else 0
len_upper_straight_Y_eff = len_upper_straight_Y if x_span_diag_Y > 0 else total_x_span_upper_Y
P_USS_Y = (P_RCX_Y[0] - D1x_Y, y_upper_Y)
P_USE_Y = (P_USS_Y[0] - len_upper_straight_Y_eff, y_upper_Y)
P_LCE_Y = (x_LCE_Y, y_intermediate_L_Y)
angle_L_start_Y_deg = np.degrees(np.arctan2(P_LCE_Y[1] - center_L_Y[1], P_LCE_Y[0] - center_L_Y[0])) # Corrected
angle_L_start_Y_deg += 360 if angle_L_start_Y_deg < 0 else 0
angle_L_end_Y_deg_plot = (angle_L_start_Y_deg + 270.0) % 360
x_LCX_Y = center_L_Y[0] + radius_Y * np.cos(np.radians(angle_L_end_Y_deg_plot))
P_LCX_Y = (x_LCX_Y, y_lower_Y)

radius_B = 7.0
y_lower_B = 15.0
y_upper_B = 25.0
delta_y_RCE_B = y_lower_B - center_R_Y[1]
P_RCE_B_dx = -np.sqrt(max(0, radius_B**2 - delta_y_RCE_B**2))
P_RCE_B = (center_R_Y[0] + P_RCE_B_dx, y_lower_B)
angle_R_start_B_deg = np.degrees(np.arctan2(delta_y_RCE_B, P_RCE_B_dx))
angle_R_start_B_deg += 360 if angle_R_start_B_deg < 0 else 0
angle_R_end_B_deg = (angle_R_start_B_deg + 270.0) % 360
P_RCX_B = (center_R_Y[0] + radius_B * np.cos(np.radians(angle_R_end_B_deg)), 
           center_R_Y[1] + radius_B * np.sin(np.radians(angle_R_end_B_deg)))
P_USS_B = (P_USS_Y[0], y_upper_B) 
P_USE_B = (P_USE_Y[0], y_upper_B)
y_intermediate_L_B = P_RCX_B[1] 
delta_y_LCE_B = y_intermediate_L_B - center_L_Y[1]
P_LCE_B_dx = np.sqrt(max(0, radius_B**2 - delta_y_LCE_B**2))
x_LCE_B = center_L_Y[0] + P_LCE_B_dx
P_LCE_B = (x_LCE_B, y_intermediate_L_B)
angle_L_start_B_deg = np.degrees(np.arctan2(P_LCE_B[1] - center_L_Y[1], P_LCE_B[0] - center_L_Y[0])) # Corrected
angle_L_start_B_deg += 360 if angle_L_start_B_deg < 0 else 0
delta_y_LCX_B_target = y_lower_B - center_L_Y[1]
P_LCX_B_dx_target = np.sqrt(max(0, radius_B**2 - delta_y_LCX_B_target**2))
x_LCX_B_target = center_L_Y[0] + P_LCX_B_dx_target
P_LCX_B = (x_LCX_B_target, y_lower_B)
angle_L_end_B_target_deg = np.degrees(np.arctan2(delta_y_LCX_B_target, P_LCX_B_dx_target))
angle_L_end_B_target_deg += 360 if angle_L_end_B_target_deg < 0 else 0
sweep_L_B_deg = angle_L_end_B_target_deg - angle_L_start_B_deg
if sweep_L_B_deg < -180:
    sweep_L_B_deg += 360
elif sweep_L_B_deg > 180:
    sweep_L_B_deg -= 360
angle_L_end_B_plot_final_deg = (angle_L_start_B_deg + sweep_L_B_deg) % 360

cone_arc_angle_step_Y_s2 = 15
cone_arc_angle_step_B_s2 = 18
cone_line_distance_step_s2 = 3.5

_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_LCX_Y, P_RCE_Y, "yellow", "Y-LS", cone_line_distance_step_s2)
_generate_cones_on_arc_s2(GROUND_TRUTH_CONES_SCENARIO_2, center_R_Y[0], center_R_Y[1], radius_Y, angle_R_start_Y_deg, angle_R_end_Y_deg, "yellow", "Y-RT", cone_arc_angle_step_Y_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_RCX_Y, P_USS_Y, "yellow", "Y-D1", cone_line_distance_step_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_USS_Y, P_USE_Y, "yellow", "Y-US", cone_line_distance_step_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_USE_Y, P_LCE_Y, "yellow", "Y-D2", cone_line_distance_step_s2)
_generate_cones_on_arc_s2(GROUND_TRUTH_CONES_SCENARIO_2, center_L_Y[0], center_L_Y[1], radius_Y, angle_L_start_Y_deg, angle_L_end_Y_deg_plot % 360, "yellow", "Y-LT", cone_arc_angle_step_Y_s2)

_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_LCX_B, P_RCE_B, "blue", "B-LS", cone_line_distance_step_s2)
_generate_cones_on_arc_s2(GROUND_TRUTH_CONES_SCENARIO_2, center_R_Y[0], center_R_Y[1], radius_B, angle_R_start_B_deg, angle_R_end_B_deg, "blue", "B-RT", cone_arc_angle_step_B_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_RCX_B, P_USS_B, "blue", "B-D1", cone_line_distance_step_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_USS_B, P_USE_B, "blue", "B-US", cone_line_distance_step_s2)
_generate_cones_on_line_s2(GROUND_TRUTH_CONES_SCENARIO_2, P_USE_B, P_LCE_B, "blue", "B-D2", cone_line_distance_step_s2)
_generate_cones_on_arc_s2(GROUND_TRUTH_CONES_SCENARIO_2, center_L_Y[0], center_L_Y[1], radius_B, angle_L_start_B_deg, angle_L_end_B_plot_final_deg % 360, "blue", "B-LT", cone_arc_angle_step_B_s2)
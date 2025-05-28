import yaml, os
import numpy as np

def extract_multi_configuration():
    # 고정된 절대 경로 사용
    config_folder = "/home/user1/ROS2_Workspace/ros2_ws/src/ros2_camera_lidar_fusion/config"
    config_file = os.path.join(config_folder, 'multi_general_configuration.yaml')
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # 설정에 config_folder 추가
    if 'general' not in config:
        config['general'] = {}
    config['general']['config_folder'] = config_folder
    
    return config

def flatten_nested_list(nested_list):
    """중첩 리스트를 1차원으로 변환"""
    return [item for sublist in nested_list for item in sublist]

def load_extrinsic_matrices(yaml_path):
    """외부 캘리브레이션 행렬 로드"""
    # 파일이 존재하지 않으면 빈 사전 반환
    if not os.path.exists(yaml_path):
        print(f"경고: 외부 캘리브레이션 파일이 없습니다: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 데이터가 None이면 빈 사전 반환
    if data is None:
        print(f"경고: 외부 캘리브레이션 파일이 비어 있습니다: {yaml_path}")
        return {}
    
    extrinsic_matrices = {}
    for camera_id in ['camera_1', 'camera_2']:
        if camera_id in data and 'extrinsic_matrix' in data[camera_id]:
            try:
                matrix_data = data[camera_id]['extrinsic_matrix']
                
                # 중첩 리스트인 경우 1차원으로 변환
                if isinstance(matrix_data[0], list):
                    matrix_data = flatten_nested_list(matrix_data)
                
                # 4x4 행렬로 변환
                T = np.array(matrix_data, dtype=np.float64).reshape(4, 4)
                extrinsic_matrices[camera_id] = T
            except Exception as e:
                print(f"경고: 카메라 {camera_id}의 외부 행렬을 처리하는 중 오류 발생: {e}")
    
    return extrinsic_matrices

def load_camera_calibrations(yaml_path):
    """내부 캘리브레이션 파라미터 로드"""
    # 파일이 존재하지 않으면 빈 사전 반환
    if not os.path.exists(yaml_path):
        print(f"경고: 내부 캘리브레이션 파일이 없습니다: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    
    # 데이터가 None이면 빈 사전 반환
    if calib_data is None:
        print(f"경고: 내부 캘리브레이션 파일이 비어 있습니다: {yaml_path}")
        return {}
    
    camera_calibrations = {}
    for camera_id in ['camera_1', 'camera_2']:
        if camera_id in calib_data:
            try:
                # 카메라 행렬 추출
                if 'camera_matrix' not in calib_data[camera_id]:
                    print(f"경고: {camera_id}에 camera_matrix가 없습니다.")
                    continue
                
                cam_mat = calib_data[camera_id]['camera_matrix']
                if 'data' not in cam_mat:
                    print(f"경고: {camera_id}의 camera_matrix에 data가 없습니다.")
                    continue
                
                cam_mat_data = cam_mat['data']
                
                # 중첩 리스트인 경우 1차원으로 변환
                if isinstance(cam_mat_data[0], list):
                    cam_mat_data = flatten_nested_list(cam_mat_data)
                
                camera_matrix = np.array(cam_mat_data, dtype=np.float64).reshape(3, 3)
                
                # 왜곡 계수 추출
                if 'distortion_coefficients' not in calib_data[camera_id]:
                    print(f"경고: {camera_id}에 distortion_coefficients가 없습니다.")
                    continue
                
                dist_coef = calib_data[camera_id]['distortion_coefficients']
                if 'data' not in dist_coef:
                    print(f"경고: {camera_id}의 distortion_coefficients에 data가 없습니다.")
                    continue
                
                dist_data = dist_coef['data']
                
                # 중첩 리스트인 경우 1차원으로 변환
                if isinstance(dist_data[0], list):
                    dist_data = flatten_nested_list(dist_data)
                
                dist_coeffs = np.array(dist_data, dtype=np.float64).reshape(1, -1)
                
                camera_calibrations[camera_id] = (camera_matrix, dist_coeffs)
            except Exception as e:
                print(f"경고: 카메라 {camera_id}의 내부 캘리브레이션을 처리하는 중 오류 발생: {e}")
    
    return camera_calibrations 
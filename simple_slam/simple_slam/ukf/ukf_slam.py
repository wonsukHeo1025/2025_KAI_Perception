#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLAM(Simultaneous Localization and Mapping)을 위한 UKF 확장.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import time
import math

from simple_slam.ukf.ukf_base import UKF


class UKFSLAM(UKF):
    """
    SLAM을 위한 UKF 확장 클래스.
    로봇 상태와 랜드마크 위치를 동시에 추정하기 위한 UKF 확장.
    """

    def __init__(self, 
                 dim_x: int,
                 dim_z: int,
                 fx: Callable,
                 hx: Callable,
                 dt: float = 0.1,
                 points: Optional[Callable] = None,
                 sqrt_fn: Optional[Callable] = None,
                 x_mean_fn: Optional[Callable] = None,
                 z_mean_fn: Optional[Callable] = None,
                 residual_x: Optional[Callable] = None,
                 residual_z: Optional[Callable] = None,
                 landmark_dim: int = 2):
        """
        UKFSLAM 초기화.

        Args:
            dim_x (int): 로봇 상태 벡터 차원 (랜드마크 제외)
            dim_z (int): 측정벡터 차원
            fx (Callable): 상태 전이 함수 f(x, dt) - 로봇 상태를 다음 시간으로 전파
            hx (Callable): 측정 함수 h(x, landmark_idx) - 특정 랜드마크 측정 예측
            dt (float): 시간 간격 (초)
            points (Callable, optional): 시그마 포인트 생성 함수
            sqrt_fn (Callable, optional): 행렬 제곱근 함수
            x_mean_fn (Callable, optional): 상태 평균 계산 함수
            z_mean_fn (Callable, optional): 측정 평균 계산 함수
            residual_x (Callable, optional): 상태 잔차 계산 함수
            residual_z (Callable, optional): 측정 잔차 계산 함수
            landmark_dim (int): 각 랜드마크 차원 (기본값: 2D 좌표)
        """
        super().__init__(dim_x, dim_z, fx, hx, dt, points, sqrt_fn,
                        x_mean_fn, z_mean_fn, residual_x, residual_z)
        
        # 랜드마크 정보
        self.landmark_dim = landmark_dim
        self.landmarks = {}  # 랜드마크 ID -> 인덱스 매핑
        self.landmark_id_to_idx = {}  # 랜드마크 ID -> 인덱스 매핑
        self.landmark_idx_to_id = {}  # 인덱스 -> 랜드마크 ID 매핑
        self.landmark_count = 0
        
        # 원래 로봇 상태 차원
        self.robot_dim = dim_x
        
        # 데이터 연관 임계값
        self.association_threshold = 5.0  # 마할라노비스 거리 임계값
        
        # 랜드마크 초기화 노이즈
        self.landmark_init_noise = np.eye(landmark_dim) * 0.1
        
        # 랜드마크 공분산 초기화 스케일
        self.landmark_cov_scale = 1.0
        
        # 마지막 데이터 연관 결과
        self.last_association_result = None
        
    def initialize(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """
        UKFSLAM 상태 및 공분산 초기화.

        Args:
            x (np.ndarray): 초기 로봇 상태 벡터
            P (np.ndarray, optional): 초기 로봇 공분산 행렬
        """
        # 상태 벡터 초기화 (로봇 상태만)
        self.x = x
        if P is not None:
            self.P = P
        else:
            self.P = np.eye(self.robot_dim)
        
        # 랜드마크 데이터 초기화
        self.landmarks = {}
        self.landmark_id_to_idx = {}
        self.landmark_idx_to_id = {}
        self.landmark_count = 0
        
        # 시간 초기화
        self.last_predict_time = time.time()
        self.last_update_time = time.time()
        self.log_likelihood = 0.0
    
    def predict(self, dt: Optional[float] = None, **fx_args):
        """
        UKFSLAM 예측 단계 (시간 갱신).
        로봇 상태만 다음 시간으로 전파하고, 랜드마크는 그대로 유지합니다.
        
        참고: 랜드마크는 정적 특성으로 전이 함수를 적용하지 않습니다.

        Args:
            dt (float, optional): 시간 간격. None이면 기본값 사용
            **fx_args: 상태 전이 함수 fx에 전달할 추가 인자
        """
        if dt is not None:
            self.dt = dt
            
        # 현재 상태 벡터 및 공분산 행렬
        x = self.x
        P = self.P
        
        # 시그마 포인트 생성
        sigmas = self.sigma_points(x, P)
        
        # 예측된 시그마 포인트 배열 초기화
        sigmas_f = np.zeros_like(sigmas)
        
        # 로봇 상태에만 전이 함수 적용
        for i in range(len(sigmas)):
            # 로봇 상태 추출 및 전파
            robot_state = sigmas[i, :self.robot_dim]
            sigmas_f[i, :self.robot_dim] = self.fx(robot_state, self.dt, **fx_args)
            
            # 랜드마크는 그대로 유지
            if len(sigmas[i]) > self.robot_dim:
                sigmas_f[i, self.robot_dim:] = sigmas[i, self.robot_dim:]
        
        self.sigmas_f = sigmas_f
        
        # 새로운 상태 계산
        self.x = self.x_mean_fn(sigmas_f, self.Wm)
        
        # 공분산 계산
        self.P = self.calculate_covariance(sigmas_f, self.x, self.Wc, self.residual_x)
        
        # 프로세스 노이즈 추가 (로봇 부분만)
        if self.landmark_count > 0:
            # Q 행렬 확장 (로봇 부분에만 노이즈 적용)
            Q_expanded = np.zeros_like(self.P)
            Q_expanded[:self.robot_dim, :self.robot_dim] = self.Q[:self.robot_dim, :self.robot_dim]
            self.P += Q_expanded
        else:
            # 랜드마크가 없으면 기본 Q 사용
            self.P += self.Q
        
        self.last_predict_time = time.time()
    
    def add_landmark(self, landmark_id: Any, initial_position: np.ndarray, 
                    uncertainty: Optional[np.ndarray] = None) -> int:
        """
        새 랜드마크를 상태 벡터에 추가.
        
        Args:
            landmark_id (Any): 랜드마크 식별자
            initial_position (np.ndarray): 초기 랜드마크 위치
            uncertainty (np.ndarray, optional): 초기 랜드마크 불확실성
            
        Returns:
            int: 랜드마크 인덱스
        """
        if landmark_id in self.landmark_id_to_idx:
            return self.landmark_id_to_idx[landmark_id]
        
        # 상태 벡터 및 공분산 행렬 확장
        old_x = self.x
        old_P = self.P
        
        # 랜드마크 인덱스 계산
        landmark_idx = self.landmark_count
        landmark_state_idx = self.robot_dim + landmark_idx * self.landmark_dim
        
        # 상태 벡터 확장
        self.x = np.zeros(len(old_x) + self.landmark_dim)
        self.x[:len(old_x)] = old_x
        self.x[landmark_state_idx:landmark_state_idx + self.landmark_dim] = initial_position
        
        # 공분산 행렬 확장
        self.P = np.zeros((len(self.x), len(self.x)))
        self.P[:len(old_P), :len(old_P)] = old_P
        
        # 랜드마크 공분산 초기화
        if uncertainty is None:
            # 기본 불확실성 사용
            uncertainty = self.landmark_init_noise * self.landmark_cov_scale
        
        # 랜드마크 공분산 설정
        self.P[landmark_state_idx:landmark_state_idx + self.landmark_dim, 
              landmark_state_idx:landmark_state_idx + self.landmark_dim] = uncertainty
        
        # 랜드마크 정보 저장
        self.landmark_id_to_idx[landmark_id] = landmark_idx
        self.landmark_idx_to_id[landmark_idx] = landmark_id
        self.landmarks[landmark_id] = {
            'idx': landmark_idx,
            'position': initial_position,
            'first_observed': time.time(),
            'last_observed': time.time(),
            'observation_count': 1
        }
        
        self.landmark_count += 1
        
        # UKF 파라미터 재계산 (차원 변경됨)
        self.dim_x = len(self.x)
        self._compute_weights()
        
        # Q 및 sigmas_f 확장
        if self.Q.shape[0] < self.dim_x:
            # Q 행렬 확장
            Q_new = np.zeros((self.dim_x, self.dim_x))
            Q_new[:self.Q.shape[0], :self.Q.shape[1]] = self.Q
            self.Q = Q_new
        
        # 시그마 포인트 무효화 (다음 예측에서 재계산)
        self.sigmas_f = None
        
        return landmark_idx
    
    def update_landmark(self, landmark_id: Any, z: np.ndarray, R: Optional[np.ndarray] = None,
                       **hx_args) -> float:
        """
        특정 랜드마크 관측을 사용해 UKF 업데이트 수행.
        
        Args:
            landmark_id (Any): 업데이트할 랜드마크 ID
            z (np.ndarray): 측정 벡터
            R (np.ndarray, optional): 측정 노이즈 공분산
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            float: 마할라노비스 거리 (측정값의 유효성 지표)
        """
        if landmark_id not in self.landmark_id_to_idx:
            raise ValueError(f"랜드마크 ID {landmark_id}가 존재하지 않습니다")
        
        landmark_idx = self.landmark_id_to_idx[landmark_id]
        
        # 랜드마크 측정 갱신
        innovation = self._update_with_landmark(landmark_idx, z, R, **hx_args)
        
        # 랜드마크 정보 갱신
        lm_state_idx = self.robot_dim + landmark_idx * self.landmark_dim
        self.landmarks[landmark_id]['position'] = self.x[lm_state_idx:lm_state_idx + self.landmark_dim].copy()
        self.landmarks[landmark_id]['last_observed'] = time.time()
        self.landmarks[landmark_id]['observation_count'] += 1
        
        return innovation
    
    def _update_with_landmark(self, landmark_idx: int, z: np.ndarray, 
                            R: Optional[np.ndarray] = None, **hx_args) -> float:
        """
        특정 랜드마크 측정으로 UKF 업데이트 수행하는 내부 메서드.
        
        Args:
            landmark_idx (int): 랜드마크 인덱스
            z (np.ndarray): 측정 벡터
            R (np.ndarray, optional): 측정 노이즈 공분산
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            float: 마할라노비스 거리
        """
        if z is None:
            return float('inf')
        
        if R is not None:
            self.R = R
        
        # 시그마 포인트가 없으면 예측 수행
        if self.sigmas_f is None:
            self.predict()
        
        # 시그마 포인트를 측정 공간으로 변환
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(len(self.sigmas_f)):
            sigmas_h[i] = self.hx(self.sigmas_f[i], landmark_idx=landmark_idx, **hx_args)
        
        self.sigmas_h = sigmas_h
        
        # 예측된 측정값 계산
        zp = self.z_mean_fn(sigmas_h, self.Wm)
        
        # 측정 공분산 계산
        self.S = self.calculate_covariance(sigmas_h, zp, self.Wc, self.residual_z)
        
        # 측정 노이즈 추가
        self.S += self.R
        
        # 교차 공분산 계산
        Pxz = self.calculate_cross_covariance(self.sigmas_f, self.x, sigmas_h, zp)
        
        # 칼만 이득 계산
        self.K = np.dot(Pxz, np.linalg.inv(self.S))
        
        # 측정 잔차 계산
        self.y = self.residual_z(z, zp)
        
        # 상태 및 공분산 업데이트
        self.x += np.dot(self.K, self.y)
        self.P -= np.dot(self.K, np.dot(self.S, self.K.T))
        
        # 로그 가능도 계산 및 저장
        self.log_likelihood += self._log_likelihood(z, zp, self.S)
        
        # 마할라노비스 거리 계산
        self.innovation = np.sqrt(np.dot(self.y.T, np.dot(np.linalg.inv(self.S), self.y)))
        
        self.last_update_time = time.time()
        self.z = z
        
        return self.innovation
    
    def process_measurements(self, measurements: List[Tuple[Any, np.ndarray]], 
                            R: Optional[np.ndarray] = None, 
                            create_new: bool = True,
                            max_distance: float = float('inf'),
                            **hx_args) -> Dict:
        """
        다수의 랜드마크 측정값을 처리하고 데이터 연관을 수행.
        
        Args:
            measurements (List[Tuple[Any, np.ndarray]]): (랜드마크 ID, 측정값) 튜플 리스트
            R (np.ndarray, optional): 측정 노이즈 공분산
            create_new (bool): 새 랜드마크 생성 여부
            max_distance (float): 최대 마할라노비스 거리 임계값
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            Dict: 처리 결과를 포함한 딕셔너리
        """
        if not measurements:
            return {
                'updated': [],
                'created': [],
                'unmatched': [],
                'innovations': []
            }
        
        # 결과 저장 변수
        updated_landmarks = []
        created_landmarks = []
        unmatched_measurements = []
        innovations = []
        
        # 측정 노이즈 설정
        if R is not None:
            self.R = R
        
        # 각 측정값 처리
        for landmark_id, z in measurements:
            if landmark_id in self.landmark_id_to_idx:
                # 기존 랜드마크 업데이트
                innovation = self.update_landmark(landmark_id, z, self.R, **hx_args)
                
                if innovation <= max_distance:
                    updated_landmarks.append(landmark_id)
                    innovations.append(innovation)
                else:
                    # 측정값이 너무 멀면 무시
                    unmatched_measurements.append((landmark_id, z))
            elif create_new:
                # 새 랜드마크 생성
                lm_pos = self._initialize_landmark_position(z, **hx_args)
                idx = self.add_landmark(landmark_id, lm_pos)
                created_landmarks.append(landmark_id)
            else:
                unmatched_measurements.append((landmark_id, z))
        
        result = {
            'updated': updated_landmarks,
            'created': created_landmarks,
            'unmatched': unmatched_measurements,
            'innovations': innovations
        }
        
        self.last_association_result = result
        return result
    
    def _initialize_landmark_position(self, z: np.ndarray, **hx_args) -> np.ndarray:
        """
        측정값에서 랜드마크 초기 위치 계산.
        기본 구현으로, 실제 적용 시에는 오버라이드해야 함.
        
        Args:
            z (np.ndarray): 측정 벡터
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            np.ndarray: 초기 랜드마크 위치
        """
        # 이 메서드는 하위 클래스에서 오버라이드해야 함
        # 기본 구현은 랜드마크 차원에 맞는 빈 배열 반환
        return np.zeros(self.landmark_dim)
    
    def data_association(self, measurements: List[Tuple[Any, np.ndarray]], 
                        threshold: Optional[float] = None,
                        **hx_args) -> Dict:
        """
        측정값과 기존 랜드마크 간의 데이터 연관 수행.
        
        Args:
            measurements (List[Tuple[Any, np.ndarray]]): (잠재 ID, 측정값) 튜플 리스트
            threshold (float, optional): 마할라노비스 거리 임계값
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            Dict: 연관 결과를 포함한 딕셔너리
        """
        if threshold is None:
            threshold = self.association_threshold
            
        if not self.landmarks:
            # 랜드마크가 없으면 모든 측정값을 새 랜드마크로 분류
            return {
                'matched': [],
                'unmatched': measurements
            }
        
        # 결과 저장 변수
        matched = []
        unmatched = []
        
        # 각 측정값에 대해
        for potential_id, z in measurements:
            best_match = None
            best_dist = float('inf')
            
            # 각 랜드마크에 대해 마할라노비스 거리 계산
            for landmark_id, landmark_info in self.landmarks.items():
                landmark_idx = landmark_info['idx']
                
                # 예측된 측정값 및 공분산 계산
                zp, S = self._predict_measurement(landmark_idx, **hx_args)
                
                # 측정 잔차 계산
                y = self.residual_z(z, zp)
                
                # 마할라노비스 거리 계산
                try:
                    dist = np.sqrt(np.dot(y.T, np.dot(np.linalg.inv(S), y)))
                except:
                    dist = float('inf')
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = landmark_id
            
            # 최적 매치가 임계값 이내인지 확인
            if best_match is not None and best_dist <= threshold:
                matched.append((best_match, potential_id, z, best_dist))
            else:
                unmatched.append((potential_id, z))
        
        return {
            'matched': matched,  # (랜드마크 ID, 측정 ID, 측정값, 거리)
            'unmatched': unmatched  # (측정 ID, 측정값)
        }
    
    def _predict_measurement(self, landmark_idx: int, **hx_args) -> Tuple[np.ndarray, np.ndarray]:
        """
        특정 랜드마크에 대한 측정값 및 공분산 예측.
        
        Args:
            landmark_idx (int): 랜드마크 인덱스
            **hx_args: 측정 함수에 전달할 추가 인자
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (예측된 측정값, 측정 공분산)
        """
        # 시그마 포인트가 없으면 예측 수행
        if self.sigmas_f is None:
            self.predict()
        
        # 시그마 포인트를 측정 공간으로 변환
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(len(self.sigmas_f)):
            sigmas_h[i] = self.hx(self.sigmas_f[i], landmark_idx=landmark_idx, **hx_args)
        
        # 예측된 측정값 계산
        zp = self.z_mean_fn(sigmas_h, self.Wm)
        
        # 측정 공분산 계산
        S = self.calculate_covariance(sigmas_h, zp, self.Wc, self.residual_z)
        
        # 측정 노이즈 추가
        S += self.R
        
        return zp, S
    
    def get_landmark_states(self) -> Dict[Any, Dict]:
        """
        모든 랜드마크의 현재 상태 정보 반환.
        
        Returns:
            Dict[Any, Dict]: 랜드마크 ID를 키로 하는 상태 딕셔너리
        """
        result = {}
        
        for landmark_id, info in self.landmarks.items():
            landmark_idx = info['idx']
            state_idx = self.robot_dim + landmark_idx * self.landmark_dim
            state = self.x[state_idx:state_idx + self.landmark_dim].copy()
            
            # 공분산 인덱스 계산
            cov_slice = slice(state_idx, state_idx + self.landmark_dim)
            covariance = self.P[cov_slice, cov_slice].copy()
            
            # 표준편차 계산
            std_devs = np.sqrt(np.diag(covariance))
            
            result[landmark_id] = {
                'state': state,
                'covariance': covariance,
                'std_devs': std_devs,
                'first_observed': info['first_observed'],
                'last_observed': info['last_observed'],
                'observation_count': info['observation_count']
            }
        
        return result
    
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        로봇 상태 및 공분산 반환.
        
        Returns:
            Dict[str, np.ndarray]: 로봇 상태 정보
        """
        state = self.x[:self.robot_dim].copy()
        covariance = self.P[:self.robot_dim, :self.robot_dim].copy()
        std_devs = np.sqrt(np.diag(covariance))
        
        return {
            'state': state,
            'covariance': covariance,
            'std_devs': std_devs
        }
    
    def remove_landmark(self, landmark_id: Any):
        """
        특정 랜드마크를 상태 벡터에서 제거.
        
        Args:
            landmark_id (Any): 제거할 랜드마크 ID
        """
        if landmark_id not in self.landmark_id_to_idx:
            return
        
        landmark_idx = self.landmark_id_to_idx[landmark_id]
        state_idx = self.robot_dim + landmark_idx * self.landmark_dim
        end_idx = state_idx + self.landmark_dim
        
        # 상태 벡터 및 공분산 갱신
        self.x = np.delete(self.x, range(state_idx, end_idx))
        self.P = np.delete(np.delete(self.P, range(state_idx, end_idx), 0), range(state_idx, end_idx), 1)
        
        # 랜드마크 정보 갱신
        del self.landmarks[landmark_id]
        del self.landmark_idx_to_id[landmark_idx]
        
        # 인덱스 갱신 (제거된 랜드마크 이후의 인덱스 조정)
        new_id_to_idx = {}
        new_idx_to_id = {}
        
        for id, idx in self.landmark_id_to_idx.items():
            if idx < landmark_idx:
                new_id_to_idx[id] = idx
                new_idx_to_id[idx] = id
            elif idx > landmark_idx:
                new_id_to_idx[id] = idx - 1
                new_idx_to_id[idx - 1] = id
        
        self.landmark_id_to_idx = new_id_to_idx
        self.landmark_idx_to_id = new_idx_to_id
        
        self.landmark_count -= 1
        self.dim_x = len(self.x)
        
        # UKF 파라미터 재계산
        self._compute_weights()
        
        # 시그마 포인트 무효화 (다음 예측에서 재계산)
        self.sigmas_f = None
    
    def prune_landmarks(self, max_age: float = float('inf'), 
                       min_observations: int = 0,
                       max_std_dev: float = float('inf')):
        """
        기준에 따라 랜드마크 제거.
        
        Args:
            max_age (float): 최대 경과 시간 (초)
            min_observations (int): 최소 관측 횟수
            max_std_dev (float): 최대 표준편차
        """
        current_time = time.time()
        landmarks_to_remove = []
        
        for landmark_id, info in self.landmarks.items():
            landmark_idx = info['idx']
            state_idx = self.robot_dim + landmark_idx * self.landmark_dim
            
            # 경과 시간 검사
            age = current_time - info['first_observed']
            if age > max_age:
                landmarks_to_remove.append(landmark_id)
                continue
            
            # 관측 횟수 검사
            if info['observation_count'] < min_observations:
                landmarks_to_remove.append(landmark_id)
                continue
            
            # 불확실성 검사
            cov_slice = slice(state_idx, state_idx + self.landmark_dim)
            std_devs = np.sqrt(np.diag(self.P[cov_slice, cov_slice]))
            
            if np.any(std_devs > max_std_dev):
                landmarks_to_remove.append(landmark_id)
                continue
        
        # 랜드마크 제거
        for landmark_id in landmarks_to_remove:
            self.remove_landmark(landmark_id)
        
        return landmarks_to_remove
    
    def reset(self):
        """
        UKFSLAM 상태 리셋.
        """
        # 기본 UKF 상태 리셋
        super().reset()
        
        # SLAM 관련 상태 리셋
        self.landmarks = {}
        self.landmark_id_to_idx = {}
        self.landmark_idx_to_id = {}
        self.landmark_count = 0
        self.dim_x = self.robot_dim
        self.x = np.zeros(self.robot_dim)
        self.P = np.eye(self.robot_dim)
        self.last_association_result = None
    
    def landmark_exists(self, landmark_id: Any) -> bool:
        """
        특정 랜드마크가 맵에 존재하는지 확인.
        
        Args:
            landmark_id (Any): 확인할 랜드마크 ID
            
        Returns:
            bool: 랜드마크 존재 여부
        """
        return landmark_id in self.landmark_id_to_idx
    
    def get_map_statistics(self) -> Dict:
        """
        맵 통계 정보 반환.
        
        Returns:
            Dict: 맵 통계 정보
        """
        if not self.landmarks:
            return {
                'landmark_count': 0,
                'avg_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'avg_observations': 0.0,
                'total_observations': 0
            }
        
        uncertainties = []
        observations = []
        
        for landmark_id, info in self.landmarks.items():
            landmark_idx = info['idx']
            state_idx = self.robot_dim + landmark_idx * self.landmark_dim
            
            # 불확실성 계산 (대각 요소의 합)
            cov_slice = slice(state_idx, state_idx + self.landmark_dim)
            uncertainty = np.trace(self.P[cov_slice, cov_slice])
            uncertainties.append(uncertainty)
            
            # 관측 횟수
            observations.append(info['observation_count'])
        
        return {
            'landmark_count': len(self.landmarks),
            'avg_uncertainty': np.mean(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'min_uncertainty': np.min(uncertainties),
            'avg_observations': np.mean(observations),
            'total_observations': np.sum(observations)
        } 
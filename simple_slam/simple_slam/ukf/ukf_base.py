#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Unscented Kalman Filter (UKF) implementation.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
import time


class UKF:
    """
    기본 비향 칼만 필터(Unscented Kalman Filter) 구현.
    이 클래스는 비선형 시스템을 위한 UKF의 기본 알고리즘을 제공합니다.
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
                 residual_z: Optional[Callable] = None):
        """
        UKF 초기화.

        Args:
            dim_x (int): 상태벡터 차원
            dim_z (int): 측정벡터 차원
            fx (Callable): 상태 전이 함수 f(x, dt) - 상태를 다음 시간으로 전파
            hx (Callable): 측정 함수 h(x) - 상태를 측정 공간으로 변환
            dt (float): 시간 간격 (초)
            points (Callable, optional): 시그마 포인트 생성 함수
            sqrt_fn (Callable, optional): 행렬 제곱근 함수
            x_mean_fn (Callable, optional): 상태 평균 계산 함수
            z_mean_fn (Callable, optional): 측정 평균 계산 함수
            residual_x (Callable, optional): 상태 잔차 계산 함수
            residual_z (Callable, optional): 측정 잔차 계산 함수
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.dt = dt
        
        # 디폴트 함수 설정
        self.points_fn = points
        self.sqrt_fn = sqrt_fn or self.cholesky
        self.x_mean_fn = x_mean_fn or self.weighted_mean
        self.z_mean_fn = z_mean_fn or self.weighted_mean
        self.residual_x = residual_x or self.subtract
        self.residual_z = residual_z or self.subtract
        
        # 상태 초기화
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        
        # 프로세스 노이즈 및 측정 노이즈
        self.Q = np.eye(dim_x)     # 프로세스 노이즈
        self.R = np.eye(dim_z)     # 측정 노이즈
        
        # UKF 파라미터 (기본값)
        self.alpha = 0.1
        self.beta = 2.0
        self.kappa = 0.0
        
        # 시그마 포인트 가중치
        self.Wm = None  # 평균 가중치
        self.Wc = None  # 공분산 가중치
        self._compute_weights()
        
        # 마지막 예측 및 업데이트 시간
        self.last_predict_time = None
        self.last_update_time = None
        
        # 측정값 유효성 검사 임계값
        self.innovation_threshold = 3.0  # 마할라노비스 거리 임계값
        
        # 디버깅 및 통계
        self.innovation = None
        self.S = None  # 측정 잔차 공분산
        self.K = None  # 칼만 이득
        self.y = None  # 측정 잔차
        self.z = None  # 최근 측정값
        self.log_likelihood = 0.0
        
        # 마지막 시그마 포인트
        self.sigmas_f = None  # 예측 시그마 포인트
        self.sigmas_h = None  # 측정 시그마 포인트

    def initialize(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """
        UKF 상태 및 공분산 초기화.

        Args:
            x (np.ndarray): 초기 상태 벡터
            P (np.ndarray, optional): 초기 공분산 행렬
        """
        self.x = x
        if P is not None:
            self.P = P
        else:
            self.P = np.eye(self.dim_x)
        
        self.last_predict_time = time.time()
        self.last_update_time = time.time()
        self.log_likelihood = 0.0

    def predict(self, dt: Optional[float] = None, **fx_args):
        """
        UKF 예측 단계 (시간 갱신).
        현재 상태를 다음 시간으로 전파합니다.

        Args:
            dt (float, optional): 시간 간격. None이면 기본값 사용
            **fx_args: 상태 전이 함수 fx에 전달할 추가 인자
        """
        if dt is not None:
            self.dt = dt
        
        # 시그마 포인트 생성
        sigmas = self.sigma_points(self.x, self.P)
        
        # 시그마 포인트를 다음 시간으로 전파
        sigmas_f = np.zeros((2 * self.dim_x + 1, self.dim_x))
        for i in range(len(sigmas)):
            sigmas_f[i] = self.fx(sigmas[i], self.dt, **fx_args)
        
        self.sigmas_f = sigmas_f
        
        # 새로운 상태 및 공분산 계산
        self.x = self.x_mean_fn(sigmas_f, self.Wm)
        
        # 공분산 계산
        self.P = self.calculate_covariance(sigmas_f, self.x, self.Wc, self.residual_x)
        
        # 프로세스 노이즈 추가
        self.P += self.Q
        
        self.last_predict_time = time.time()

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None, **hx_args):
        """
        UKF 업데이트 단계 (측정 갱신).
        측정값을 사용하여 상태를 업데이트합니다.

        Args:
            z (np.ndarray): 측정 벡터
            R (np.ndarray, optional): 측정 노이즈 공분산. None이면 기본값 사용
            **hx_args: 측정 함수 hx에 전달할 추가 인자
        
        Returns:
            float: 마할라노비스 거리 (측정값의 유효성 지표)
        """
        if z is None:
            return
        
        if R is not None:
            self.R = R
        
        # 시그마 포인트가 없으면 예측 먼저 실행
        if self.sigmas_f is None:
            self.predict()
        
        # 시그마 포인트를 측정 공간으로 변환
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(len(self.sigmas_f)):
            sigmas_h[i] = self.hx(self.sigmas_f[i], **hx_args)
        
        self.sigmas_h = sigmas_h
        
        # 예측된 측정값 및 공분산 계산
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
        
        # 마할라노비스 거리 반환
        self.last_update_time = time.time()
        self.z = z
        
        return self.innovation

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        UKF 시그마 포인트 생성.
        기본적으로 표준 sigma-point 알고리즘을 사용하지만,
        points_fn이 제공되면 해당 함수를 사용합니다.

        Args:
            x (np.ndarray): 상태 벡터
            P (np.ndarray): 공분산 행렬
        
        Returns:
            np.ndarray: 시그마 포인트
        """
        if self.points_fn is not None:
            return self.points_fn(x, P, self)
        
        n = self.dim_x
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # 행렬 제곱근 계산
        U = self.sqrt_fn(P * (n + lambda_))
        
        # 시그마 포인트 배열
        sigmas = np.zeros((2*n + 1, n))
        sigmas[0] = x
        
        for i in range(n):
            sigmas[i+1] = x + U[i]
            sigmas[n+i+1] = x - U[i]
        
        return sigmas

    def cholesky(self, A: np.ndarray) -> np.ndarray:
        """
        콜레스키 분해를 사용한 행렬 제곱근 계산.
        
        Args:
            A (np.ndarray): 양의 정부호 행렬
            
        Returns:
            np.ndarray: 하삼각행렬 L, A = L*L.T
        """
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # 양의 정부호가 아닌 경우 처리
            # 대각선 항을 약간 증가시켜 양의 정부호로 만듦
            print("경고: 콜레스키 분해 실패. 행렬 정규화 시도")
            A_reg = A + np.eye(A.shape[0]) * 1e-3
            return np.linalg.cholesky(A_reg)

    def _compute_weights(self):
        """
        UKF 시그마 포인트 가중치 계산.
        """
        n = self.dim_x
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # 기본 가중치 계산
        self.Wm = np.zeros(2*n + 1)
        self.Wc = np.zeros(2*n + 1)
        
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            self.Wm[i] = 1 / (2 * (n + lambda_))
            self.Wc[i] = 1 / (2 * (n + lambda_))

    def calculate_covariance(self, sigmas: np.ndarray, mean: np.ndarray, 
                             Wc: np.ndarray, residual_fn: Callable) -> np.ndarray:
        """
        시그마 포인트로부터 공분산 행렬 계산.
        
        Args:
            sigmas (np.ndarray): 시그마 포인트
            mean (np.ndarray): 시그마 포인트의 가중 평균
            Wc (np.ndarray): 공분산 가중치
            residual_fn (Callable): 잔차 계산 함수
            
        Returns:
            np.ndarray: 공분산 행렬
        """
        n = sigmas.shape[1]
        cov = np.zeros((n, n))
        
        for i in range(len(sigmas)):
            # 가중치 * (시그마 포인트 - 평균) * (시그마 포인트 - 평균).T
            y = residual_fn(sigmas[i], mean)
            cov += Wc[i] * np.outer(y, y)
            
        return cov

    def calculate_cross_covariance(self, x_sigmas: np.ndarray, x_mean: np.ndarray, 
                                  z_sigmas: np.ndarray, z_mean: np.ndarray) -> np.ndarray:
        """
        상태와 측정 간의 교차 공분산 계산.
        
        Args:
            x_sigmas (np.ndarray): 상태 시그마 포인트
            x_mean (np.ndarray): 상태 평균
            z_sigmas (np.ndarray): 측정 시그마 포인트
            z_mean (np.ndarray): 측정 평균
            
        Returns:
            np.ndarray: 교차 공분산 행렬
        """
        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(len(x_sigmas)):
            dx = self.residual_x(x_sigmas[i], x_mean)
            dz = self.residual_z(z_sigmas[i], z_mean)
            Pxz += self.Wc[i] * np.outer(dx, dz)
            
        return Pxz

    def weighted_mean(self, sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
        """
        시그마 포인트의 가중 평균 계산.
        
        Args:
            sigmas (np.ndarray): 시그마 포인트
            Wm (np.ndarray): 평균 가중치
            
        Returns:
            np.ndarray: 가중 평균
        """
        return np.dot(Wm, sigmas)

    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        벡터의 차이 계산 (기본 잔차 함수).
        
        Args:
            a (np.ndarray): 첫 번째 벡터
            b (np.ndarray): 두 번째 벡터
            
        Returns:
            np.ndarray: a - b
        """
        return a - b

    def _log_likelihood(self, z: np.ndarray, z_mean: np.ndarray, S: np.ndarray) -> float:
        """
        측정값의 로그 가능도 계산.
        
        Args:
            z (np.ndarray): 측정 벡터
            z_mean (np.ndarray): 예측된 측정 벡터
            S (np.ndarray): 측정 잔차 공분산
            
        Returns:
            float: 로그 가능도
        """
        n = len(z)
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        
        # 마할라노비스 거리 제곱
        d = self.residual_z(z, z_mean)
        dist = np.dot(d.T, np.dot(inv, d))
        
        # 로그 가능도
        log_likelihood = -0.5 * (np.log(det) + dist + n * np.log(2 * np.pi))
        
        return float(log_likelihood)

    def set_alpha_beta_kappa(self, alpha: float, beta: float, kappa: float):
        """
        UKF 파라미터 설정 및 가중치 재계산.
        
        Args:
            alpha (float): UKF 알파 파라미터
            beta (float): UKF 베타 파라미터
            kappa (float): UKF 카파 파라미터
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 상태 및 공분산 반환.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (상태 벡터, 공분산 행렬)
        """
        return self.x, self.P
    
    def get_state_with_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        상태, 공분산 및 불확실성 지표를 포함한 정보 반환.
        
        Returns:
            Dict[str, np.ndarray]: 상태 및 불확실성 정보
        """
        # 대각선 요소에서 표준 편차 계산
        std_devs = np.sqrt(np.diag(self.P))
        
        return {
            'state': self.x.copy(),
            'covariance': self.P.copy(),
            'std_devs': std_devs,
            'eigenvalues': np.linalg.eigvals(self.P)
        }

    def get_last_update_info(self) -> Dict[str, Any]:
        """
        마지막 업데이트에 대한 정보 반환.
        
        Returns:
            Dict[str, Any]: 마지막 업데이트 정보
        """
        if self.z is None:
            return None
            
        return {
            'measurement': self.z.copy() if self.z is not None else None,
            'predicted_measurement': self.z_mean_fn(self.sigmas_h, self.Wm) if self.sigmas_h is not None else None,
            'innovation': self.y.copy() if self.y is not None else None,
            'innovation_covariance': self.S.copy() if self.S is not None else None,
            'kalman_gain': self.K.copy() if self.K is not None else None,
            'mahalanobis_distance': float(self.innovation) if self.innovation is not None else None,
            'log_likelihood': float(self.log_likelihood)
        }

    def reset(self):
        """
        UKF 상태 리셋.
        """
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)
        self.sigmas_f = None
        self.sigmas_h = None
        self.log_likelihood = 0.0
        self.last_predict_time = None
        self.last_update_time = None
        self.z = None
        self.y = None
        self.K = None
        self.S = None
        self.innovation = None 
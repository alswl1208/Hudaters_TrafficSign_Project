# -*- coding: utf-8 -*-
"""
Ego-compensated Kalman filter for DeepSORT-compatible API.

- State: [x, y, a, h, vx, vy, va, vh]
- predict(mean, P, ego=None) 에서 에고모션(affine/homography/shift/scale)을 반영
"""

from __future__ import annotations
import numpy as np
import scipy.linalg

__all__ = ["KalmanFilterEgo", "KalmanFilter"]

chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}

def _apply_affine_point(x, y, A):
    X = np.array([x, y, 1.0], dtype=float)
    q = A @ X
    return float(q[0]), float(q[1])

def _apply_homography_point(x, y, H):
    X = np.array([x, y, 1.0], dtype=float)
    q = H @ X
    return float(q[0] / max(q[2], 1e-8)), float(q[1] / max(q[2], 1e-8))

def _approx_isotropic_scale_from_affine(A):
    M = A[:, :2]
    svals = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(svals))

class KalmanFilterEgo(object):
    """
    8D state: [x, y, a, h, vx, vy, va, vh]
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        mean_pos = measurement.astype(float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, ego=None):
        # 기본 상수속도 예측
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        # 에고 보정
        if ego is not None:
            if 'H' in ego and ego['H'] is not None:
                H = ego['H']
                if H.shape == (2, 3):  # affine
                    x, y = _apply_affine_point(mean[0], mean[1], H)
                    ds = ego.get('ds', _approx_isotropic_scale_from_affine(H))
                else:  # homography
                    x, y = _apply_homography_point(mean[0], mean[1], H)
                    ds = ego.get('ds', 1.0)
                mean[0], mean[1] = x, y
                mean[2] += float(ego.get('da', 0.0))
                ds = float(ds if np.isfinite(ds) else 1.0)
                if ds != 1.0:
                    mean[3] *= ds
                    S = np.eye(8); S[3, 3] = ds; S[7, 7] = ds
                    covariance = S @ covariance @ S.T
            else:
                dx = float(ego.get('dx', 0.0))
                dy = float(ego.get('dy', 0.0))
                ds = float(ego.get('ds', 1.0))
                da = float(ego.get('da', 0.0))
                mean[0] += dx; mean[1] += dy; mean[2] += da
                if ds != 1.0:
                    mean[3] *= ds
                    S = np.eye(8); S[3, 3] = ds; S[7, 7] = ds
                    covariance = S @ covariance @ S.T
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True,
                                          check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)

# DeepSORT 호환 별칭
KalmanFilter = KalmanFilterEgo

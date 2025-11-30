# -*- coding: utf-8 -*-
"""
Ego-compensated Kalman filter for DeepSORT (robust version)
- 8D state: [x, y, a, h, vx, vy, va, vh]
- predict(mean, P, ego=None): supports ego = {'H': 3x3 homography or 2x3 affine,
  'dx','dy','ds','da'} subsets.
- Strong guards to prevent overflow / NaN propagation when applying ego motion.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg

__all__ = ["KalmanFilterEgo", "KalmanFilter", "chi2inv95"]

# 95% chi^2 for 4D and 2D gating (DeepSORT uses this)
chi2inv95 = {2: 5.9915, 4: 9.4877}

def _is_finite_array(x) -> bool:
    return np.isfinite(x).all()

def _nan_to_num_(x, lim=1e7):
    """In-place nan/inf clamp for vectors."""
    np.nan_to_num(x, copy=False, posinf=lim, neginf=-lim)
    x[:] = np.clip(x, -lim, lim)

def _apply_affine_point(x, y, A23):
    """A23: 2x3 affine"""
    X = np.array([x, y, 1.0], dtype=float)
    q = A23 @ X
    xp, yp = float(q[0]), float(q[1])
    return xp, yp

def _apply_homography_point(x, y, H):
    """Robust homography application with hard guards."""
    X = np.array([x, y, 1.0], dtype=float)
    q = H @ X
    denom = q[2]

    # denominator guard
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        return float(x), float(y)

    xp = float(q[0] / denom)
    yp = float(q[1] / denom)

    # result sanity guard (huge excursions are discarded)
    if (not np.isfinite(xp)) or (not np.isfinite(yp)) or \
       (abs(xp) > 1e6) or (abs(yp) > 1e6):
        return float(x), float(y)

    return xp, yp

def _approx_isotropic_scale_from_affine(A23):
    """Heuristic isotropic scale from a 2x3 affine matrix."""
    A = np.asarray(A23, dtype=float)[:, :2]
    # sqrt of average squared singular values -> isotropic scale
    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s is None or len(s) == 0:
            return 1.0
        return float(np.clip(np.mean(s), 1e-3, 1e3))
    except Exception:
        return 1.0


class KalmanFilterEgo:
    def __init__(self):
        ndim, dt = 4, 1.0

        # state: x, y, a, h, vx, vy, va, vh
        self._motion_mat = np.eye(2 * ndim, dtype=float)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim, dtype=float)

        # Default noise weights (DeepSORT convention)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ----------------------- noise floors -----------------------
    def _floor_pos_std(self, h: float) -> float:
        # min px noise for position
        return float(max(0.05 * max(h, 30.0), 8.0))

    def _floor_vel_std(self, h: float) -> float:
        # min px noise for velocity
        return float(max(0.01 * max(h, 30.0), 1.0))

    # ----------------------- API -----------------------
    def initiate(self, measurement: np.ndarray):
        """
        measurement z = [x, y, a, h]
        returns mean, covariance
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = float(measurement[3])
        pos_std = self._floor_pos_std(h)
        vel_std = self._floor_vel_std(h)

        std = [
            max(self._std_weight_position * h, pos_std),
            max(self._std_weight_position * h, pos_std),
            1e-2,  # a (ratio) loose
            max(self._std_weight_position * h, pos_std),
            max(self._std_weight_velocity * h, vel_std),
            max(self._std_weight_velocity * h, vel_std),
            1e-3,
            max(self._std_weight_velocity * h, vel_std),
        ]
        covariance = np.diag(np.square(std))
        return mean.astype(float), covariance.astype(float)

    def predict(self, mean: np.ndarray, covariance: np.ndarray, ego: dict | None = None):
        """Standard motion + ego compensation with guards."""
        # Motion noise adapted by current height
        h = float(max(mean[3], 1.0))
        pos_std = self._floor_pos_std(h)
        vel_std = self._floor_vel_std(h)

        std_pos = [
            max(self._std_weight_position * h, pos_std),
            max(self._std_weight_position * h, pos_std),
            1e-2,
            max(self._std_weight_position * h, pos_std),
        ]
        std_vel = [
            max(self._std_weight_velocity * h, vel_std),
            max(self._std_weight_velocity * h, vel_std),
            1e-3,
            max(self._std_weight_velocity * h, vel_std),
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # standard kinematic prediction
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        # ----- Ego-motion compensation (robust) -----
        if ego is not None:
            try:
                if 'H' in ego and ego['H'] is not None:
                    H = np.asarray(ego['H'], dtype=float)
                    if H.shape == (2, 3):
                        x_new, y_new = _apply_affine_point(mean[0], mean[1], H)
                        ds = ego.get('ds', _approx_isotropic_scale_from_affine(H))
                    else:
                        x_new, y_new = _apply_homography_point(mean[0], mean[1], H)
                        ds = ego.get('ds', 1.0)

                    # 2nd-level validation; if invalid, keep old mean
                    if np.isfinite([x_new, y_new]).all() and \
                       abs(x_new) <= 1e7 and abs(y_new) <= 1e7:
                        mean[0], mean[1] = x_new, y_new

                    mean[2] += float(ego.get('da', 0.0))
                    ds = float(ds if np.isfinite(ds) else 1.0)
                    if ds != 1.0 and 1e-3 <= ds <= 1e3:
                        mean[3] *= ds
                        S = np.eye(8)
                        S[3, 3] = ds; S[7, 7] = ds  # scale h and vh
                        covariance = S @ covariance @ S.T

                else:
                    # Shift/scale-only path
                    dx = float(ego.get('dx', 0.0))
                    dy = float(ego.get('dy', 0.0))
                    ds = float(ego.get('ds', 1.0))
                    da = float(ego.get('da', 0.0))

                    if np.isfinite([dx, dy]).all():
                        mean[0] += dx; mean[1] += dy
                    mean[2] += da
                    if ds != 1.0 and 1e-3 <= ds <= 1e3 and np.isfinite(ds):
                        mean[3] *= ds
                        S = np.eye(8); S[3, 3] = ds; S[7, 7] = ds
                        covariance = S @ covariance @ S.T

            except Exception:
                # Any ego failure -> ignore ego this step
                pass

        # Final hard clamps to prevent numeric blowups
        if not _is_finite_array(mean):
            _nan_to_num_(mean)

        # x,y bounded; h positive; a bounded to avoid absurd ratios
        mean[0] = float(np.clip(mean[0], -1e7, 1e7))
        mean[1] = float(np.clip(mean[1], -1e7, 1e7))
        mean[3] = float(np.clip(mean[3], 1.0, 1e7))
        if not np.isfinite(mean[2]) or abs(mean[2]) > 1e6:
            mean[2] = 0.0

        # covariance symmetrization + tiny floor
        covariance = 0.5 * (covariance + covariance.T)
        diag = np.diag(covariance)
        diag = np.maximum(diag, 1e-6)
        covariance[np.diag_indices_from(covariance)] = diag

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space."""
        h = float(max(mean[3], 1.0))
        pos_std = self._floor_pos_std(h)

        std = [
            max(self._std_weight_position * h, pos_std),
            max(self._std_weight_position * h, pos_std),
            1e-2,
            max(self._std_weight_position * h, pos_std),
        ]
        innovation_cov = np.diag(np.square(std))

        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ covariance @ self._update_mat.T

        # Stabilize projected covariance
        cov_proj = 0.5 * (cov_proj + cov_proj.T)
        d = np.diag(cov_proj)
        d = np.maximum(d, 1e-6)
        cov_proj[np.diag_indices_from(cov_proj)] = d

        return mean_proj, cov_proj + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman update for a single measurement z=[x, y, a, h]."""
        mean_proj, cov_proj = self.project(mean, covariance)

        # Cholesky can fail if cov becomes non-PD; guard by tiny jitter
        jitter = 0
        while True:
            try:
                chol = np.linalg.cholesky(cov_proj)
                break
            except np.linalg.LinAlgError:
                jitter += 1
                cov_proj[np.diag_indices_from(cov_proj)] += 1e-6
                if jitter > 5:
                    # last resort – symmetrize & bail out
                    cov_proj = 0.5 * (cov_proj + cov_proj.T)
                    chol = np.linalg.cholesky(cov_proj)
                    break

        # Kalman gain
        K = scipy.linalg.solve_triangular(chol, self._update_mat @ covariance, lower=True,
                                          check_finite=False, overwrite_b=False).T
        K = scipy.linalg.solve_triangular(chol.T, K.T, lower=False,
                                          check_finite=False, overwrite_b=False).T

        innovation = measurement - mean_proj
        new_mean = mean + K @ innovation
        new_cov = covariance - K @ (chol @ chol.T) @ K.T

        # Final guards
        if not _is_finite_array(new_mean):
            _nan_to_num_(new_mean)
        new_cov = 0.5 * (new_cov + new_cov.T)
        d = np.diag(new_cov)
        d = np.maximum(d, 1e-6)
        new_cov[np.diag_indices_from(new_cov)] = d

        # Keep measurement-domain constraints sensible
        new_mean[3] = float(max(new_mean[3], 1.0))
        if not np.isfinite(new_mean[2]) or abs(new_mean[2]) > 1e6:
            new_mean[2] = 0.0

        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray, only_position: bool = False):
        """
        Compute squared Mahalanobis distances between projected mean and
        measurements, optionally using only x,y.
        """
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj, cov_proj = mean_proj[:2], cov_proj[:2, :2]
            measurements = measurements[:, :2]
        # Use Cholesky for stability
        chol = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True,
                                          check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# Backwards-compatible alias for DeepSORT codebases
KalmanFilter = KalmanFilterEgo

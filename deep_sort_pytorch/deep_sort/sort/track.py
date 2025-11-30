# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Single target track with 8D state [x,y,a,h,vx,vy,va,vh].
    - 이번 프레임에서 사용된 '측정(detection)'을 저장하여 출력에 사용
    - 예측-측정 잔차가 크면 상태를 측정으로 재초기화(innovation reset)
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, class_id=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.class_id = class_id
        self.oid = class_id

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            if not isinstance(feature, np.ndarray):
                feature = np.asarray(feature, dtype=np.float32)
            elif feature.dtype != np.float32:
                feature = feature.astype(np.float32, copy=False)
            self.features.append(feature)

        self._n_init = int(n_init)
        self._max_age = int(max_age)

        # 이번 프레임 측정(출력용)
        self.last_det = None
        self.was_updated = False

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf, ego_H=None):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance, ego={'H': ego_H})
        self.increment_age()
        self.was_updated = False  # 새 프레임 시작

    def update(self, kf, detection):
        """
        Innovation을 확인하여 크게 빗나가면 상태를 '측정으로 재시작'.
        그 외에는 표준 Kalman update 수행.
        """
        projected_mean, projected_cov = kf.project(self.mean, self.covariance)
        innovation = detection.to_xyah() - projected_mean
        try:
            L = np.linalg.cholesky(projected_cov)
            maha = np.sum(np.square(np.linalg.solve(L, innovation)))
        except np.linalg.LinAlgError:
            maha = np.inf

        RESET_THR = 16.0  # 표지판 등 작은 타깃 기준 경험적 임계
        if maha > RESET_THR:
            self.mean, self.covariance = kf.initiate(detection.to_xyah())
        else:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.to_xyah()
            )

        # feature append
        feat = detection.feature
        if feat is not None:
            if not isinstance(feat, np.ndarray):
                feat = np.asarray(feat, dtype=np.float32)
            elif feat.dtype != np.float32:
                feat = feat.astype(np.float32, copy=False)
            self.features.append(feat)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # 이번 프레임 측정 저장 (출력에 사용)
        self.last_det = detection
        self.was_updated = True

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

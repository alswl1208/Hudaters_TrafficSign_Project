# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState


def _iou_tlwh(a, b):
    """IoU for tlwh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-9)


class Tracker:
    """
    DeepSORT tracker with ego-motion compensated Kalman filter
    + ID-bridge to keep identities stable through short gaps.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=2):
        self.metric = metric
        self.max_iou_distance = float(max_iou_distance)
        self.max_age = int(max_age)
        self.n_init = int(n_init)

        self.kf = kalman_filter.KalmanFilterEgo()

        self.tracks = []
        self._next_id = 1

    # --------------------------- lifecycle ---------------------------
    def predict(self, ego_H=None):
        """
        Propagate all tracks one step (optionally with ego-motion).
        """
        for track in self.tracks:
            track.predict(self.kf, ego_H=ego_H)

    def increment_ages(self):
        """
        Increase age and mark tracks missed (useful when a frame is skipped).
        """
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    # --------------------------- association ---------------------------
    def update(self, detections):
        """
        Perform measurement update and manage tracks.
        1) Appearance matching cascade (gated)
        2) IoU fallback
        3) Initiate new tracks
        4) ID-bridge: newly created tentative tracks may inherit ID from
           recently missed confirmed tracks if they spatially overlap.
        """
        # 1-2) matching
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # matched updates
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # unmatched tracks: miss
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # unmatched detections: new tracks
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # -------- ID BRIDGE --------
        # 새로 생긴 트랙(Tentative & hits==1)이 있고,
        # 최근(time_since_update<=3) 놓친 Confirmed 트랙이 있으면,
        # IoU >= 0.30이면 새 트랙이 옛 트랙의 ID를 승계한다.
        NEW_IOU_THR = 0.30
        RECENT_MISS_MAX = 3

        new_tracks = [t for t in self.tracks
                      if t.state == TrackState.Tentative and t.hits == 1]
        recent_missed = [t for t in self.tracks
                         if t.state == TrackState.Confirmed and 1 <= t.time_since_update <= RECENT_MISS_MAX]

        for nt in new_tracks:
            nt_box = nt.to_tlwh()
            best, best_iou = None, 0.0
            for ot in recent_missed:
                iou = _iou_tlwh(nt_box, ot.to_tlwh())
                if iou > best_iou:
                    best, best_iou = ot, iou
            if best is not None and best_iou >= NEW_IOU_THR:
                # 새 트랙이 옛 트랙의 ID를 승계
                old_id = best.track_id
                nt.track_id = old_id
                nt.state = TrackState.Confirmed
                nt.hits = max(nt.hits, best.hits + 1)
                # 예전 트랙은 삭제로 정리
                best.state = TrackState.Deleted

        # 삭제된 트랙 제거
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # metric 업데이트
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        if len(features) > 0:
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets
            )

    def _match(self, detections):
        """
        Appearance-first matching + IoU fallback (DeepSORT standard).
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            feats = np.array([dets[i].feature for i in detection_indices])
            tids = np.array([tracks[i].track_id for i in track_indices])
            cost = self.metric.distance(feats, tids)
            cost = linear_assignment.gate_cost_matrix(
                self.kf, cost, tracks, dets, track_indices, detection_indices
            )
            return cost

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks
            )

        iou_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_candidates, unmatched_detections
            )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    # --------------------------- track create ---------------------------
    def _initiate_track(self, detection):
        """
        Create a new track from detection.
        class_id를 detection.oid 또는 detection.class_id에서 가져와 저장.
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_id = getattr(detection, "oid", getattr(detection, "class_id", None))
        self.tracks.append(
            Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                  feature=detection.feature, class_id=class_id)
        )
        self._next_id += 1

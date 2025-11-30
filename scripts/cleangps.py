import pandas as pd
import numpy as np
from collections import Counter
import math

EARTH_R = 6378137.0  # meters
RADIUS_M = 50.0      # <-- 반경 20m

def _mode_safe(series):
    try:
        return series.mode(dropna=True).iloc[0]
    except Exception:
        vals = [v for v in series if pd.notnull(v)]
        return Counter(vals).most_common(1)[0][0] if vals else np.nan

def _haversine_m(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def _pick_best_latlon(df):
    """sign_lat/lon 우선, 없으면 cam_lat/lon로 대체한 lat/lon 컬럼을 추가."""
    has_sign = df.columns.__contains__("sign_lat") and df.columns.__contains__("sign_lon")
    has_cam  = df.columns.__contains__("cam_lat") and df.columns.__contains__("cam_lon")
    lat = df["sign_lat"] if has_sign else (df["cam_lat"] if has_cam else np.nan)
    lon = df["sign_lon"] if has_sign else (df["cam_lon"] if has_cam else np.nan)
    if has_cam:
        lat = pd.to_numeric(lat, errors="coerce").fillna(pd.to_numeric(df["cam_lat"], errors="coerce"))
        lon = pd.to_numeric(lon,  errors="coerce").fillna(pd.to_numeric(df["cam_lon"], errors="coerce"))
    else:
        lat = pd.to_numeric(lat, errors="coerce")
        lon = pd.to_numeric(lon, errors="coerce")
    df["lat"] = lat
    df["lon"] = lon
    return df

def _cluster_indices_dbscan(latlon_deg, radius_m):
    """DBSCAN(haversine)로 클러스터 인덱스 반환. scikit-learn 없으면 None."""
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        return None
    # haversine은 라디안 좌표 기준, eps는 라디안 거리
    latlon_rad = np.radians(latlon_deg)
    eps = radius_m / EARTH_R
    labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit(latlon_rad).labels_
    return labels

def _cluster_indices_greedy(latlon_deg, radius_m):
    """간단한 그리디: 기존 클러스터의 '현재 평균 중심'과의 거리 ≤ R 이면 편입."""
    clusters = []  # list of dict: {"idxs":[...], "center":[lat,lon]}
    for i, (la, lo) in enumerate(latlon_deg):
        if not (np.isfinite(la) and np.isfinite(lo)):
            # 좌표 불가 → 자기 자신을 개별 클러스터로
            clusters.append({"idxs":[i], "center":[la, lo]})
            continue
        placed = False
        for c in clusters:
            cla, clo = c["center"]
            d = _haversine_m(la, lo, cla, clo)
            if d <= radius_m:
                c["idxs"].append(i)
                # 중심 업데이트 (산술평균)
                pts = np.array([latlon_deg[j] for j in c["idxs"] if np.isfinite(latlon_deg[j][0]) and np.isfinite(latlon_deg[j][1])])
                if len(pts):
                    c["center"] = [float(pts[:,0].mean()), float(pts[:,1].mean())]
                placed = True
                break
        if not placed:
            clusters.append({"idxs":[i], "center":[la, lo]})
    # 라벨 반환
    labels = np.full(len(latlon_deg), -1, dtype=int)
    for cid, c in enumerate(clusters):
        for j in c["idxs"]:
            labels[j] = cid
    return labels

def _cluster_indices(latlon_deg, radius_m):
    labels = _cluster_indices_dbscan(latlon_deg, radius_m)
    if labels is not None:
        return labels
    return _cluster_indices_greedy(latlon_deg, radius_m)

def clean_csv(in_path, out_path):
    df = pd.read_csv(in_path, encoding="utf-8-sig")

    # 숫자형 변환
    for col in ["cam_lat","cam_lon","sign_lat","sign_lon","center_x_px","center_y_px","width_px","height_px","area_px2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 라벨/좌표 준비
    if "cls_name" not in df.columns:
        raise ValueError("Input CSV must contain 'cls_name'.")
    df = _pick_best_latlon(df)

    # cls_name별로 20m 클러스터링
    out_rows = []
    for cls, part in df.groupby("cls_name", dropna=False):
        part = part.reset_index(drop=True)
        latlon = part[["lat","lon"]].to_numpy()

        labels = _cluster_indices(latlon, RADIUS_M)
        part["cluster_id"] = labels

        # 각 클러스터별로 집계
        for cid, grp in part.groupby("cluster_id"):
            # 원래 요구된 평균값들
            row = {
                "cls_name": cls,
                "cluster_id": int(cid),
                "n_frames": int(len(grp)),
                "cam_lat": float(grp["cam_lat"].mean()) if "cam_lat" in grp else np.nan,
                "cam_lon": float(grp["cam_lon"].mean()) if "cam_lon" in grp else np.nan,
                "sign_lat": float(grp["sign_lat"].mean()) if "sign_lat" in grp else np.nan,
                "sign_lon": float(grp["sign_lon"].mean()) if "sign_lon" in grp else np.nan,
                # 대표 좌표(lat/lon)도 같이 넣어두면 이후 매칭에 편함
                "rep_lat": float(grp["lat"].mean()),
                "rep_lon": float(grp["lon"].mean()),
            }
            # 선택 컬럼 평균
            for col in ["center_x_px","center_y_px","width_px","height_px","area_px2"]:
                if col in grp.columns:
                    row[col] = float(grp[col].mean())
            # yolo_class 최빈값 유지
            if "yolo_class" in grp.columns:
                row["yolo_class_mode"] = _mode_safe(grp["yolo_class"])
            out_rows.append(row)

    agg = pd.DataFrame(out_rows)

    # 컬럼 순서 정리
    base_cols = ["cls_name","cluster_id","n_frames"]
    if "yolo_class_mode" in agg.columns:
        base_cols.append("yolo_class_mode")
    gps_cols = ["cam_lat","cam_lon","sign_lat","sign_lon","rep_lat","rep_lon"]
    other = [c for c in ["center_x_px","center_y_px","width_px","height_px","area_px2"] if c in agg.columns]
    agg = agg[base_cols + gps_cols + other]

    agg.to_csv(out_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    # 입력 경로는 너가 쓰던 경로에 맞춤
    clean_csv("before/yoloold.csv", "yoloold_clean.csv")
    clean_csv("after/yolonew.csv", "yolonew_clean.csv")
    print("Saved yoloold_clean.csv and yolonew_clean.csv (cls별 20m 클러스터링 반영)")

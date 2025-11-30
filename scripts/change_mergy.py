# apply_changes_and_export.py (strict change semantics)
# -*- coding: utf-8 -*-
"""
입력:
  - yoloold_clean.csv
  - changes_radius30.csv (status ∈ {appeared_new, changed, removed})
출력:
  - yolomerged_clean.csv
  - yolomerged_map.html

핵심 수정:
  - status == "changed"일 때,
      1) ref old 좌표(old_lat/old_lon) 주변 R 이내의 'old_cls'를 모두 제거
      2) 추가로, 새 좌표(new_lat/new_lon) 주변 R 이내에서 'new_cls'가 아닌 표지판을 제거(겹침 방지 옵션)
"""

import os
import numpy as np
import pandas as pd

RADIUS_M = 30.0
EARTH_R = 6378137.0  # meters

PATH_OLD      = "yoloold_clean.csv"
PATH_CHANGES  = "changes_radius30.csv"

OUT_MERGED_CSV = "yolomerged_clean.csv"
OUT_MERGED_MAP = "yolomerged_map.html"

# ---------- utils ----------
def haversine_m(lat1, lon1, lat2, lon2):
    """vectorized: lat1/lon1 can be arrays, lat2/lon2 scalars"""
    lat1 = np.asarray(lat1, dtype=float); lon1 = np.asarray(lon1, dtype=float)
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def to_num(x): return pd.to_numeric(x, errors="coerce")

def pick_best_latlon_row(row):
    lat = row.get("sign_lat", np.nan); lon = row.get("sign_lon", np.nan)
    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("cam_lat", np.nan); lon = row.get("cam_lon", np.nan)
    return to_num(lat), to_num(lon)

def normalize_old(df_old: pd.DataFrame) -> pd.DataFrame:
    df = df_old.copy()
    for c in ["sign_lat","sign_lon","cam_lat","cam_lon"]:
        if c in df.columns: df[c] = to_num(df[c])

    rows = []
    for _, r in df.iterrows():
        cls = str(r.get("cls_name", "")).strip()
        if not cls: continue
        lat, lon = pick_best_latlon_row(r)
        if pd.notna(lat) and pd.notna(lon):
            rows.append({"cls_name": cls, "sign_lat": float(lat), "sign_lon": float(lon)})
    return pd.DataFrame(rows)

def drop_indices(df, idxs):
    if len(idxs) == 0: return df
    return df.drop(index=list(idxs), errors="ignore")

def find_within_radius(df, lat0, lon0, radius_m, cls_filter=None, invert_cls=False):
    """
    반환: 조건에 맞는 행 index 배열
      - cls_filter: None이면 클래스 무관
      - cls_filter: '속도70'처럼 문자열이면
          invert_cls=False -> 해당 클래스만
          invert_cls=True  -> 해당 클래스가 아닌 것
    """
    if df.empty: return np.array([], dtype=int)
    lat = df["sign_lat"].to_numpy(float)
    lon = df["sign_lon"].to_numpy(float)
    d = haversine_m(lat, lon, lat0, lon0)
    sel = d <= float(radius_m)

    if cls_filter is not None:
        cls = df["cls_name"].astype(str).to_numpy()
        if invert_cls:
            sel &= (cls != str(cls_filter))
        else:
            sel &= (cls == str(cls_filter))

    return df.index.to_numpy()[sel]

# ---------- apply changes with stricter rules ----------
def apply_changes_strict(df_old_norm: pd.DataFrame, df_changes: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    df = df_old_norm.copy()

    # 1) removed : 해당 old 좌표 반경 R 이내의 같은 클래스 전부 삭제
    removed = df_changes[df_changes["status"] == "removed"].copy()
    for _, r in removed.iterrows():
        old_cls, old_lat, old_lon = r["old_cls"], r["old_lat"], r["old_lon"]
        if pd.isna(old_lat) or pd.isna(old_lon) or not str(old_cls).strip():
            continue
        idxs = find_within_radius(df, float(old_lat), float(old_lon), radius_m,
                                  cls_filter=str(old_cls), invert_cls=False)
        df = drop_indices(df, idxs)

    # 2) changed : (A) old 주변의 old_cls 모두 삭제  (B) new 주변의 new_cls가 아닌 것도 삭제(옵션)  (C) new 추가
    changed = df_changes[df_changes["status"] == "changed"].copy()
    for _, r in changed.iterrows():
        old_cls, old_lat, old_lon = r["old_cls"], r["old_lat"], r["old_lon"]
        new_cls, new_lat, new_lon = r["new_cls"], r["new_lat"], r["new_lon"]

        # (A) old 근처의 old_cls 모두 제거
        if pd.notna(old_lat) and pd.notna(old_lon) and str(old_cls).strip():
            idxs_old = find_within_radius(df, float(old_lat), float(old_lon), radius_m,
                                          cls_filter=str(old_cls), invert_cls=False)
            df = drop_indices(df, idxs_old)

        # (B) new 근처의 'new_cls가 아닌' 것 제거 (겹침 방지; 위치가 거의 같으나 클래스만 다른 잔존물 제거)
        if pd.notna(new_lat) and pd.notna(new_lon):
            idxs_bad = find_within_radius(df, float(new_lat), float(new_lon), radius_m,
                                          cls_filter=str(new_cls), invert_cls=True)
            df = drop_indices(df, idxs_bad)

        # (C) new 추가
        if pd.notna(new_lat) and pd.notna(new_lon) and str(new_cls).strip():
            df = pd.concat([df, pd.DataFrame([{
                "cls_name": str(new_cls).strip(),
                "sign_lat": float(new_lat),
                "sign_lon": float(new_lon),
            }])], ignore_index=True)

    # 3) appeared_new : 그냥 추가 (이 경우, 원 정의상 반경 R 내 old가 없어야 정상)
    appeared = df_changes[df_changes["status"] == "appeared_new"].copy()
    for _, r in appeared.iterrows():
        new_cls, new_lat, new_lon = r["new_cls"], r["new_lat"], r["new_lon"]
        if pd.notna(new_lat) and pd.notna(new_lon) and str(new_cls).strip():
            df = pd.concat([df, pd.DataFrame([{
                "cls_name": str(new_cls).strip(),
                "sign_lat": float(new_lat),
                "sign_lon": float(new_lon),
            }])], ignore_index=True)

    # (선택) 동일 클래스가 R의 매우 작은 분수(예: 5m) 안에 중복 존재 시, 한 개만 남기고 정리하고 싶다면 아래를 켜세요.
    # df = consolidate_nearby_same_class(df, merge_radius_m=5.0)

    return df.reset_index(drop=True)

# ---------- optional consolidator (off by default) ----------
def consolidate_nearby_same_class(df, merge_radius_m=5.0):
    if df.empty: return df
    keep = np.ones(len(df), dtype=bool)
    coords = df[["sign_lat","sign_lon"]].to_numpy(float)
    classes = df["cls_name"].astype(str).to_numpy()
    for i in range(len(df)):
        if not keep[i]: continue
        # 같은 클래스이면서 i보다 뒤쪽에서 아주 가까운 항목 제거
        d = haversine_m(coords[i+1:,0], coords[i+1:,1], coords[i,0], coords[i,1])
        dup = (classes[i+1:] == classes[i]) & (d <= merge_radius_m)
        keep[i+1:] &= ~dup
    return df.iloc[keep].copy().reset_index(drop=True)

# ---------- map ----------
def draw_merged_map(df_merged, outfile, radius_m):
    try:
        import folium
        from folium import LayerControl

        if df_merged.empty:
            m = folium.Map(location=[36.5, 127.5], zoom_start=7, control_scale=True, tiles="OpenStreetMap")
            m.save(outfile); print(f"[OK] Saved EMPTY map: {outfile}"); return

        center = [float(df_merged["sign_lat"].mean()), float(df_merged["sign_lon"].mean())]
        m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")

        for _, r in df_merged.iterrows():
            lat, lon, cls = float(r["sign_lat"]), float(r["sign_lon"]), str(r["cls_name"])
            folium.Circle(location=[lat, lon], radius=radius_m, color="#1E90FF",
                          fill=False, weight=1, opacity=0.5).add_to(m)
            folium.CircleMarker(location=[lat, lon], radius=5, color="#1E90FF",
                                fill=True, fill_opacity=0.95,
                                popup=f"{cls}", tooltip=cls).add_to(m)

        legend_html = f"""
        <div style="
            position: fixed; bottom: 20px; left: 20px; z-index:9999;
            background: white; padding: 10px 12px; border:1px solid #ccc; border-radius:6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size:14px;">
            <div style="font-weight:600; margin-bottom:6px;">Merged (R = {int(radius_m)} m)</div>
            <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #1E90FF;margin-right:6px;border-radius:50%;"></span>Current signs</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        LayerControl(collapsed=False).add_to(m)
        m.save(outfile)
        print(f"[OK] Saved: {outfile}")
    except Exception as e:
        print(f"[INFO] Map generation failed: {e}")

# ---------- main ----------
def main():
    if not os.path.exists(PATH_OLD):
        raise FileNotFoundError(PATH_OLD)
    if not os.path.exists(PATH_CHANGES):
        raise FileNotFoundError(PATH_CHANGES)

    df_old_raw = pd.read_csv(PATH_OLD, encoding="utf-8-sig")
    df_changes = pd.read_csv(PATH_CHANGES, encoding="utf-8-sig")

    # 필수 컬럼 확인
    need_cols = {"status","new_cls","new_lat","new_lon","old_cls","old_lat","old_lon"}
    missing = [c for c in need_cols if c not in df_changes.columns]
    if missing:
        raise ValueError(f"changes csv missing columns: {missing}")

    df_old_norm = normalize_old(df_old_raw)
    merged = apply_changes_strict(df_old_norm, df_changes, RADIUS_M)

    # 저장
    merged.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {OUT_MERGED_CSV} (rows={len(merged)})")

    draw_merged_map(merged, OUT_MERGED_MAP, RADIUS_M)

if __name__ == "__main__":
    main()

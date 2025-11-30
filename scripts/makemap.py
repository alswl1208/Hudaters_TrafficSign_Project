# compare_radius10.py  (integrated v3: ONLY 2 outputs)
# -*- coding: utf-8 -*-
"""
출력은 단 2개:
1) compare_radius30_map.html  : 표지판 비교 지도
2) changes_radius30.csv       : 결국 뭐가 바뀌었는지(appeared_new / changed / removed)만 담은 CSV
"""

import os
import json
import numpy as np
import pandas as pd

RADIUS_M = 30.0
EARTH_R = 6378137.0  # meters

PATH_OLD = "yoloold_clean.csv"
# NEW는 yolonew.csv 우선, 없으면 yolonew_clean.csv 폴백
PATH_NEW_PRIMARY = "yolonew.csv"
PATH_NEW_FALLBACK = "yolonew_clean.csv"

# 최종 출력 파일 2개만!
OUT_MAP      = "compare_radius30_map.html"
OUT_CHANGES  = "changes_radius30.csv"

def haversine_m(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def pick_best_latlon(row):
    lat = row.get("sign_lat", np.nan); lon = row.get("sign_lon", np.nan)
    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("cam_lat", np.nan); lon = row.get("cam_lon", np.nan)
    return pd.to_numeric(lat, errors="coerce"), pd.to_numeric(lon, errors="coerce")

def main():
    if not os.path.exists(PATH_OLD):
        raise FileNotFoundError(PATH_OLD)

    # NEW 파일 선택
    if os.path.exists(PATH_NEW_PRIMARY):
        PATH_NEW = PATH_NEW_PRIMARY
    elif os.path.exists(PATH_NEW_FALLBACK):
        PATH_NEW = PATH_NEW_FALLBACK
    else:
        raise FileNotFoundError(f"{PATH_NEW_PRIMARY} (또는 {PATH_NEW_FALLBACK})")

    old = pd.read_csv(PATH_OLD, encoding="utf-8-sig")
    new = pd.read_csv(PATH_NEW, encoding="utf-8-sig")

    for c in ["sign_lat","sign_lon","cam_lat","cam_lon"]:
        if c in old.columns: old[c] = pd.to_numeric(old[c], errors="coerce")
        if c in new.columns: new[c] = pd.to_numeric(new[c], errors="coerce")

    # --- OLD dict & list ---
    dict_old_by_class = {}
    old_points = []   # [(lat, lon, cls)]
    for _, r in old.iterrows():
        cls = str(r.get("cls_name", "")).strip()
        if not cls:
            continue
        lat, lon = pick_best_latlon(r)
        if pd.notna(lat) and pd.notna(lon):
            dict_old_by_class[cls] = (float(lat), float(lon))
            old_points.append((float(lat), float(lon), cls))

    # --- NEW 포인트 목록 ---
    new_points = []
    for _, r in new.iterrows():
        cls = str(r.get("cls_name", "")).strip()
        lat, lon = pick_best_latlon(r)
        if pd.notna(lat) and pd.notna(lon):
            new_points.append((float(lat), float(lon), cls))

    # --- NEW 기준 스캔(동일/변경/추가) ---
    new_scan_rows = []  # 지도/후처리를 위해 유지
    for (lat_n, lon_n, cls_n) in new_points:
        if cls_n in dict_old_by_class:
            lat_o, lon_o = dict_old_by_class[cls_n]
            d = float(haversine_m(lat_n, lon_n, lat_o, lon_o))
            if d <= RADIUS_M:
                decision = "same_sign"
                ref_old_cls, ref_old_lat, ref_old_lon = cls_n, lat_o, lon_o
                dist = round(d, 2)
            else:
                decision = "appeared_new"
                ref_old_cls = ref_old_lat = ref_old_lon = None
                dist = None
        else:
            # 같은 cls 없음 → old 전체 중 R 이내가 있는지 검사
            if old_points:
                dists = [haversine_m(lat_n, lon_n, lt, ln) for (lt, ln, _c) in old_points]
                j = int(np.argmin(dists)); d = float(dists[j])
                lt, ln, c_old = old_points[j]
                if d <= RADIUS_M:
                    decision = f"changed_from_{c_old}_to_{cls_n}"
                    ref_old_cls, ref_old_lat, ref_old_lon = c_old, lt, ln
                    dist = round(d, 2)
                else:
                    decision = "appeared_new"
                    ref_old_cls = ref_old_lat = ref_old_lon = None
                    dist = None
            else:
                decision = "appeared_new"
                ref_old_cls = ref_old_lat = ref_old_lon = None
                dist = None

        new_scan_rows.append({
            "new_cls": cls_n, "new_lat": lat_n, "new_lon": lon_n,
            "decision": decision,
            "ref_old_cls": ref_old_cls, "ref_old_lat": ref_old_lat, "ref_old_lon": ref_old_lon,
            "distance_m": dist
        })

    # --- OLD 기준 제거 탐지(removed) ---
    removed_rows = []
    for (lat_o, lon_o, cls_o) in old_points:
        if new_points:
            dists = [haversine_m(lat_o, lon_o, lt, ln) for (lt, ln, _c) in new_points]
            j = int(np.argmin(dists)); d = float(dists[j])
            lt_n, ln_n, cls_n = new_points[j]
            if d > RADIUS_M:
                removed_rows.append({
                    "old_cls": cls_o, "old_lat": lat_o, "old_lon": lon_o,
                    "nearest_new_cls": cls_n, "nearest_new_lat": lt_n, "nearest_new_lon": ln_n,
                    "nearest_distance_m": round(d, 2)
                })
        else:
            removed_rows.append({
                "old_cls": cls_o, "old_lat": lat_o, "old_lon": lon_o,
                "nearest_new_cls": None, "nearest_new_lat": None, "nearest_new_lon": None,
                "nearest_distance_m": None
            })

    # --- 최종 CSV(변경 사항만) 구성 ---
    changes = []

    # NEW 쪽: appeared_new / changed만 추림 (same_sign 제외)
    for r in new_scan_rows:
        dec = r["decision"]
        if dec == "appeared_new":
            changes.append({
                "status": "appeared_new",
                "new_cls": r["new_cls"], "new_lat": r["new_lat"], "new_lon": r["new_lon"],
                "old_cls": None, "old_lat": None, "old_lon": None,
                "distance_m": r["distance_m"]
            })
        elif dec.startswith("changed_from_"):
            # changed_from_{old}_to_{new}
            old_cls = r["ref_old_cls"]
            changes.append({
                "status": "changed",
                "new_cls": r["new_cls"], "new_lat": r["new_lat"], "new_lon": r["new_lon"],
                "old_cls": old_cls, "old_lat": r["ref_old_lat"], "old_lon": r["ref_old_lon"],
                "distance_m": r["distance_m"]
            })
        # same_sign은 CSV에 넣지 않음

    # OLD 쪽: removed
    for rr in removed_rows:
        changes.append({
            "status": "removed",
            "new_cls": None, "new_lat": None, "new_lon": None,
            "old_cls": rr["old_cls"], "old_lat": rr["old_lat"], "old_lon": rr["old_lon"],
            "distance_m": rr["nearest_distance_m"]
        })

    df_changes = pd.DataFrame(changes, columns=[
        "status",
        "new_cls","new_lat","new_lon",
        "old_cls","old_lat","old_lon",
        "distance_m"
    ])
    df_changes.to_csv(OUT_CHANGES, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {OUT_CHANGES}")

    # ===== 지도: 비교 지도 (compare_radius30_map.html) =====
    try:
        import folium
        from folium import FeatureGroup, LayerControl

        all_lat = [lt for (lt, ln, _) in old_points] + [r["new_lat"] for r in new_scan_rows]
        all_lon = [ln for (lt, ln, _) in old_points] + [r["new_lon"] for r in new_scan_rows]
        center = [np.mean(all_lat) if all_lat else 36.5, np.mean(all_lon) if all_lon else 127.5]

        m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")

        g_old   = FeatureGroup(name=f"OLD (radius {int(RADIUS_M)} m)")
        g_same  = FeatureGroup(name="NEW same_sign")
        g_chg   = FeatureGroup(name="NEW changed")
        g_app   = FeatureGroup(name="NEW appeared_new")
        g_rem   = FeatureGroup(name="OLD removed")

        for (lt, ln, cls) in old_points:
            folium.Circle(location=[lt, ln], radius=RADIUS_M, color="#1E90FF",
                          fill=False, weight=1, opacity=0.8).add_to(g_old)
            folium.CircleMarker(location=[lt, ln], radius=5, color="#1E90FF", fill=True, fill_opacity=0.9,
                                popup=f"OLD: {cls}", tooltip=f"OLD: {cls}").add_to(g_old)

        for r in new_scan_rows:
            lat, lon, cls, dec = r["new_lat"], r["new_lon"], r["new_cls"], r["decision"]
            if dec == "same_sign":
                color = "#00B050"; grp = g_same
            elif dec.startswith("changed_from_"):
                color = "#FF8C00"; grp = g_chg
            else:
                color = "#DC143C"; grp = g_app
            popup = f"NEW: {cls}<br>{dec}"
            if r.get("distance_m") is not None:
                popup += f"<br>d={r['distance_m']} m"
            folium.CircleMarker(location=[lat, lon], radius=5, color=color, fill=True, fill_opacity=0.9,
                                popup=popup, tooltip=f"NEW: {cls}").add_to(grp)

            if dec.startswith("changed_from_") and r["ref_old_lat"] and r["ref_old_lon"]:
                folium.PolyLine(locations=[[r["ref_old_lat"], r["ref_old_lon"]], [lat, lon]],
                                weight=2, color="#FF8C00", opacity=0.7).add_to(grp)

        for rr in removed_rows:
            folium.CircleMarker(location=[rr["old_lat"], rr["old_lon"]], radius=6, color="#6A5ACD",
                                fill=True, fill_opacity=0.95,
                                popup=f"OLD removed: {rr['old_cls']}", tooltip=f"REMOVED: {rr['old_cls']}").add_to(g_rem)

        g_old.add_to(m); g_same.add_to(m); g_chg.add_to(m); g_app.add_to(m); g_rem.add_to(m)
        LayerControl(collapsed=False).add_to(m)

        legend_html = f"""
        <div style="
            position: fixed; bottom: 20px; left: 20px; z-index:9999;
            background: white; padding: 10px 12px; border:1px solid #ccc; border-radius:6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size:14px;">
            <div style="font-weight:600; margin-bottom:6px;">Legend (R = {int(RADIUS_M)} m)</div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#00B050;margin-right:6px;border-radius:50%;"></span>NEW same_sign (동일)</div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#FF8C00;margin-right:6px;border-radius:50%;"></span>NEW changed (변경)</div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#DC143C;margin-right:6px;border-radius:50%;"></span>NEW appeared (추가)</div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#6A5ACD;margin-right:6px;border-radius:50%;"></span>OLD removed (제거)</div>
            <div style="margin-top:6px;"><span style="display:inline-block;width:12px;height:12px;border:2px solid #1E90FF;margin-right:6px;border-radius:50%;"></span>OLD {int(RADIUS_M)} m 반경</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        m.save(OUT_MAP)
        print(f"[OK] Saved: {OUT_MAP}")
    except Exception as e:
        print(f"[INFO] folium unavailable or map save failed: {e}")

if __name__ == "__main__":
    main()

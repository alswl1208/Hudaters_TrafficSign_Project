# make_yolo_cluster_map.py
# Folium 지도에 yoloold_clean.csv의 클러스터(대표 좌표) 표시

import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path

def make_cluster_map(
    csv_path: str,
    output_html: str = "yoloold_clusters_map.html",
    radius_m: float = 50.0,  # 클러스터 반경(시각화용 원의 반지름)
    zoom_start: int = 15,
    show_per_class_layers: bool = True,
):
    """
    csv_path: clean_csv로 만든 파일 경로 (예: yoloold_clean.csv)
    output_html: 저장될 HTML 지도 파일명
    radius_m: 각 클러스터 중심에 그릴 원의 반경(미터)
    zoom_start: 초기 줌 레벨
    show_per_class_layers: 클래스별 레이어를 분리해 토글 가능하게 표시할지 여부
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 필수 컬럼 존재 확인
    required_cols = {"cls_name", "cluster_id", "n_frames", "rep_lat", "rep_lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # NaN 제거 (대표 좌표가 없는 경우 지도에 못 올림)
    df = df.dropna(subset=["rep_lat", "rep_lon"])

    if df.empty:
        raise ValueError("No valid rows with rep_lat/rep_lon to plot.")

    # 지도 중심점: 대표 좌표 평균
    center_lat = float(df["rep_lat"].mean())
    center_lon = float(df["rep_lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)

    # 클래스별 레이어 분리 OR 하나의 레이어에 모두 넣기
    if show_per_class_layers:
        classes = sorted(df["cls_name"].astype(str).unique())
        for cls in classes:
            sub = df[df["cls_name"].astype(str) == cls]
            layer = folium.FeatureGroup(name=f"{cls} ({len(sub)} clusters)", show=True)
            mc = MarkerCluster(name=f"{cls} markers", control=False)
            _add_clusters_to_layer(sub, layer, mc, radius_m)
            layer.add_child(mc)
            m.add_child(layer)
        folium.LayerControl(collapsed=False).add_to(m)
    else:
        layer = folium.FeatureGroup(name="All clusters", show=True)
        mc = MarkerCluster(name="markers", control=False)
        _add_clusters_to_layer(df, layer, mc, radius_m)
        layer.add_child(mc)
        m.add_child(layer)

    m.save(str(output_html))
    print(f"[OK] Saved map → {output_html}")

def _add_clusters_to_layer(df: pd.DataFrame, layer: folium.FeatureGroup, mc: MarkerCluster, radius_m: float):
    """DataFrame의 각 클러스터를 원+마커로 레이어에 추가"""
    for _, r in df.iterrows():
        cls = str(r["cls_name"])
        cid = int(r["cluster_id"]) if pd.notnull(r["cluster_id"]) else -1
        n = int(r["n_frames"]) if pd.notnull(r["n_frames"]) else 0
        lat = float(r["rep_lat"])
        lon = float(r["rep_lon"])

        popup_html = (
            f"<b>Class</b>: {cls}<br>"
            f"<b>Cluster ID</b>: {cid}<br>"
            f"<b>Frames</b>: {n}<br>"
            f"<b>Lat</b>: {lat:.6f}<br>"
            f"<b>Lon</b>: {lon:.6f}"
        )

        # 반경 원(클러스터 반경 시각화)
        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            weight=1,
            color="blue",
            fill=True,
            fill_opacity=0.10,
        ).add_to(layer)

        # 마커
        folium.Marker(
            [lat, lon],
            tooltip=f"{cls}#{cid} (n={n})",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(mc)

if __name__ == "__main__":
    # 예시 사용법
    # - 같은 폴더에 yoloold_clean.csv가 있다고 가정
    make_cluster_map(
        csv_path="yolonew_clean.csv",
        output_html="yolonew_clusters_map.html",
        radius_m=40.0,         # clean_csv에서 사용한 반경과 일치시키는 것을 권장
        zoom_start=15,
        show_per_class_layers=True,
    )

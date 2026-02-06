# backend/app/route/pathfinding.py

from __future__ import annotations

import math
from typing import List, Dict, Tuple
from collections import defaultdict  # ✅ 추가

import osmnx as ox
import networkx as nx
from sqlalchemy.orm import Session

from app.route.models import Obstacle


# --- 내부 유틸: 거리 계산 (meter) ---

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    위도/경도 두 점 사이의 거리(m)를 계산.
    """
    R = 6371000  # 지구 반지름(m)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --- 그래프 로딩 ---

def load_graph_for_route(start: Tuple[float, float],
                         end: Tuple[float, float],
                         network_type: str = "walk"):
    """
    start ~ end 영역을 포함하는 OSM 그래프 로딩.
    Overpass 엔드포인트는 외부 설정값 사용.
    """
    lat1, lon1 = start
    lat2, lon2 = end

    # osmnx 기본 설정
    ox.settings.use_cache = False #잠시 바꿔둠.
    ox.settings.log_console = False
    ox.settings.overpass_rate_limit = True
    ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"

    # start/end 를 모두 포함하는 bounding box (약 1km 마진)
    margin_deg = 0.01  # 위도/경도 ~1.1km 정도
    north = max(lat1, lat2) + margin_deg
    south = min(lat1, lat2) - margin_deg
    east = max(lon1, lon2) + margin_deg
    west = min(lon1, lon2) - margin_deg

    G = ox.graph_from_bbox(
        north=north,
        south=south,
        east=east,
        west=west,
        network_type=network_type,
    )

    # 가장 큰 연결 성분만 사용 (고립된 조각 제거)
    G = ox.utils_graph.get_largest_component(G, strongly=False)
    return G, (south, north, west, east)


# --- 메인: A* + 장애물 패널티 + 회피 통계 계산 ---

def astar_path_with_penalty(
    start: Tuple[float, float],
    end: Tuple[float, float],
    db: Session,
    avoid_types: List[str],
    radius_m: float,
    penalties: Dict[str, float],
):
    """
    - OSM 그래프 기반 A* 경로 탐색
    - YOLO 장애물(Obstacle) 테이블을 반영해 edge weight에 패널티 적용
    - 개별 장애물 단위로 회피 성공/실패 개수를 집계
    - risk_factors: 선택한 타입 중 하나라도 실패한 타입 목록 (타입 단위)
    - obstacle_stats: 타입별 total / success / failed 개수
    - unavoidable: 실제 경로 반경 내에 포함된 장애물 목록
    """

    # 0. 그래프 로딩
    G, (south, north, west, east) = load_graph_for_route(start, end, network_type="walk")

    # 1. 시작/끝 노드 매핑
    start_lat, start_lng = start
    end_lat, end_lng = end

    start_node = ox.nearest_nodes(G, X=start_lng, Y=start_lat)
    end_node = ox.nearest_nodes(G, X=end_lng, Y=end_lat)

    # 2. 회피 대상 장애물 조회 (체크박스에서 선택한 타입만)
    obstacles: List[Obstacle] = []
    if avoid_types:
        q = (
            db.query(Obstacle)
            .filter(Obstacle.type.in_(avoid_types))
            .filter(Obstacle.lat >= south, Obstacle.lat <= north)
            .filter(Obstacle.lng >= west, Obstacle.lng <= east)
        )
        obstacles = q.all()

    # (lat, lng, type) 형태로 단순화
    obs_list: List[Tuple[float, float, str]] = [
        (o.lat, o.lng, o.type) for o in obstacles
    ]

    # 3. edge weight 계산: 거리 + 장애물 패널티
    for u, v, key, data in G.edges(keys=True, data=True):
        # 기본 길이 계산
        length = data.get("length")
        if length is None:
            y1 = G.nodes[u]["y"]
            x1 = G.nodes[u]["x"]
            y2 = G.nodes[v]["y"]
            x2 = G.nodes[v]["x"]
            length = haversine_m(y1, x1, y2, x2)
            data["length"] = length

        penalty_total = 0.0

        # === 1) 장애물 패널티 (기존 코드) ===
        if avoid_types and obs_list:
            y_mid = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            x_mid = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2

            for obs_lat, obs_lng, obs_type in obs_list:
                d = haversine_m(y_mid, x_mid, obs_lat, obs_lng)
                if d <= radius_m:
                    penalty_total += penalties.get(obs_type, 0.0)

        # === 2) 차량 도로 패널티 추가 (핵심) ===
        hw = data.get("highway", "")

        # 여러 타입일 수 있으니 리스트 처리
        if isinstance(hw, list):
            hw_main = hw[0]
        else:
            hw_main = hw

        # 차도 판단 기준: 보행 중심이 아닌 도로들
        car_roads = [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link"
        ]

        # 차량 기반 도로는 보행 가능하더라도 패널티 강하게 부여
        if hw_main in car_roads:
            penalty_total += 10000  # 👈 핵심 패널티 (원하면 더 올려도 됨)

        # 최종 가중치
        data["weight"] = length + penalty_total


    # 4. A* 경로 탐색
    try:
        path_nodes = nx.astar_path(G, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        # 경로 자체가 없으면 직선 + 모든 선택 타입을 실패로 간주 (임시 fallback)
        fallback_distance = haversine_m(start_lat, start_lng, end_lat, end_lng)
        return {
            "route": [start, end],
            "distance_m": fallback_distance,
            "risk_factors": avoid_types,  # 전부 실패
            "obstacle_stats": {},         # 통계 없음
            "unavoidable": [],            # 알 수 있는 장애물 없음
        }

    # 5. 경로 길이 계산 (m)
    total_distance = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edges = G.get_edge_data(u, v)
        if not edges:
            continue

        min_len = None
        for _, edata in edges.items():
            l = edata.get("length")
            if l is None:
                y1 = G.nodes[u]["y"]
                x1 = G.nodes[u]["x"]
                y2 = G.nodes[v]["y"]
                x2 = G.nodes[v]["x"]
                l = haversine_m(y1, x1, y2, x2)
            if (min_len is None) or (l < min_len):
                min_len = l
        if min_len:
            total_distance += min_len

    # 6. 경로 좌표 리스트 (lat, lng) 형태로 변환
    route_coords: List[Tuple[float, float]] = [
        (G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes
    ]

    # 7. 개별 장애물 단위로 회피 성공/실패 집계
    type_total = defaultdict(int)
    type_failed = defaultdict(int)
    type_success = defaultdict(int)
    unavoidable_list: List[Dict[str, float | str]] = []

    if avoid_types and obs_list:
        # 경로 좌표 리스트 생성 (노드 기반)
        route_coords_for_check: List[Tuple[float, float]] = [
            (G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes
        ]

        for obs_lat, obs_lng, obs_type in obs_list:
            type_total[obs_type] += 1
            hit = False

            # 경로 위 좌표들 중 반경 내에 들어오는지 확인
            # 노드뿐만 아니라 노드 사이의 경로(edge)도 고려
            for i in range(len(route_coords_for_check)):
                route_lat, route_lng = route_coords_for_check[i]
                d = haversine_m(route_lat, route_lng, obs_lat, obs_lng)
                if d <= radius_m:
                    hit = True
                    break

                # 다음 노드와의 세그먼트도 확인 (노드 사이 경로를 놓치지 않기 위해)
                if i < len(route_coords_for_check) - 1:
                    next_lat, next_lng = route_coords_for_check[i + 1]
                    # 세그먼트의 최단 거리 근사: 두 노드 사이의 중간점들도 체크
                    for j in range(1, 4):  # 3개의 중간점 체크
                        mid_lat = route_lat + (next_lat - route_lat) * (j / 4.0)
                        mid_lng = route_lng + (next_lng - route_lng) * (j / 4.0)
                        d_mid = haversine_m(mid_lat, mid_lng, obs_lat, obs_lng)
                        if d_mid <= radius_m:
                            hit = True
                            break
                    if hit:
                        break

            # hit된 경우에만 unavoidable_list에 추가 (중복 방지)
            if hit:
                type_failed[obs_type] += 1
                unavoidable_list.append(
                    {
                        "type": obs_type,
                        "lat": obs_lat,
                        "lng": obs_lng,
                    }
                )
            else:
                type_success[obs_type] += 1

    # 8. 타입별 통계 및 risk_factors 생성
    obstacle_stats: Dict[str, Dict[str, int]] = {}
    risk_factors: List[str] = []

    for t in avoid_types:
        total = type_total[t]
        failed = type_failed[t]
        success = type_success[t]
        obstacle_stats[t] = {
            "total": total,
            "success": success,
            "failed": failed,
        }
        if failed > 0:
            risk_factors.append(t)

    return {
        "route": route_coords,
        "distance_m": total_distance,
        "risk_factors": risk_factors,   # 장애물 타입 단위 실패 여부
        "obstacle_stats": obstacle_stats,  # 타입별 total/success/failed
        "unavoidable": unavoidable_list,   # 실제 경로 반경 내 장애물 목록
    }

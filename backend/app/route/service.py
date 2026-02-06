import math
from typing import List, Dict, Tuple
from collections import defaultdict

from sqlalchemy.orm import Session

from app.route.pathfinding import astar_path_with_penalty, haversine_m
from app.route.models import RouteResult, Obstacle


# ---------------------------------------------------------
# 1) 경로 계산 (DB 저장 없음)
# ---------------------------------------------------------
def find_path_from_request(req, db: Session, user_id: int):
    return find_best_path(req, db, user_id)


# ---------------------------------------------------------
# 2) 사용자가 선택한 경로 저장
# ---------------------------------------------------------
def save_route(req, db: Session, user_id: int) -> RouteResult:
    try:
        # avoided 리스트를 문자열로 변환 (빈 리스트 처리)
        avoided_str = ",".join(req.avoided) if req.avoided else ""

        route_obj = RouteResult(
            user_id=user_id,
            start_lat=req.start_lat,
            start_lng=req.start_lng,
            end_lat=req.end_lat,
            end_lng=req.end_lng,
            route_points=req.route_points,  # JSON 컬럼이므로 자동 직렬화됨
            distance_m=req.distance_m,
            avoided=avoided_str,
        )

        db.add(route_obj)
        db.commit()
        db.refresh(route_obj)
        return route_obj
    except Exception as e:
        db.rollback()
        raise e


# ---------------------------------------------------------
# 3) 저장된 경로 삭제
# ---------------------------------------------------------
def delete_route(route_id: int, db: Session, user_id: int):
    route = (
        db.query(RouteResult)
        .filter(RouteResult.id == route_id, RouteResult.user_id == user_id)
        .first()
    )

    if not route:
        return None

    db.delete(route)
    db.commit()
    return True


# ---------------------------------------------------------
# 4) 저장된 경로 목록 조회
# ---------------------------------------------------------
def get_my_routes(db: Session, user_id: int):
    return (
        db.query(RouteResult)
        .filter(RouteResult.user_id == user_id)
        .order_by(RouteResult.created_at.desc())
        .all()
    )


# ---------------------------------------------------------
# 5) 장애물 전체 조회
# ---------------------------------------------------------
def get_all_obstacles(db: Session):
    return db.query(Obstacle).all()


# ---------------------------------------------------------
# 5-1) 장애물 존재 여부 확인
# ---------------------------------------------------------
def has_obstacles(db: Session) -> bool:
    """DB에 장애물이 하나라도 있는지 확인"""
    count = db.query(Obstacle).count()
    return count > 0


# ---------------------------------------------------------
# 6) 최적 회피 경로 계산 (핵심 기능)
# ---------------------------------------------------------
def calculate_stats_for_route(
    route_coords: List[Tuple[float, float]],
    original_avoid_types: List[str],
    db: Session,
    radius_m: float,
    start: Tuple[float, float] = None,
    end: Tuple[float, float] = None,
) -> Dict[str, Dict[str, int]]:
    """
    최종 경로에 대해 원래 선택한 모든 타입의 장애물 통계를 계산.
    경로를 재계산하지 않고 기존 경로 좌표에 대한 통계만 계산.
    """
    if not original_avoid_types or not route_coords:
        return {}

    # bounding box 계산: 시작/끝점이 있으면 그것을 기준으로, 없으면 경로 좌표 기준
    # pathfinding.py의 load_graph_for_route와 동일한 방식 사용
    if start and end:
        lat1, lon1 = start
        lat2, lon2 = end
        margin_deg = 0.01  # 위도/경도 ~1.1km 정도 (pathfinding.py와 동일)
        north = max(lat1, lat2) + margin_deg
        south = min(lat1, lat2) - margin_deg
        east = max(lon1, lon2) + margin_deg
        west = min(lon1, lon2) - margin_deg
    else:
        # fallback: 경로 좌표 기준 (반경을 고려해 더 넓게)
        lats = [coord[0] for coord in route_coords]
        lngs = [coord[1] for coord in route_coords]
        # 반경을 고려해 더 넓은 margin 사용 (약 0.001도 = 약 111m)
        margin_deg = 0.001 + (radius_m / 111000.0)  # 반경을 도 단위로 변환
        north = max(lats) + margin_deg
        south = min(lats) - margin_deg
        east = max(lngs) + margin_deg
        west = min(lngs) - margin_deg

    # 원래 선택한 모든 타입의 장애물 조회
    obstacles = (
        db.query(Obstacle)
        .filter(Obstacle.type.in_(original_avoid_types))
        .filter(Obstacle.lat >= south, Obstacle.lat <= north)
        .filter(Obstacle.lng >= west, Obstacle.lng <= east)
        .all()
    )

    print(f"📊 [통계 재계산] 조회된 장애물: {len(obstacles)}개")
    print(f"📊 [통계 재계산] 경로 좌표 수: {len(route_coords)}개")
    print(f"📊 [통계 재계산] 반경: {radius_m}m")

    # 통계 계산
    type_total = defaultdict(int)
    type_failed = defaultdict(int)
    type_success = defaultdict(int)
    failed_obstacles = []  # 디버깅용

    for obs in obstacles:
        obs_type = obs.type
        type_total[obs_type] += 1
        hit = False
        min_dist = float('inf')

        # 경로 위 좌표들 중 반경 내에 들어오는지 확인
        # 노드뿐만 아니라 노드 사이의 경로(edge)도 고려
        for i in range(len(route_coords)):
            route_lat, route_lng = route_coords[i]
            d = haversine_m(route_lat, route_lng, obs.lat, obs.lng)
            min_dist = min(min_dist, d)
            if d <= radius_m:
                hit = True
                break

            # 다음 노드와의 세그먼트도 확인 (노드 사이 경로를 놓치지 않기 위해)
            if i < len(route_coords) - 1:
                next_lat, next_lng = route_coords[i + 1]
                # 세그먼트의 최단 거리 근사: 두 노드 사이의 중간점들도 체크
                # 더 정확하게 하려면 여러 중간점을 체크
                for j in range(1, 4):  # 3개의 중간점 체크
                    mid_lat = route_lat + (next_lat - route_lat) * (j / 4.0)
                    mid_lng = route_lng + (next_lng - route_lng) * (j / 4.0)
                    d_mid = haversine_m(mid_lat, mid_lng, obs.lat, obs.lng)
                    min_dist = min(min_dist, d_mid)
                    if d_mid <= radius_m:
                        hit = True
                        break
                if hit:
                    break

        if hit:
            type_failed[obs_type] += 1
            failed_obstacles.append({
                "id": obs.id,  # 중복 확인을 위해 ID 추가
                "type": obs_type,
                "lat": obs.lat,
                "lng": obs.lng,
                "distance": min_dist
            })
        else:
            type_success[obs_type] += 1

    # 타입별 통계 생성 (원래 선택한 모든 타입 포함)
    obstacle_stats: Dict[str, Dict[str, int]] = {}
    for t in original_avoid_types:
        obstacle_stats[t] = {
            "total": type_total[t],
            "success": type_success[t],
            "failed": type_failed[t],
        }

    if failed_obstacles:
        print(f"⚠️  [통계 재계산] 회피 실패한 장애물 {len(failed_obstacles)}개:")
        for obs in failed_obstacles[:5]:  # 처음 5개만 출력
            print(f"   - {obs['type']} (ID: {obs['id']}): ({obs['lat']:.6f}, {obs['lng']:.6f}), 최소거리: {obs['distance']:.2f}m")
        if len(failed_obstacles) > 5:
            print(f"   ... 외 {len(failed_obstacles) - 5}개")
    else:
        print(f"✅ [통계 재계산] 모든 장애물 회피 성공!")

    return obstacle_stats


def find_best_path(req, db, user_id):

    current_avoid = list(req.avoid_types)
    original_avoid_types = list(req.avoid_types)  # 원래 선택한 타입 저장

    print(f"\n🚀 [경로 찾기 시작] 시작: ({req.start_lat}, {req.start_lng}), 끝: ({req.end_lat}, {req.end_lng})")
    print(f"🚀 [경로 찾기] 회피 대상 타입: {original_avoid_types}")
    print(f"🚀 [경로 찾기] 반경: {req.radius_m}m, 패널티: {req.penalties}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- 반복 {iteration} ---")
        # 1) 경로 계산
        res = astar_path_with_penalty(
            start=(req.start_lat, req.start_lng),
            end=(req.end_lat, req.end_lng),
            db=db,
            avoid_types=current_avoid,
            radius_m=req.radius_m,
            penalties=req.penalties,
        )

        failed = res["risk_factors"]

        # 2) 모든 회피 성공 → 최종 경로에 대해 원래 선택한 모든 타입의 통계 계산
        if not failed:
            # 최종 경로에 대해 원래 선택한 모든 타입의 통계를 다시 계산
            final_stats = calculate_stats_for_route(
                route_coords=res["route"],
                original_avoid_types=original_avoid_types,
                db=db,
                radius_m=req.radius_m,
                start=(req.start_lat, req.start_lng),
                end=(req.end_lat, req.end_lng),
            )

            # risk_factors는 원래 타입 기준으로 재계산
            final_risk_factors = [
                t for t in original_avoid_types
                if final_stats.get(t, {}).get("failed", 0) > 0
            ]

            return {
                "route": res["route"],
                "distance_m": res["distance_m"],
                "risk_factors": final_risk_factors,
                "avoided_final": current_avoid,
                "obstacle_stats": final_stats  # 원래 선택한 모든 타입의 통계
            }

        # 3) 실패한 회피 제거
        for f in failed:
            if f in current_avoid:
                current_avoid.remove(f)

        # 4) 더 이상 회피할 것이 없으면 → 최단거리 경로
        if not current_avoid:
            final = astar_path_with_penalty(
                start=(req.start_lat, req.start_lng),
                end=(req.end_lat, req.end_lng),
                db=db,
                avoid_types=[],
                radius_m=req.radius_m,
                penalties=req.penalties
            )

            # 최종 경로에 대해 원래 선택한 모든 타입의 통계 계산
            final_stats = calculate_stats_for_route(
                route_coords=final["route"],
                original_avoid_types=original_avoid_types,
                db=db,
                radius_m=req.radius_m,
                start=(req.start_lat, req.start_lng),
                end=(req.end_lat, req.end_lng),
            )

            # risk_factors는 원래 타입 기준으로 재계산
            final_risk_factors = [
                t for t in original_avoid_types
                if final_stats.get(t, {}).get("failed", 0) > 0
            ]

            return {
                "route": final["route"],
                "distance_m": final["distance_m"],
                "risk_factors": final_risk_factors,
                "avoided_final": [],
                "obstacle_stats": final_stats  # 원래 선택한 모든 타입의 통계
            }

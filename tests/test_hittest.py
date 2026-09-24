"""点位命中测试（hit-test）的几何单元测试。"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.hittest import MAX_POINTS, compute_hit_test  # noqa: E402
from app.svggeo import analyze_contours  # noqa: E402


def _hit(d, points, fill_rule="nonzero"):
    contours, _ = analyze_contours(d, fill_rule)
    return compute_hit_test(contours, list(points), fill_rule)


def _results(res):
    return [r["result"] for r in res["results"]]


# ---------------------------------------------------------------------------
# 基本判定与边界优先
# ---------------------------------------------------------------------------

def test_simple_square_inside_outside_boundary():
    res = _hit("M0,0h10v10h-10Z",
               [(5, 5), (20, 5), (0, 5), (0, 0), (10, 10)])
    assert _results(res) == [
        "inside", "outside", "boundary", "boundary", "boundary"]
    assert res["point_count"] == 5
    assert res["checked_contour_count"] == 1
    assert res["inside_count"] == 1
    assert res["outside_count"] == 1
    assert res["boundary_count"] == 3
    assert res["skipped_contours"] == []


def test_boundary_on_hole_edge_has_priority_over_inside_outer():
    # 点 (10,20) 在孔的边上、同时在外环内部：边界优先
    res = _hit("M0,0h40v40h-40Z M10,10h20v20h-20Z",
               [(10, 20), (5, 5), (20, 20)])
    assert _results(res) == ["boundary", "inside", "inside"]


def test_results_preserve_input_order_and_echo_points():
    pts = [(20, 20), (0, 0), (5, 5)]
    res = _hit("M0,0h10v10h-10Z", pts)
    assert [r["index"] for r in res["results"]] == [0, 1, 2]
    assert [(r["x"], r["y"]) for r in res["results"]] == pts
    assert _results(res) == ["outside", "boundary", "inside"]


# ---------------------------------------------------------------------------
# 填充规则：交叉次数奇偶 vs 有向边绕数
# ---------------------------------------------------------------------------

def test_same_direction_nested_squares_differ_by_fill_rule():
    # 两个同向嵌套正方形：内点 nonzero 绕数 2 -> inside，evenodd 交叉 2 次 -> outside
    d = "M0,0h40v40h-40Z M10,10h20v20h-20Z"
    assert _results(_hit(d, [(20, 20), (5, 5)], "nonzero")) == [
        "inside", "inside"]
    assert _results(_hit(d, [(20, 20), (5, 5)], "evenodd")) == [
        "outside", "inside"]


def test_nonzero_hole_written_ccw_cancels_winding():
    # 外环顺时针、孔逆时针：孔内绕数 0 -> outside；环间 -> inside
    d = "M0,0h40v40h-40Z M10,30h20v-20h-20Z"
    assert _results(_hit(d, [(20, 20), (5, 5)], "nonzero")) == [
        "outside", "inside"]
    # 同一几何 evenodd 与方向无关，结果一致
    assert _results(_hit(d, [(20, 20), (5, 5)], "evenodd")) == [
        "outside", "inside"]


def test_nonzero_winding_sign_does_not_matter():
    # 正方形顺时针/逆时针书写，nonzero 下内部点均为 inside
    assert _results(_hit("M0,0h10v10h-10Z", [(5, 5)])) == ["inside"]
    assert _results(_hit("M0,0 L0,10 L10,10 L10,0 Z", [(5, 5)])) == ["inside"]


def test_island_three_level_nesting_both_rules():
    # 外-孔-岛三层（孔反向书写）：岛内 inside、孔内 outside、外环带 inside
    d = "M0,0h10v10h-10Z M1,9h8v-8h-8Z M6,6h2v2h-2Z"
    pts = [(0.5, 0.5), (5, 5), (7, 7)]
    assert _results(_hit(d, pts, "nonzero")) == [
        "inside", "outside", "inside"]
    assert _results(_hit(d, pts, "evenodd")) == [
        "inside", "outside", "inside"]


# ---------------------------------------------------------------------------
# 参与规则：闭合子路径全部参与，开放子路径不参与
# ---------------------------------------------------------------------------

def test_zero_area_self_intersecting_closed_path_participates():
    # 蝴蝶结：净面积为 0 的闭合自交路径仍参与（左瓣内 inside，上瓣外 outside）
    d = "M0,0 L10,10 L10,0 L0,10 Z"
    res = _hit(d, [(2, 5), (5, 2)], "nonzero")
    assert res["checked_contour_count"] == 1
    assert _results(res) == ["inside", "outside"]
    assert _results(_hit(d, [(2, 5), (5, 2)], "evenodd")) == [
        "inside", "outside"]
    # 两瓣交点 (5,5) 落在边上 -> boundary
    assert _results(_hit(d, [(5, 5)])) == ["boundary"]


def test_closed_degenerate_line_still_gives_boundary():
    # 零面积闭合“线段环”：点落在线上 -> boundary，线外 -> outside
    res = _hit("M0,0 L10,0 Z", [(5, 0), (5, 1)])
    assert res["checked_contour_count"] == 1
    assert _results(res) == ["boundary", "outside"]


def test_open_subpath_does_not_participate():
    # 开放折线不参与：点 (5,2) 虽在开放边上，不判 boundary（只算外环 -> inside）
    d = "M0,0h10v10h-10Z M2,2 L8,2 L8,8"
    res = _hit(d, [(5, 2), (20, 20)])
    assert _results(res) == ["inside", "outside"]
    assert res["checked_contour_count"] == 1
    assert res["skipped_contours"] == [
        {"original_index": 1, "role": "open", "reason": "open"}]


def test_only_open_subpaths_everything_outside():
    res = _hit("M0,0 L10,0 L10,10", [(5, 5), (5, 0)])
    assert _results(res) == ["outside", "outside"]
    assert res["checked_contour_count"] == 0
    assert res["outside_count"] == 2


def test_empty_path_everything_outside():
    res = _hit("", [(0, 0), (1, 1)])
    assert _results(res) == ["outside", "outside"]
    assert res["checked_contour_count"] == 0
    assert res["skipped_contours"] == []


# ---------------------------------------------------------------------------
# 曲线细分与其他
# ---------------------------------------------------------------------------

def test_curve_flattened_edges():
    # 单位圆（两段弧）：圆心 inside，弧端点 (1,0) 在细分折线上 -> boundary
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    assert _results(_hit(d, [(0, 0), (1, 0), (2, 0)])) == [
        "inside", "boundary", "outside"]


def test_many_points_up_to_limit():
    res = _hit("M0,0h10v10h-10Z",
               [(0.5 + i % 10, 0.5 + (i // 10) % 10)
                for i in range(MAX_POINTS)])
    assert res["point_count"] == MAX_POINTS
    assert res["inside_count"] == MAX_POINTS
    assert len(res["results"]) == MAX_POINTS


def test_deterministic_across_calls():
    d = "M0,0h40v40h-40Z M10,10h20v20h-20Z M2,2 L8,2"
    pts = [(20, 20), (5, 5), (10, 10), (41, 0)]
    assert _hit(d, pts) == _hit(d, pts)

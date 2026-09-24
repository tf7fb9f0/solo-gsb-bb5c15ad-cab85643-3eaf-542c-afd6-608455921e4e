"""点位命中测试（hit-test，只读即时计算）。

对一份已存 SVG path 即时解析（复用 :func:`app.svggeo.analyze_contours`
的解析与曲线细分折线，见 :func:`app.svggeo.flatten_subpath`），对请求中的
每个测试点给出 ``inside`` / ``outside`` / ``boundary`` 判定：

- 参与判定的是原始 ``d`` 中的**全部闭合子路径**（贝塞尔/弧已细分为
  折线，逐边判断）；**开放子路径不参与**（列入 ``skipped_contours``）；
  闭合但退化的子路径（如零净面积的闭合自交路径）仍按字面参与；
- **边界优先**：点落在任一参与子路径的任一条细分边上（含端点，
  容差按环尺寸自适应，与 :func:`app.svggeo.point_in_ring` 一致）
  直接判 ``boundary``；
- 其余点按保存的 ``fill_rule`` 汇总全部闭合子路径的射线穿越：

  - ``evenodd``：+x 射线与各边的交叉次数（半开边规则
    ``(y1 > y) != (y2 > y)``，水平边不计、顶点不重复计数），
    奇数为 ``inside``；
  - ``nonzero``：有向边绕数（上行边左穿 +1、下行边右穿 −1，
    交点严格位于点右侧才计数），非零为 ``inside``。

整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import math
from typing import Any

from .svggeo import Contour

# 单次请求允许的最多测试点数，超限由路由层返回 422
MAX_POINTS = 500
# 边界判定相对容差（与 svggeo.point_in_ring 一致，按环尺寸自适应）
_EPS_REL = 1e-9


def compute_hit_test(contours: list[Contour],
                     points: list[tuple[float, float]],
                     fill_rule: str) -> dict[str, Any]:
    """对已解析的轮廓列表逐点判定 inside / outside / boundary。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓；
            仅 ``closed`` 子路径参与判定（含闭合退化轮廓）
        points: 测试点 ``(x, y)`` 列表（路由层已校验为有限数值）
        fill_rule: ``nonzero`` 或 ``evenodd``

    返回:
        ``{point_count, checked_contour_count, inside_count,
        outside_count, boundary_count, results, skipped_contours}``；
        ``results`` 按输入顺序每项给出 ``{index, x, y, result}``。
    """
    closed = sorted((c for c in contours if c.closed),
                    key=lambda c: c.original_index)
    rings = [(c.points, _ring_eps(c)) for c in closed]

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if not c.closed
    ]

    results: list[dict[str, Any]] = []
    counts = {"inside": 0, "outside": 0, "boundary": 0}
    for idx, (x, y) in enumerate(points):
        r = _classify_point(x, y, rings, fill_rule)
        counts[r] += 1
        results.append({"index": idx, "x": x, "y": y, "result": r})

    return {
        "point_count": len(points),
        "checked_contour_count": len(closed),
        "inside_count": counts["inside"],
        "outside_count": counts["outside"],
        "boundary_count": counts["boundary"],
        "results": results,
        "skipped_contours": skipped,
    }


def _ring_eps(c: Contour) -> float:
    """边界判定容差：按环包围盒尺寸自适应（与 point_in_ring 一致）。"""
    minx, miny, maxx, maxy = c.bounds
    return _EPS_REL * max(1.0, maxx - minx, maxy - miny)


def _classify_point(x: float, y: float,
                    rings: list[tuple[list[tuple[float, float]], float]],
                    fill_rule: str) -> str:
    """单点判定：边界优先，其后按填充规则汇总全部闭合环的射线穿越。"""
    crossings = 0  # evenodd：+x 射线交叉次数
    winding = 0    # nonzero：有向边绕数
    for pts, eps in rings:
        n = len(pts)
        for k in range(n):
            ax, ay = pts[k]
            bx, by = pts[(k + 1) % n]
            if _on_segment(ax, ay, bx, by, x, y, eps):
                return "boundary"
            if (ay > y) == (by > y):
                continue  # 半开边规则：水平边与不横跨 y 的边不计
            if fill_rule == "evenodd":
                xint = ax + (bx - ax) * (y - ay) / (by - ay)
                if xint > x:
                    crossings += 1
            else:
                # 有向边绕数：交点严格在点右侧时，上行 +1、下行 −1
                is_left = (bx - ax) * (y - ay) - (x - ax) * (by - ay)
                if ay <= y:  # 上行边（by > ay）
                    if is_left > 0:
                        winding += 1
                elif is_left < 0:  # 下行边
                    winding -= 1
    if fill_rule == "evenodd":
        return "inside" if crossings % 2 else "outside"
    return "inside" if winding != 0 else "outside"


def _on_segment(ax: float, ay: float, bx: float, by: float,
                px: float, py: float, eps: float) -> bool:
    """点 (px, py) 是否在线段 (ax, ay)-(bx, by) 上（含端点），容差按边长缩放。"""
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, math.hypot(bx - ax, by - ay)):
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps)

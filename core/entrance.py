# -*- coding: utf-8 -*-
"""
入口引索匹配（新匹配模式）
========================
用户先选入口(正门/侧门/二楼)；游戏内捕获后，按检测到的入口白箭头图标裁出入口区域，
与各种子的入口引索裁图(entrance_index/{seed}_{type}.png)严格对比：
分类(迷雾剔除) + 多尺度 SQDIFF。排名种子。

原理：正门房间形状固定(不判别)，侧门/二楼房间形状各异(主要判别依据)。
故侧门/二楼入口匹配能定种子；正门通常分不出(各种子相似)。

§3.1：find_seed_by_entrance 除返回 (results, icon_pos, icon_score) 外，每条 result
额外带 (匹配尺度 s, minMaxLoc 参考图匹配位置 mloc)，供两段式对齐第一段构造 M1。
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import cv2
import numpy as np

from core.vision import (FIXED_PANEL, FOG_BGR, _find_icon, classify_region,
                         load_bgr, walls_as_floors)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)

# 入口引索目录（config.paths.entrance_index，相对项目根解析）。
ENTRANCE_INDEX_DIR = (ROOT / _CFG["paths"]["entrance_index"]).resolve()
ENTRANCE_FLOOR = {"正门": "一楼", "侧门": "一楼", "二楼": "二楼"}
# 入口裁图尺度接近(都~200px)，窄范围多尺度对齐。下限 0.35：游戏默认（不碰缩放）入口
# 状态 s≈0.40（2026-09-14 实测 6 局图标 k=1.0，s=0.4/k），旧下限 0.6 会把默认态全部
# 撞底钉住 → 匹分失真/同分退化。上界不变（观测域）。
SCALES_ENT = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
# 最小模板世界覆盖闸：模板边长 < 此值直接跳过该尺度。旧下限 0.6 隐式兼任防小模板假
# 获胜（小模板塞进均匀区 SQDIFF 偏低——17.13 实测种子16 在 s=0.55 以 112px 模板拿
# 0.039 反超真种子 0.046）；扩域到 0.35 后必须显式补上。实测真匹配模板世界覆盖
# 163-204 参考px（k∈[0.40,1.0] 经 ruler/基线裁样），150 = 下沿留 ~8% 余量。大裁样
# （k 放大后 ~464px）在 0.35 档模板 162px 仍可搜，204px 基线裁样 0.735 以下全挡。
MIN_TEMPLATE_PX = 150
# 图标当尺子·棘轮基准：k 高于此值才放大裁样（k≤此值保持基线 half=102px）。8 月全部
# 基线数据 k∈[0.40,0.45]（细网格 NCC 峰）→ 棘轮保证其逐像素零扰动；默认档 k≈1.0 →
# half≈232，世界覆盖回到 8 月水平（~184 参考px）。s 与裁框大小无关，扩大取样梯子救
# 不了缩放——缩放只有这条 ruler 路径；只放大不缩小，缩小会让小模板假获胜（17.13 实测
# 纯 ruler 缩到 186px → 错种子 0.031 反超）。
ICON_K_REF = float(_CFG["match"].get("icon_k_ref", 0.44))
# 渐变迷雾剔除 + 房间加权（config [match]；迷雾是渐变色，仅 tol24 精确色剔不净外圈，
# 膨胀雾核一并剔；房间亮度两侧都远离雾色、误判率最低→加权，通路易被雾污染→降权）
_FOG_DILATE = int(_CFG["match"].get("fog_dilate", 5))
_ROOM_W = float(_CFG["match"].get("room_weight", 2.0))
_PASS_W = float(_CFG["match"].get("passage_weight", 0.5))
# 贴墙增益（模板侧软加权）：cls5 墙线 ±9px 带内像素权重 ×(1+wall_boost)，0=关(基线零扰动)。
# 依据(2026-09-15 实测)：真走廊中位离墙 10-22px、雾 50-206px，但硬剔除「离墙>12px」会
# 误杀 36-68% 真走廊(前沿雾侧墙不可见/宽走廊中心远/tol漏检暗墙段)——只能软加权不能剔。
# 墙色雾免疫(雾最亮~75 < 墙带下沿~95)，贴墙像素=最不易被雾环污染的可靠结构。
_WALL_BOOST = float(_CFG["match"].get("wall_boost", 0.0))
_WALL_BAND = 9


def _crop_around_icon(shot, panel, icon_pos, half_frac=0.18, icon_k=None):
    """以入口图标为中心裁方区域，裁样大小按图标检测尺度 k 放大（「图标当尺子·棘轮」）。

    游戏内放大（默认档 k≈1.0 / 玩家缩放）会让固定 half_frac 的裁样只覆盖一小块世界 →
    房形信息缺失、单类退化；且 s 与裁框大小无关，扩大取样救不了缩放。故 k>K_REF 时
    half ∝ k/ICON_K_REF 放大裁样，把世界覆盖拉回基线水平。
    **只放大不缩小（棘轮）**：k≤K_REF 保持基线 half_frac 行为——缩小裁样会丢信息让
    小模板假获胜（2026-09-14 实测 17.13 k=0.40 纯 ruler 缩到 186px → 错种子 0.031 反超），
    且 8 月全部基线数据 k≤0.45，棘轮保证其逐像素零扰动。
    half 另设上限 min(pw,ph)//2：k 量偏大/贴边档放大时防止裁样超过 228/SCALES_ENT[0]
    → 全尺度放不进引索 → 匹配恒空（同日实测 710px 裁样全'-'）。
    icon 贴面板边缘时自动增大到 0.25 档（同乘 k 放大）。方框 clamp 入面板不截断。
    icon_k=None（手框样本等无图标尺度的场景）→ 保持旧行为不缩放。"""
    px, py, pw, ph = panel
    cx, cy = icon_pos
    cxp, cyp = cx - px, cy - py
    kf = max(1.0, (icon_k / ICON_K_REF)) if icon_k else 1.0
    # 上限 316=228/0.36/2：放大档(k≈1.0→真s≈0.36-0.40)下侧长 632×0.36=228 恰可搜，
    # 再大会把真尺度挤出可行域（710px 裁样实测全'-'）。同时保护梯子：0.25/0.32 档
    # 在默认缩放下不至于全顶到同一个帽（284 帽实测压扁梯子 → 144715 丢失梯子救援）。
    half_cap = 316
    half = min(int(min(pw, ph) * half_frac * kf), half_cap)
    edge = min(int(min(pw, ph) * 0.25 * kf), half_cap)
    if cxp < edge or cxp > pw - edge or cyp < edge or cyp > ph - edge:
        half = edge  # 贴边 → 增大样本（17.13 中央不触发，基线不变）
    region = shot[py:py + ph, px:px + pw]
    side = 2 * half
    x0 = min(max(cxp - half, 0), max(0, pw - side))
    x1 = x0 + side
    y0 = min(max(cyp - half, 0), max(0, ph - side))
    y1 = y0 + side
    return region[y0:y1, x0:x1].copy()


def build_sample_mask(crop):
    """样本分类 mask：迷雾(含渐变外圈)剔除 + 房间/通路加权。

    迷雾是渐变色：tol24 精确色只剔得掉雾核，渐变外圈会被 classify_region 判成
    通路(cls3) 污染 mask——故把雾核膨胀 _FOG_DILATE px 一并剔除（渐变段与雾核
    空间相邻）。房间(cls2)亮度两侧都远离雾色、误判率最低→权重 _ROOM_W；
    通路(cls3)最易被渐变雾污染→降权 _PASS_W。参数见 config [match]。
    返回 (cls, mask_u8, weights_f32)。"""
    cls_raw = classify_region(crop)
    wall = cls_raw == 5
    cls = walls_as_floors(cls_raw, crop)
    fog = (np.abs(crop.astype(np.int16) - FOG_BGR).sum(axis=2) < 24)
    if _FOG_DILATE > 0:
        fog = cv2.dilate(fog.astype(np.uint8),
                         np.ones((_FOG_DILATE, _FOG_DILATE), np.uint8)) > 0
    mask = (((cls == 1) | (cls == 2) | (cls == 3)) & (~fog)).astype(np.uint8)
    w = mask.astype(np.float32) * _PASS_W
    w[cls == 2] = _ROOM_W
    w[cls == 1] = 1.0
    if _WALL_BOOST > 0 and wall.any():
        near = cv2.dilate(wall.astype(np.uint8),
                          np.ones((2 * _WALL_BAND + 1,) * 2, np.uint8)) > 0
        w[near & (mask > 0)] *= (1.0 + _WALL_BOOST)
    return cls, mask, w


def find_seed_by_entrance(shot, lib, entrance_type: str,
                          index_dir=ENTRANCE_INDEX_DIR, panel=FIXED_PANEL,
                          top_n=6, sample_crop=None):
    """入口引索匹配。返回 ([(score, seed, 方向-门, 楼层, s, mloc), ...], icon_pos, icon_score, icon_k)。

    score 越小越匹配。entrance_type ∈ {正门, 侧门, 二楼}。
    s 为获胜尺度(SCALES_ENT)，mloc=(mlx,mly) 为该尺度下 matchTemplate 在参考裁图里的
    argmin 位置；二者供 build_entrance_transform 构造两段式对齐第一段 M1。缺时为 None。
    icon_k 为图标检测获胜尺度（「图标当尺子」，裁样按它缩放），缺时 None。
    sample_crop: 手动框选的入口样本(HxWx3 BGR)，阶段3 手框纠错用；给了则用它做
    in_cls/in_mask(跳过 _crop_around_icon)，None 则自动以图标为中心裁。默认 None=基线。
    """
    icon_pos, icon_score, icon_k = _find_icon(shot, *panel)
    if icon_pos is None and sample_crop is None:
        return [], None, 0.0, None
    in_crop = sample_crop if sample_crop is not None else _crop_around_icon(
        shot, panel, icon_pos, icon_k=icon_k)
    in_cls, in_mask, in_w = build_sample_mask(in_crop)
    if in_mask.sum() < 100:
        return [], icon_pos, icon_score, icon_k

    fl = ENTRANCE_FLOOR[entrance_type]
    results = []
    for seed in lib.seeds():
        idx_path = index_dir / f"{seed}_{entrance_type}.png"
        if not idx_path.exists():
            continue
        idx_bgr = load_bgr(str(idx_path))
        idx_cls = walls_as_floors(classify_region(idx_bgr), idx_bgr)
        idx_f = idx_cls.astype(np.float32)
        best_sc, best_s, best_mloc = 1e9, None, None
        for s in SCALES_ENT:
            tw, th = int(in_cls.shape[1] * s), int(in_cls.shape[0] * s)
            if tw < 10 or th < 10 or th > idx_cls.shape[0] or tw > idx_cls.shape[1]:
                continue
            if min(tw, th) < MIN_TEMPLATE_PX:  # 防小模板假获胜（见常量注释）
                continue
            tcl = cv2.resize(in_cls, (tw, th), interpolation=cv2.INTER_NEAREST)
            tmk = cv2.resize(in_w, (tw, th), interpolation=cv2.INTER_NEAREST)
            if tmk.sum() < 50:
                continue
            res = cv2.matchTemplate(idx_f, tcl.astype(np.float32), cv2.TM_SQDIFF,
                                   mask=tmk.astype(np.float32))
            mn, _, _, ml = cv2.minMaxLoc(res)   # §3.1：取 argmin 位置，旧版只读 res.min()
            sc = float(mn) / float(tmk.sum())  # = float(res.min())/sum，与旧版同值→基线不变
            if sc < best_sc:
                best_sc, best_s, best_mloc = sc, s, (int(ml[0]), int(ml[1]))
        info = lib.get(seed, "一楼")
        if best_s is None:
            # 无任何尺度放得下（样本大于该种子引索裁图，如手框过大）：不计入。
            # 旧版计入 1e9 哨兵分，多种子全 1e9 时排序无意义 → 假种子居中显示（2026-09-13 实测）。
            continue
        results.append((best_sc, seed, info.key if info else str(seed), fl,
                        best_s, best_mloc))
    results.sort(key=lambda x: x[0])
    return results[:top_n], icon_pos, icon_score, icon_k


def load_index(seed: int, index_dir=ENTRANCE_INDEX_DIR) -> dict | None:
    """加载某种子的 {seed}_index.json（含各入口的 cx/cy/box）。不存在返回 None。"""
    p = index_dir / f"{seed}_index.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_entrance_transform(best, entrance_type: str, index_json: dict,
                             icon_pos, panel=FIXED_PANEL):
    """两段式对齐第一段：由入口引索匹配结果构造相似变换 M1(ref→屏幕，北朝上无旋转)。

    best: find_seed_by_entrance 的单条结果 (score, seed, key, fl, s, mloc)。
    对应关系：屏幕入口图标 icon_pos ↔ 参考图入口图标 (cx,cy)（来自 index_json）；
    尺度：入口匹配尺度 s（各裁图均原生分辨率，s 与 find_overlay_transform 的 s 同义，
    见 CLAUDE.md 两段式备忘）。M1 = [[1/s,0, ix-cx/s],[0,1/s, iy-cy/s]]。
    返回 (M1(2x3), hint_s=s) 或 None（缺尺度/缺入口坐标）。
    """
    _sc, _seed, _key, _fl, s, _mloc = best
    if s is None:
        return None
    entry = index_json.get(entrance_type)
    if entry is None:
        return None
    cx, cy = entry["cx"], entry["cy"]
    a = 1.0 / s                 # M[0,0] = inv_s = 1/s（ref→屏幕）
    ix, iy = float(icon_pos[0]), float(icon_pos[1])
    tx = ix - a * cx
    ty = iy - a * cy
    M1 = np.array([[a, 0.0, tx], [0.0, a, ty]], dtype=np.float64)
    return M1, s


if __name__ == "__main__":
    import sys
    from core.map_library import MapLibrary
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib = MapLibrary.load(r"D:\Pictures\20260818摸金地图")
    # 已知正确种子(用户确认)，测各入口类型能否定到
    GT = [("26", "17.13.06.98", 18), ("26", "17.11.35.17", 18),
          ("26", "01.11.13.38", 27), ("26", "16.42.33.89", 9)]
    for dt, ts, seed in GT:
        p = rf"D:\Videos\NVIDIA\IdentityV\IdentityV Screenshot 2026.08.{dt} - {ts}.png"
        shot = load_bgr(p)
        print(f"\n=== {ts} 真种子{seed} ===")
        for et in ("正门", "侧门", "二楼"):
            res, ip, isc, _k = find_seed_by_entrance(shot, lib, et, top_n=3)
            if not res:
                print(f"  [{et}] 图标未检出/无匹配 (图标分{isc:.2f})")
                continue
            s = " | ".join(f"{c[2]}[{c[3]}]={c[0]:.3f}" + (" <==真" if c[1] == seed else "")
                           for c in res)
            hit = "✓" if res[0][1] == seed else "✗"
            print(f"  [{et}] 图标{isc:.2f}@{ip} {hit} {s}")

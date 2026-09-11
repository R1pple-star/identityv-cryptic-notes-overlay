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

from core.vision import FIXED_PANEL, FOG_BGR, _find_icon, classify_region, load_bgr

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)

# 入口引索目录（config.paths.entrance_index，相对项目根解析）。
ENTRANCE_INDEX_DIR = (ROOT / _CFG["paths"]["entrance_index"]).resolve()
ENTRANCE_FLOOR = {"正门": "一楼", "侧门": "一楼", "二楼": "二楼"}
# 入口裁图尺度接近(都~200px)，窄范围多尺度对齐
SCALES_ENT = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)


def _crop_around_icon(shot, panel, icon_pos, half_frac=0.18):
    """以入口图标为中心，裁面板 min 维的 half_frac*2 宽的方区域。

    图标贴面板边缘时，方框整体 clamp 入面板（保持完整 2*half 宽，不截断成窄条），
    避免样本退化致匹配 0.000。icon 在面板中央时与旧 max/min 裁法结果一致（基线不变）。
    """
    px, py, pw, ph = panel
    cx, cy = icon_pos
    cxp, cyp = cx - px, cy - py
    half = int(min(pw, ph) * half_frac)
    region = shot[py:py + ph, px:px + pw]
    side = 2 * half
    x0 = min(max(cxp - half, 0), max(0, pw - side))
    x1 = x0 + side
    y0 = min(max(cyp - half, 0), max(0, ph - side))
    y1 = y0 + side
    return region[y0:y1, x0:x1].copy()


def find_seed_by_entrance(shot, lib, entrance_type: str,
                          index_dir=ENTRANCE_INDEX_DIR, panel=FIXED_PANEL,
                          top_n=6, sample_crop=None):
    """入口引索匹配。返回 ([(score, seed, 方向-门, 楼层, s, mloc), ...], icon_pos, icon_score)。

    score 越小越匹配。entrance_type ∈ {正门, 侧门, 二楼}。
    s 为获胜尺度(SCALES_ENT)，mloc=(mlx,mly) 为该尺度下 matchTemplate 在参考裁图里的
    argmin 位置；二者供 build_entrance_transform 构造两段式对齐第一段 M1。缺时为 None。
    sample_crop: 手动框选的入口样本(HxWx3 BGR)，阶段3 手框纠错用；给了则用它做
    in_cls/in_mask(跳过 _crop_around_icon)，None 则自动以图标为中心裁。默认 None=基线。
    """
    icon_pos, icon_score = _find_icon(shot, *panel)
    if icon_pos is None and sample_crop is None:
        return [], None, 0.0
    in_crop = sample_crop if sample_crop is not None else _crop_around_icon(shot, panel, icon_pos)
    in_cls = classify_region(in_crop)
    in_fog = (np.abs(in_crop.astype(np.int16) - FOG_BGR).sum(axis=2) < 24)
    in_mask = (((in_cls == 1) | (in_cls == 2) | (in_cls == 3)) & (~in_fog)).astype(np.uint8)
    if in_mask.sum() < 100:
        return [], icon_pos, icon_score

    fl = ENTRANCE_FLOOR[entrance_type]
    results = []
    for seed in lib.seeds():
        idx_path = index_dir / f"{seed}_{entrance_type}.png"
        if not idx_path.exists():
            continue
        idx_bgr = load_bgr(str(idx_path))
        idx_cls = classify_region(idx_bgr)
        idx_f = idx_cls.astype(np.float32)
        best_sc, best_s, best_mloc = 1e9, None, None
        for s in SCALES_ENT:
            tw, th = int(in_cls.shape[1] * s), int(in_cls.shape[0] * s)
            if tw < 10 or th < 10 or th > idx_cls.shape[0] or tw > idx_cls.shape[1]:
                continue
            tcl = cv2.resize(in_cls, (tw, th), interpolation=cv2.INTER_NEAREST)
            tmk = cv2.resize(in_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            if tmk.sum() < 50:
                continue
            res = cv2.matchTemplate(idx_f, tcl.astype(np.float32), cv2.TM_SQDIFF,
                                   mask=tmk.astype(np.float32))
            mn, _, _, ml = cv2.minMaxLoc(res)   # §3.1：取 argmin 位置，旧版只读 res.min()
            sc = float(mn) / float(tmk.sum())  # = float(res.min())/sum，与旧版同值→基线不变
            if sc < best_sc:
                best_sc, best_s, best_mloc = sc, s, (int(ml[0]), int(ml[1]))
        info = lib.get(seed, "一楼")
        results.append((best_sc, seed, info.key if info else str(seed), fl,
                        best_s, best_mloc))
    results.sort(key=lambda x: x[0])
    return results[:top_n], icon_pos, icon_score


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
            res, ip, isc = find_seed_by_entrance(shot, lib, et, top_n=3)
            if not res:
                print(f"  [{et}] 图标未检出/无匹配 (图标分{isc:.2f})")
                continue
            s = " | ".join(f"{c[2]}[{c[3]}]={c[0]:.3f}" + (" <==真" if c[1] == seed else "")
                           for c in res)
            hit = "✓" if res[0][1] == seed else "✗"
            print(f"  [{et}] 图标{isc:.2f}@{ip} {hit} {s}")

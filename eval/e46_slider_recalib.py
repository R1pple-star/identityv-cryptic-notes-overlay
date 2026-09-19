# -*- coding: utf-8 -*-
"""缩放滑条重标定（用户 2026-09-18 报「缩大小不准」）。

现状表 `ZOOM_TABLE_Y/S` 只覆盖模板 y∈[382,708]（来自用户 09-17 那段视频），
而 09-18 的演示证明圆点能走到 **y=752**（真尺度 0.83，图标锚点误差 5px + 图标尺子 0.39/0.45=0.87
双重确认），且轨道一直延伸到 y≈790。⇒ 放大/缩小两端都够不着，读数被截断在 0.60。

本脚本用**独立真值**重建标定：对标注集里每一张「刚进入口」的全屏帧，
  真尺度 = 让引索入口图标 ref 坐标经 M 映到屏幕实测图标位置、误差最小的那个 s
（`find_overlay_transform(fast=True, hint_s=s)` 是单尺度全平移，~100ms/格，比全搜快 15 倍；
 图标锚点法成立的前提是「图标在入口处」，故只用标注集里刚进入口的帧）。

用法: python eval/e46_slider_recalib.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402

from core.alignment import find_overlay_transform  # noqa: E402
from core.entrance import load_index  # noqa: E402
from core.map_library import MapLibrary  # noqa: E402
from core.vision import (NAV_BAND, _find_icon, load_bgr, nav_column_ncc,  # noqa: E402
                         panel_for_screen, roi_of)

NVIDIA = Path(r"D:\Videos\NVIDIA\IdentityV")
PANEL = panel_for_screen(1920, 1080)
PX, PY, PW, PH = PANEL
NX0, NY0, NX1, _NY1 = NAV_BAND
X0, Y0 = PX + PW + NX0, PY + NY0
LIB = MapLibrary.load(r"D:\Pictures\20260818摸金地图")
SCAN = tuple(np.round(np.arange(0.15, 1.06, 0.05), 2))
DOT_LO, DOT_HI = 315, 805                 # 加宽后的圆点窗（模板坐标）


def dot_y(shot):
    band = shot[Y0:Y0 + 820, X0:X0 + NX1]
    if band.shape[0] < 820:
        return None, 0.0
    seg = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, 81:88].mean(axis=1)
    base = float(np.median(seg[DOT_LO:DOT_HI]))
    i = int(np.argmax(seg[DOT_LO:DOT_HI])) + DOT_LO
    return i, float(seg[i]) - base


def truth_s(shot, ref, cx, cy, ip, k):
    """图标锚点真值：先 0.05 粗扫，再在最优点 ±0.045 上 0.01 精扫。返回 (s, err)。"""
    def err_at(s):
        r = find_overlay_transform(shot, ref, PANEL, hint_s=float(s), fast=True)
        if r is None:
            return None
        a = float(r[0][0, 0])
        return float(np.hypot(r[0][0, 2] + cx * a - ip[0], r[0][1, 2] + cy * a - ip[1]))
    best = None
    for s in SCAN:
        e = err_at(s)
        if e is not None and (best is None or e < best[0]):
            best = (e, float(s))
    if best is None:
        return None, None
    for s in np.arange(best[1] - 0.045, best[1] + 0.046, 0.01):
        if 0.12 <= s <= 1.08:
            e = err_at(round(float(s), 3))
            if e is not None and e < best[0]:
                best = (e, round(float(s), 3))
    return best[1], best[0]


def shot_paths(f):
    for d in (ROOT / "captures", NVIDIA):
        if (d / f).exists():
            return d / f
    return None


rows = []
for row in csv.DictReader(open(ROOT / "eval" / "labels.csv", encoding="utf-8-sig")):
    if row.get("is_entrance_view", "").strip() != "1":
        continue
    p = shot_paths(row["file"])
    if p is None:
        continue
    shot = load_bgr(str(p))
    if float(shot[-3:].mean()) >= 60:      # 窗口化：导航列不在标定位置，跳过
        continue
    roi = roi_of(shot, PANEL)
    ncc = nav_column_ncc(roi, PANEL)
    if ncc is None or ncc < 0.75:
        continue
    y, q = dot_y(shot)
    ip, isc, k = _find_icon(shot, *PANEL)
    if ip is None or isc < 0.5:
        continue
    seed, et = int(row["seed"]), row["entrance_type"]
    idx = load_index(seed)
    info = LIB.get(seed, row.get("floor", "一楼") or "一楼")
    if idx is None or et not in idx or info is None:
        continue
    s, e = truth_s(shot, load_bgr(str(info.path)), idx[et]["cx"], idx[et]["cy"], ip, k)
    if s is None:
        continue
    rows.append(dict(f=row["file"], y=y, q=q, s=s, err=e, k=k, seed=seed, et=et, ncc=ncc))
    print(f"   {row['file'][-26:]:<26} 圆点y={y:>4} 峰-基{q:>4.0f}  真尺度{s:.3f} "
          f"(图标误差{e:>4.0f}px)  k={k:.2f} 0.39/k={0.39 / k:.3f}  种子{seed}{et}")

rows.sort(key=lambda r: r["y"])
print(f"\n共 {len(rows)} 点（只保留图标误差 <25px 的为可信真值）")
ok = [r for r in rows if r["err"] < 25]
print(f"可信 {len(ok)} 点：y∈[{min(r['y'] for r in ok)},{max(r['y'] for r in ok)}]  "
      f"s∈[{min(r['s'] for r in ok):.3f},{max(r['s'] for r in ok):.3f}]")
print(f"\n{'圆点y':>6}{'真尺度':>9}{'图标误差':>9}{'k':>7}{'0.39/k':>8}   文件")
for r in ok:
    print(f"{r['y']:>6}{r['s']:>9.3f}{r['err']:>9.0f}{r['k']:>7.2f}{0.39 / r['k']:>8.3f}"
          f"   {r['f'][-30:]}")
if len(ok) >= 3:
    ys = [r["y"] for r in ok]
    mono = all(ok[i]["s"] <= ok[i + 1]["s"] for i in range(len(ok) - 1))
    print(f"\n单调（y 越大 s 越大）: {'✓' if mono else '✗ 有反例'}")
    res = []
    for i in range(len(ok)):
        rest = [r for j, r in enumerate(ok) if j != i]
        pred = float(np.interp(ok[i]["y"], [r["y"] for r in rest], [r["s"] for r in rest]))
        res.append(abs(pred - ok[i]["s"]))
    if res:
        print(f"留一交叉验证 |Δs|：中位 {np.median(res):.3f}  最大 {max(res):.3f}")
    # 输出候选表（0.05 步长采样插值）
    ys2 = np.arange(min(ys), max(ys) + 1, 5)
    ss2 = np.interp(ys2, ys, [r["s"] for r in ok])
    print("\n候选 ZOOM_TABLE（每 5 px 一点）:")
    print("ZOOM_TABLE_Y = (" + ", ".join(f"{int(v)}" for v in ys2) + ")")
    print("ZOOM_TABLE_S = (" + ", ".join(f"{v:.3f}" for v in ss2) + ")")

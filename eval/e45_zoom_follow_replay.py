# -*- coding: utf-8 -*-
"""只改缩放时跟随为什么不跟/不准 —— 用 09-18 那 4 帧演示（种子12，侧门）离线复刻。

真值来源：图标锚点法（引索 json 的图标 ref 坐标 ↔ `_find_icon` 的屏幕实测位置）。
4 帧里屏幕图标几乎不动（(1097,486)→(1089,526)，全程只挪 40px）而尺度变了 3.7 倍
⇒ **游戏缩放是绕图标（玩家）为中心的**。

A 每帧真值（圆点 y / 滑条读数 / 真尺度 / 图标尺子 0.374÷k）
B 复刻 app._track_tick 的**新**逻辑：滑条(1ms) → 图标尺子(200ms) → 两个都没有就地判丢
B2 只留图标尺子（模拟导航列被挡但图标可见）—— 备用源必须也能跟住
C 对照：两个尺度源都掐掉（模拟窗口化/图标也看不到）⇒ 只许小窗有界微调，大动就判丢

用法: python eval/e45_zoom_follow_replay.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.alignment import find_overlay_transform  # noqa: E402
from core.entrance import load_index  # noqa: E402
from core.map_library import MapLibrary  # noqa: E402
from core.vision import (_find_icon, load_bgr, map_roi, panel_for_screen,  # noqa: E402
                         roi_of, zoom_scale_from_k, zoom_scale_from_roi)
from ui.track import (TRACK_DEADBAND_S, TRACK_NEAR_R, TRACK_OK, Tracker,  # noqa: E402
                      ruler_slop, verdict)

NVIDIA = Path(r"D:\Videos\NVIDIA\IdentityV")
PANEL = panel_for_screen(1920, 1080)
PX, PY, PW, PH = PANEL
RX, RY = map_roi(PANEL)[:2]
LIB = MapLibrary.load(r"D:\Pictures\20260818摸金地图")
SEED, FLOOR, ENT = 12, "一楼", "侧门"
DEMO = ["11.07.13.59", "11.07.15.58", "11.07.21.41", "11.07.23.86"]
SCAN = tuple(np.round(np.arange(0.15, 1.11, 0.05), 2))
IDX = load_index(SEED)
REF = load_bgr(str(LIB.get(SEED, FLOOR).path))
CX, CY = IDX[ENT]["cx"], IDX[ENT]["cy"]
PATH_REF = str(LIB.get(SEED, FLOOR).path)


def ierr(M, ip):
    a = float(M[0, 0])
    return float(np.hypot(M[0, 2] + CX * a - ip[0], M[1, 2] + CY * a - ip[1]))


# ---- A) 逐帧真值 ----
print("=" * 88)
print(f"A) 真值（种子{SEED} {ENT}）")
print(f"   {'帧':<12}{'滑条读数':>9}{'真尺度':>8}{'读数误差':>9}{'图标k':>7}{'尺子0.374/k':>12}"
      f"{'尺子误差':>9}{'图标屏幕位置':>15}")
frames = []
for t in DEMO:
    shot = load_bgr(str(NVIDIA / f"IdentityV Screenshot 2026.09.18 - {t}.png"))
    roi = roi_of(shot, PANEL)
    s_ui, why = zoom_scale_from_roi(roi, PANEL)
    ip, isc, ik = _find_icon(shot, *PANEL)
    best = None
    for s in SCAN:
        r = find_overlay_transform(shot, REF, PANEL, scales=(float(s),))
        if r is not None and (best is None or ierr(r[0], ip) < best[0]):
            best = (ierr(r[0], ip), float(s))
    for s in np.arange(best[1] - 0.048, best[1] + 0.049, 0.01):
        if 0.15 <= s <= 1.10:
            r = find_overlay_transform(shot, REF, PANEL, scales=(float(round(s, 3)),))
            if r is not None and ierr(r[0], ip) < best[0]:
                best = (ierr(r[0], ip), float(round(s, 3)))
    hot = find_overlay_transform(shot, REF, PANEL, hint_icon=(CX, CY, ip[0], ip[1]))
    s_k = zoom_scale_from_k(ik)
    frames.append(dict(t=t, shot=shot, roi=roi, ip=ip, s_ui=s_ui, why=why, ik=ik,
                       s_tr=best[1], err_tr=best[0], M_hot=hot[0], s_hot=1 / hot[0][0, 0]))
    d = frames[-1]
    print(f"   {t:<12}{(f'{s_ui:.3f}' if s_ui else '读不到'):>9}{d['s_tr']:>8.2f}"
          f"{(s_ui - d['s_tr'] if s_ui else float('nan')):>9.3f}{ik:>7.2f}{s_k:>12.3f}"
          f"{(s_k - d['s_tr']):>9.3f}{str(ip):>15}")
    print(f"        热键式对齐（hint_icon 钉平移 + 全档扫尺度）：s={d['s_hot']:.3f} "
          f"分{hot[1]:.3f}  真尺度 {d['s_tr']:.2f}（图标误差 {d['err_tr']:.0f}px）"
          + (f"   ⚠ 滑条读数失败：{why}" if s_ui is None else ""))


# ---- B/C) 复刻 app._track_tick（新逻辑） ----
def replay(label, scale_mode="auto"):
    """复刻 app._track_tick，返回 {'errs': [采纳后图标误差], 'lost': 累计判丢} 供断言。"""
    print()
    print("=" * 88)
    print(label)
    errs = []
    f0 = frames[0]
    tr = Tracker()
    tr.seed_from(PANEL, SEED, FLOOR, PATH_REF, f0["M_hot"], f0["s_hot"])
    print(f"   起始（=热键式对齐）s={tr.s:.3f}  图标误差 {ierr(tr.M, f0['ip']):.0f}px")
    for f in frames[1:]:
        shot, ip = f["shot"], f["ip"]
        roi = f["roi"]
        region = shot[PY:PY + PH, PX:PX + PW]
        s_ui, why_s = (zoom_scale_from_roi(roi, PANEL) if scale_mode != "none"
                       else (None, "滑条源已掐掉（对照）"))
        if scale_mode == "ruler":
            s_ui, why_s = None, "滑条源已掐掉（对照）"
        s_src = "滑条"
        if s_ui is None and scale_mode != "none":
            _ip, isc_i, k_i = _find_icon(roi, PX - RX, PY - RY, PW, PH)
            s_k = zoom_scale_from_k(k_i) if (isc_i or 0) >= 0.65 else None
            if s_k is None:
                why_s = f"{why_s}；图标尺子也不可用(分{isc_i:.2f})"
            elif abs(s_k - tr.s) > ruler_slop(s_k, k_i):
                s_ui, s_src = s_k, "图标尺子"
            else:
                s_ui, s_src = tr.s, "沿用上次"
        scale_moved = s_ui is not None and abs(s_ui - tr.s) > TRACK_DEADBAND_S
        print(f"\n   ── {f['t']}  尺度源={s_src if s_ui else '无'} "
              f"s={s_ui if s_ui else '-'}  tr.s={tr.s:.3f}  scale_moved={scale_moved}")
        sc_near = None
        if not scale_moved:
            near = find_overlay_transform(None, REF, PANEL, region=region, hint_s=tr.s,
                                          hint_icon=tr.pin(), fast=True,
                                          hint_radius=TRACK_NEAR_R, ref_key=PATH_REF)
            sc_near = near[1] if near is not None else None
            print(f"      1) 小窗 ±{TRACK_NEAR_R}px @s={tr.s:.3f}: "
                  + (f"分{sc_near:.3f} ov{near[2]:.2f} 图标误差{ierr(near[0], ip):.0f}px"
                     if near is not None else "无解"))
            if near is not None and sc_near < TRACK_OK and near[2] >= 0.40:
                tr.adopt(near[0], tr.s)
                errs.append((ierr(tr.M, ip), "near"))
                print(f"      ⇒ 采纳小窗，图标误差 {errs[-1][0]:.0f}px"
                      f"（真值 {f['err_tr']:.0f}px）")
                continue
        if s_ui is None:
            tr.lost += 1
            print(f"      1.5) 没有可信尺度 ⇒ 不做全平移，就地判丢 ×{tr.lost}"
                  + ("（连 2 次 app 会隐藏投影）" if tr.lost >= 2 else ""))
            continue
        cands = [s_ui]
        if abs(s_ui - tr.s) > 0.02:
            cands.append(tr.s)
        hit = False
        for s in cands:
            r = find_overlay_transform(None, REF, PANEL, region=region, hint_s=s, fast=True,
                                       ref_key=PATH_REF)
            if r is None:
                print(f"      2) 全平移 @s={s:.3f}: 无解"); continue
            v = verdict(sc_near, r[1], r[2])
            print(f"      2) 全平移 @s={s:.3f}: 分{r[1]:.3f} ov{r[2]:.2f} "
                  f"图标误差{ierr(r[0], ip):.0f}px → {v}")
            if v == "accept":
                tr.adopt(r[0], float(s)); hit = True
                errs.append((ierr(tr.M, ip), "full"))
                print(f"      ⇒ 采纳 s={s:.3f}（真 {f['s_tr']:.2f}）"
                      f" 图标误差 {errs[-1][0]:.0f}px（真对齐 {f['err_tr']:.0f}px）")
                break
            if v == "keep":
                hit = True; break
        if not hit:
            tr.lost += 1
            print(f"      ⇒ lost ×{tr.lost}"
                  + ("（连 2 次 app 会隐藏投影）" if tr.lost >= 2 else ""))
    return {"errs": errs, "lost": tr.lost}


rB = replay("B) 新逻辑：滑条 → 图标尺子 → 两个都没有就地判丢")
rB2 = replay("B2) 滑条掐掉、只留图标尺子（模拟导航列被挡但图标可见）", scale_mode="ruler")
rC = replay("C) 对照：两个尺度源都掐掉（模拟窗口化/地图上图标也看不到）", scale_mode="none")

# ---- 断言 ----
print()
print("=" * 88)
checks = [
    ("B 滑条路径：3 步全采到、末步走全平移、图标误差 ≤5px（旧代码末步 184px）",
     len(rB["errs"]) == 3 and rB["errs"][-1][1] == "full"
     and max(e for e, _p in rB["errs"]) <= 5),
    ("B2 图标尺子路径：3 步全采到且 ≤5px",
     len(rB2["errs"]) == 3 and max(e for e, _p in rB2["errs"]) <= 5),
    ("C 无尺度源：不做任何全平移采纳（小窗的有界微调可以有），且判丢 ≥2（会隐藏）",
     all(p == "near" for _e, p in rC["errs"]) and rC["lost"] >= 2),
]
for name, ok in checks:
    print(f"   {'✓' if ok else '✗'} {name}")
print(f"\n跟随·缩放链离线断言: {sum(1 for _n, ok in checks if ok)}/{len(checks)} 通过"
      + ("" if all(ok for _n, ok in checks) else "   ← 有失败"))
sys.exit(0 if all(ok for _n, ok in checks) else 1)

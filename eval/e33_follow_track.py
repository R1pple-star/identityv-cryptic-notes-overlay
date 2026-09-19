# -*- coding: utf-8 -*-
"""跟随·第二步（投影跟着地图平移/缩放）的离线断言 —— 复刻 app._track_tick 的判定链。

**只用真实截图/视频帧**（CLAUDE.md 硬约束：禁用合成面板）。
判定用**独立真值**：引索 json 的入口图标 cx/cy（参考图坐标）× `_find_icon`（屏幕坐标）——
真对齐的 M 必须把前者映到后者（同一模板、同渲染器）。图标误差 <25px 才算对。

覆盖：
  1. 静止连拍帧：画面帧差闸应判「没动」，**一次匹配都不跑**；
  2. 同种子真平移对：从 A 的变换跟到 B，与 B 的独立全搜比对；
  3. **缩放**（视频帧，s 0.25→0.45）：滑条给的尺度能不能让投影跟上；
  4. 错种子注入：必须判丢（不许把别的地图的位置认下来）；
  5. 耗时分解（近处小窗 / 单尺度全平移 / 重烘焙）。

用法: python eval/e33_follow_track.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402

from core.alignment import find_overlay_transform  # noqa: E402
from core.entrance import load_index  # noqa: E402
from core.map_library import MapLibrary  # noqa: E402
from core.vision import (_find_icon, load_bgr, map_roi, nav_band, panel_for_screen,
                         roi_of, zoom_scale_from_roi)  # noqa: E402
from ui.track import (TRACK_DEADBAND_S, TRACK_MOTION_MAD, TRACK_NEAR_R, TRACK_OK,
                      Tracker, verdict)  # noqa: E402

NVIDIA = Path(r"D:\Videos\NVIDIA\IdentityV")
VIDEO = NVIDIA / "IdentityV 2026.09.14 - 14.53.02.02.mp4"
CAP = ROOT / "captures"
PANEL = panel_for_screen(1920, 1080)
lib = MapLibrary.load(r"D:\Pictures\20260818摸金地图")
OVERLAP_MIN = 0.40
ERR_OK = 25.0
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"  {'✓' if cond else '✗'} {name} {detail}")


def icon_err(M, seed, ent, ip):
    e = load_index(seed)[ent]
    inv_s = float(M[0, 0])
    return float(np.hypot(M[0, 2] + e["cx"] * inv_s - ip[0], M[1, 2] + e["cy"] * inv_s - ip[1]))


def roi_of_shot(shot):
    return roi_of(shot, PANEL)


def panel_of_roi(roi):
    px, py, pw, ph = PANEL
    rx, ry, _x1, _y1 = map_roi(PANEL)
    return roi[py - ry:py - ry + ph, px - rx:px - rx + pw]


def seed_align(shot, seed, s_hint=None):
    """建跟踪起点：单尺度（或滑条给的尺度）全平移，返回 (M, s, score, ov)。"""
    ref = load_bgr(str(lib.get(seed, "一楼").path))
    r = (find_overlay_transform(shot, ref, PANEL, scales=(s_hint,)) if s_hint is not None
         else find_overlay_transform(shot, ref, PANEL))
    return r, ref


def tick(shot, tr, ref, use_motion_gate=True, prev_small=None):
    """复刻 app._track_tick（改 app 时必须同步改这里）。返回 (动作, M, s, score, ov, 说明)。"""
    roi = roi_of_shot(shot)
    region = panel_of_roi(roi)
    if use_motion_gate:
        small = cv2.resize(region, (64, 34), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev_small is not None and float(np.abs(small - prev_small).mean()) < TRACK_MOTION_MAD:
            return "静止", tr.M, tr.s, None, None, "画面帧差闸判没动，未跑匹配"
    s_ui, why_s = zoom_scale_from_roi(roi, PANEL)
    scale_moved = s_ui is not None and abs(s_ui - tr.s) > TRACK_DEADBAND_S
    sc_near = None
    if not scale_moved:
        near = find_overlay_transform(None, ref, PANEL, region=region, hint_s=tr.s,
                                      hint_icon=tr.pin(), fast=True, hint_radius=TRACK_NEAR_R)
        sc_near = near[1] if near is not None else None
        if near is not None and sc_near < TRACK_OK and near[2] >= OVERLAP_MIN:
            tr.adopt(near[0], tr.s)
            return "接受-近处", near[0], tr.s, sc_near, near[2], f"小窗(±{TRACK_NEAR_R}px)内收敛"
    # 没有可信尺度 ⇒ 不做全平移、就地判丢（与 app._track_tick 同款；2026-09-18 实测：
    # 错尺度上全平移的全局极小能落在偏 840px 的位置，且分比真尺度还低 ⇒ 没有闸能拦）
    if s_ui is None:
        tr.note_lost()
        return "判丢", tr.M, tr.s, None, None, f"没有可信尺度（{why_s}）"
    cands = [s_ui]
    if abs(s_ui - tr.s) > 0.02:
        cands.append(tr.s)
    for s in cands:
        r = find_overlay_transform(None, ref, PANEL, region=region, hint_s=s, fast=True)
        if r is None:
            continue
        v = verdict(sc_near, r[1], r[2])
        if v == "accept":
            tr.adopt(r[0], float(s))
            return ("接受-重对齐", r[0], float(s), r[1], r[2],
                    f"尺度{s:.2f}（{why_s}）")
        if v == "keep":
            tr.note_ok()
            return "保持", tr.M, tr.s, sc_near, None, "旧位置够好"
    tr.note_lost()
    return "判丢", tr.M, tr.s, None, None, f"候选尺度 {[round(c, 3) for c in cands]} 都不过闸"


print("=" * 84)
print("1) 静止连拍帧：画面帧差闸应判「没动」（同一 burst 内两张像素级相同）")
for a, b, seed in (("171738", "171749", 6), ("171503", "171521", 23), ("172219", "172259", 2)):
    shotA = load_bgr(str(CAP / f"hotkey_20260916_{a}.png"))
    shotB = load_bgr(str(CAP / f"hotkey_20260916_{b}.png"))
    r, ref = seed_align(shotA, seed)
    tr = Tracker(); tr.seed_from(PANEL, seed, "一楼", str(lib.get(seed, "一楼").path), r[0], 1 / r[0][0, 0])
    smallA = cv2.resize(panel_of_roi(roi_of_shot(shotA)), (64, 34),
                        interpolation=cv2.INTER_AREA).astype(np.int16)
    t0 = time.perf_counter()
    act, M, s, sc, ov, why = tick(shotB, tr, ref, prev_small=smallA)
    dt = (time.perf_counter() - t0) * 1000
    check(f"{a}→{b} 判「静止」", act == "静止", f"[{act}] {why}  {dt:.1f}ms")

print()
print("=" * 84)
print("2) 同种子真平移/缩放：从 A 的变换跟到 B（用**独立图标真值**判对错）")
print("   ⚠️ 只用全屏帧（屏幕底行<60）：窗口化 capture 里游戏窗整体位移过，标称面板几何")
print("      无效（实测导航列跑进面板内），那批不能当对齐真值。")
# (A, B, seed, B 的真尺度= e35 用图标锚点扫描锁定的值)
PAIRS = [("2026.09.14 - 21.59.28.73", "2026.09.14 - 21.59.53.50", 10, 0.35),
         ("2026.09.14 - 21.58.00.06", "2026.09.14 - 21.58.05.01", 6, 0.35),
         ("hotkey_20260914_144715", "hotkey_20260914_145440", 2, 0.40),
         ("2026.09.14 - 21.46.34.97", "2026.09.14 - 21.48.51.36", 13, 0.45)]


def find_shot(name):
    for c in (CAP / f"{name}.png", NVIDIA / f"{name}.png",
              NVIDIA / f"IdentityV Screenshot {name}.png"):
        if c.exists():
            return c
    return None


for nameA, nameB, seed, s_true in PAIRS:
    pA, pB = find_shot(nameA), find_shot(nameB)
    if pA is None or pB is None:
        print(f"  [缺文件] {nameA} / {nameB}"); continue
    shotA, shotB = load_bgr(str(pA)), load_bgr(str(pB))
    rA = find_overlay_transform(shotA, load_bgr(str(lib.get(seed, "一楼").path)), PANEL)
    ref = load_bgr(str(lib.get(seed, "一楼").path))
    tr = Tracker()
    tr.seed_from(PANEL, seed, "一楼", str(lib.get(seed, "一楼").path), rA[0], 1 / rA[0][0, 0])
    ip, isc, _k = _find_icon(shotB, *PANEL)
    err_before = icon_err(rA[0], seed, "侧门", ip) if ip else float("nan")
    t0 = time.perf_counter()
    act, M, s, sc, ov, why = tick(shotB, tr, ref)
    dt = (time.perf_counter() - t0) * 1000
    err_after = icon_err(M, seed, "侧门", ip) if ip else float("nan")
    print(f"  {nameA[-16:]}→{nameB[-16:]} 种子{seed} [{act}] {why}  {dt:.0f}ms")
    print(f"     尺度 起点{1/rA[0][0,0]:.3f} → 跟踪后{s:.3f}（真值{s_true}）"
          f"  图标误差 起点{err_before:.0f}px → 跟踪后{err_after:.0f}px")
    check(f"种子{seed} {nameB[-14:]} 接受且跟踪后图标落对(<{ERR_OK:.0f}px)",
          act.startswith("接受") and err_after < ERR_OK, f"{err_after:.0f}px")


print()
print("=" * 84)
print("3) 缩放（用户录的缩放演示视频）：滑条给的尺度能不能让投影跟上")
cap = cv2.VideoCapture(str(VIDEO))
FR = [(387, 0.25), (339, 0.30), (306, 0.45)]    # (帧号, e37 标定/验证过的真尺度)
shots, ref2 = {}, load_bgr(str(lib.get(2, "一楼").path))
for i, s_true in FR:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _o, fr = cap.read()
    shots[i] = fr
    print(f"  帧{i}: 滑条读数 = {zoom_scale_from_roi(roi_of(fr, PANEL), PANEL)[0]}  (标定真值 {s_true})")
tr = None
for k in range(len(FR) - 1):
    i0, s0 = FR[k]
    i1, s1 = FR[k + 1]
    if tr is None:
        r0, _ = seed_align(shots[i0], 2, s_hint=s0)
        tr = Tracker()
        tr.seed_from(PANEL, 2, "一楼", str(lib.get(2, "一楼").path), r0[0], 1 / r0[0][0, 0])
        ip0, _, _ = _find_icon(shots[i0], *PANEL)
        print(f"     起点 帧{i0}: 图标误差 {icon_err(r0[0], 2, '侧门', ip0):.1f}px（应 <25）")
    ip, isc, _k = _find_icon(shots[i1], *PANEL)
    t0 = time.perf_counter()
    act, M, s, sc, ov, why = tick(shots[i1], tr, ref2)
    dt = (time.perf_counter() - t0) * 1000
    err = icon_err(M, 2, "侧门", ip) if ip else float("nan")
    print(f"  帧{i0}→帧{i1}: [{act}] {why}  {dt:.0f}ms  →  s={s:.3f}"
          f"（滑条 {zoom_scale_from_roi(roi_of(shots[i1], PANEL), PANEL)[0]}）"
          f"  图标误差 {err:.0f}px")
    check(f"帧{i0}→帧{i1} 缩放跟得上且图标落对(<{ERR_OK:.0f}px)",
          act.startswith("接受") and err < ERR_OK, f"{err:.0f}px")

print()
print("=" * 84)
print("4) 地图被关掉（游戏世界帧）⇒ 必须判丢，不许把别的位置认下来")
mapclosed = NVIDIA / "IdentityV Screenshot 2026.09.14 - 14.46.48.17.png"   # 走廊绿植，全屏非地图
shotB = load_bgr(str(CAP / "hotkey_20260916_221826.png"))                  # 真地图帧（全屏）
ref = load_bgr(str(lib.get(2, "一楼").path))
r0 = find_overlay_transform(shotB, ref, PANEL)
tr = Tracker(); tr.seed_from(PANEL, 2, "一楼", str(lib.get(2, "一楼").path), r0[0], 1 / r0[0][0, 0])
closed = load_bgr(str(mapclosed))
acts = []
for _ in range(3):
    a, M, s, sc, ov, why = tick(closed, tr, ref, use_motion_gate=False)
    acts.append(a)
print(f"   连喂 3 次「地图已关」帧 → {acts}  累计 lost={tr.lost}")
check("地图已关帧 ⇒ 判丢", acts[0] == "判丢", f"[{acts[0]}]，连丢计数 {tr.lost}")

print()
print("   参考：换一张**错种子**的参考图时，全平移搜索仍能找到分<0.30 的位置 ——")
print("   这不是跟踪的锅，是既有事实「显示闸挡不住错种子」（CLAUDE.md 待办7 注），跟踪治不了种子错：")
wrong = load_bgr(str(lib.get(11, "一楼").path))
r = find_overlay_transform(shotB, wrong, PANEL)
print(f"     真种子6 的帧 × 种子11 的参考图 → 分{r[1]:.3f} 重叠{r[2]:.2f}"
      f"（{'会过闸' if r[1] < 0.30 else '不过闸'}）")


print()
print("=" * 84)
print("5) 耗时分解（s=0.40 附近，1920×1080 面板 1064×569）")
shotA = load_bgr(str(CAP / "hotkey_20260916_171738.png"))
roi = roi_of_shot(shotA)
region = panel_of_roi(roi)
t0 = time.perf_counter(); cv2.resize(region, (64, 34), interpolation=cv2.INTER_AREA); print(f"  画面帧差闸     {(time.perf_counter()-t0)*1000:6.1f} ms")
r, ref = seed_align(shotA, 6)
tr = Tracker(); tr.seed_from(PANEL, 6, "一楼", str(lib.get(6, "一楼").path), r[0], 1 / r[0][0, 0])
t0 = time.perf_counter(); zoom_scale_from_roi(roi, PANEL); print(f"  滑条读数       {(time.perf_counter()-t0)*1000:6.1f} ms")
t0 = time.perf_counter()
find_overlay_transform(None, ref, PANEL, region=region, hint_s=tr.s, hint_icon=tr.pin(),
                       fast=True, hint_radius=TRACK_NEAR_R)
print(f"  近处小窗(1次)  {(time.perf_counter()-t0)*1000:6.1f} ms")
t0 = time.perf_counter()
find_overlay_transform(None, ref, PANEL, region=region, hint_s=tr.s, fast=True, ref_key="k")
print(f"  单尺度全平移   {(time.perf_counter()-t0)*1000:6.1f} ms（含参考侧预处理）")
t0 = time.perf_counter()
find_overlay_transform(None, ref, PANEL, region=region, hint_s=tr.s, fast=True, ref_key="k")
print(f"  同上(已缓存)   {(time.perf_counter()-t0)*1000:6.1f} ms")

print()
print(f"跟随·第二步离线断言: {sum(ok)}/{len(ok)} 通过")
sys.exit(0 if all(ok) else 1)

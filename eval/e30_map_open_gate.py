# -*- coding: utf-8 -*-
"""地图开合判据：新（侧栏导航列）vs 旧（雾/结构双特征）全库对拍。

用法: python eval/e30_map_open_gate.py
输出三块：
  A. 两判据不一致的全部样本（旧开新关 = 修掉的误判；旧关新开 = 新引入的风险）
  B. 已知负样本的逐张断言（真实游戏世界 / 桌面 / 黑屏）
  C. 走雾兜底的样本（nav 低但被判开）——这些是要人眼复核的
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.vision import (FOG_OPEN_MIN, NAV_NCC_MIN, follow_features, load_bgr,
                         map_is_open, nav_column_ncc, panel_for_screen, roi_of)

NVIDIA = Path(r"D:\Videos\NVIDIA\IdentityV")
PANEL = panel_for_screen(1920, 1080)

# 已知负样本（2026-09-17 逐张看图确认）
NEG = {
    "IdentityV Screenshot 2026.08.25 - 20.02.47.44.png": "真实游戏世界(木楼梯)",
    "IdentityV Screenshot 2026.09.14 - 14.46.48.17.png": "真实游戏世界(走廊绿植)",
    "hotkey_20260916_230629.png": "桌面/IDE 截图",
    "hotkey_20260909_224332.png": "暗屏(内容 0.07)",
}

rows = []
for d, pat in ((NVIDIA, "*.png"), (ROOT / "captures", "hotkey_*.png")):
    for p in sorted(d.glob(pat)):
        shot = load_bgr(str(p))
        if shot.shape[:2] != (1080, 1920):
            continue
        reg = shot[PANEL[1]:PANEL[1] + PANEL[3], PANEL[0]:PANEL[0] + PANEL[2]]
        content, fog, struct = follow_features(reg)
        old = (fog >= 0.10) or (struct >= 0.10)
        new, why = map_is_open(shot, PANEL)
        rows.append((p.name, old, new, why, nav_column_ncc(roi_of(shot, PANEL), PANEL), fog, struct))

print(f"语料 {len(rows)} 张   新判据 = 导航列 NCC ≥ {NAV_NCC_MIN}，兜底 雾 ≥ {FOG_OPEN_MIN}")
print(f"  旧开新关 {sum(1 for r in rows if r[1] and not r[2])} 张"
      f"   旧关新开 {sum(1 for r in rows if r[2] and not r[1])} 张"
      f"   两者皆开 {sum(1 for r in rows if r[1] and r[2])} 张\n")

print("A. 两判据不一致的：")
print(f"{'图':<44}{'旧':<4}{'新':<4}{'依据':<26}{'雾':>6}{'结构':>7}")
for n, old, new, why, _nc, fog, struct in rows:
    if old != new:
        print(f"{n[-42:]:<44}{'开' if old else '关':<4}{'开' if new else '关':<4}"
              f"{why:<26}{fog:>6.2f}{struct:>7.2f}")

print("\nB. 已知负样本断言：")
ok = True
for name, desc in NEG.items():
    hit = [r for r in rows if r[0] == name]
    if not hit:
        print(f"  ⚠ {name} 不在语料里"); continue
    _n, old, new, why, _nc, _f, _s = hit[0]
    good = not new
    ok &= good
    print(f"  {'✓' if good else '✗'} {desc:<22} 新判{'开' if new else '关'}({why})"
          f"  旧判{'开' if old else '关'}  {name[-34:]}")

print("\nC. 走雾兜底被判「开」的（nav 低，需人眼复核）：")
for n, _old, new, why, nc, fog, struct in rows:
    if new and nc is not None and nc < NAV_NCC_MIN:
        print(f"   nav{nc:>6.2f} 雾{fog:.2f} 结构{struct:.2f}  {n[-42:]}")

print(f"\n负样本断言: {'全部通过' if ok else '有失败'}")

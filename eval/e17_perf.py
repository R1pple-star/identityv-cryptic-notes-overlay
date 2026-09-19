# -*- coding: utf-8 -*-
"""性能测量（只读）：热键主链路的耗时（入口匹配 / 对齐 锚定·全搜·fast）。

用法: python eval/e17_perf.py [截图名]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tomllib

from core.alignment import find_overlay_transform
from core.entrance import find_seed_by_entrance, load_index
from core.map_library import MapLibrary
from core.vision import _find_icon, detect_fog_panel, load_bgr

with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)

SHOT = sys.argv[1] if len(sys.argv) > 1 else "hotkey_20260915_213638.png"
lib = MapLibrary.load(_CFG["paths"]["map_library"])
shot = load_bgr(str(ROOT / "captures" / SHOT))
panel = detect_fog_panel(shot)
print(f"口径 = 错配率(数一致 CCORR)   截图 = {SHOT}")

ip, isc, ik = _find_icon(shot, *panel)
idx = load_index(10) or {}
ent = idx.get("侧门")
hint_icon = (ent["cx"], ent["cy"], ip[0], ip[1])
ref = load_bgr(str(lib.get(10, "一楼").path))


def timeit(fn, n=3, tag=""):
    fn()  # 预热
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    dt = (time.perf_counter() - t0) / n
    print(f"  {tag:<34} {dt * 1000:8.1f} ms")
    return dt, r


print("入口匹配 find_seed_by_entrance（28 种子×13 尺度）:")
timeit(lambda: find_seed_by_entrance(shot, lib, "侧门", panel=panel, top_n=3), 2,
       "整体")
print("对齐 find_overlay_transform:")
timeit(lambda: find_overlay_transform(shot, ref, panel, hint_icon=hint_icon), 2,
       "锚定(全尺度粗+细+微)")
timeit(lambda: find_overlay_transform(shot, ref, panel), 2, "全搜")
timeit(lambda: find_overlay_transform(shot, ref, panel, hint_s=0.41, fast=True), 2,
       "fast 单尺度(跟随循环用)")

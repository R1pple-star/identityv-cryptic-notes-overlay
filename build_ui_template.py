# -*- coding: utf-8 -*-
"""生成 assets/ui_nav_column.png —— 地图画面侧栏导航列的模板（地图开合判据用）。

只吃**全屏实机**截图（游戏全屏、地图打开）。⚠️ 千万别喂照片查看器里的截图：查看器会把
画面缩放位移并加标题栏，裁出来的根本不是那列 UI（NCC 一反一正就是它造成的）。
判据见 core/vision.py 的 NAV_BAND 长注。

多张输入取像素均值（同一固定 UI ⇒ 均值降噪、背景是同一片雾）。脚本会打印每张输入
与成品模板的 NCC，明显偏低的那张就是喂错了（查看器截图/非全屏）。

用法:
  python build_ui_template.py <截图1> [截图2 ...]
  python build_ui_template.py captures/hotkey_20260916_221826.png ...
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from core.vision import (FIXED_PANEL, load_bgr, nav_band, map_roi, panel_for_screen)

OUT = ROOT / "assets" / "ui_nav_column.png"


def band_of(full_bgr):
    panel = panel_for_screen(full_bgr.shape[1], full_bgr.shape[0])
    if panel is None:
        return None
    x0, y0, x1, y1 = nav_band(panel)
    return full_bgr[y0:y1, x0:x1]


def ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    bands = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"  跳过（不存在）: {arg}"); continue
        shot = load_bgr(str(p))
        if shot.shape[:2] != (1080, 1920):
            print(f"  跳过（非 1920×1080）: {arg}"); continue
        b = band_of(shot)
        if b is None:
            print(f"  跳过（分辨率未适配）: {arg}"); continue
        bands.append((p.name, b))
    if not bands:
        print("没有可用输入"); return 1
    stack = np.stack([b.astype(np.float32) for _n, b in bands])
    tpl = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), tpl)
    print(f"模板 {tpl.shape[1]}×{tpl.shape[0]} → {OUT.relative_to(ROOT)}"
          f"（{len(bands)} 张均值）\n")
    print("各输入与成品模板的 NCC（应 ≥0.75；明显低的那张 = 喂了查看器截图/非全屏）:")
    for name, b in sorted(((n, ncc(b, tpl)) for n, b in bands), key=lambda t: t[1]):
        print(f"  {b:>6.3f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

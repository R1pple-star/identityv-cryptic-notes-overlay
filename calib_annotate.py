# -*- coding: utf-8 -*-
"""
地图校准工具
============
把检测到的结构叠加标注到截图上，输出一张带颜色的标注图，
让你亲眼确认「墙/房间/通路/图标/迷雾」我认得对不对，据此校准参数。

用法:
    python calib_annotate.py <截图路径> [--out 输出路径]
输出默认存到 校准输出/ 目录，是个可看的 PNG。

注：classify_region / _find_icon 直接 import core.vision 的核心实现，
不在本文件维护副本（副本必然 drift）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from core.vision import FIXED_PANEL, FOG_BGR, classify_region, _find_icon, load_bgr

OUT_DIR = Path(__file__).resolve().parent / "校准输出"

# 标注颜色 (BGR)
C_ICON = (0, 0, 255)     # 红: 图标框
C_WALL = (0, 255, 0)     # 绿: 墙(细线)
C_ROOM = (0, 165, 255)   # 橙: 房间/棕
C_PASS = (255, 0, 0)     # 蓝: 通路(暗色走廊)
C_FOG = (128, 128, 128)  # 灰: 迷雾
C_PANEL = (255, 255, 255)  # 白: 面板边界

_CMAP = {
    0: (0, 0, 0),          # dark 保留(这里给暗色底，稍后混合原图)
    1: C_WALL,
    2: C_ROOM,
    3: C_PASS,
    4: C_FOG,
}


def make_annotated(bgr_full, panel=FIXED_PANEL, overlay=True):
    """生成标注图。返回 (标注图, 图标位置或None, 图标分数)。"""
    px, py, pw, ph = panel
    img = bgr_full.copy()
    region = img[py:py + ph, px:px + pw]

    cls = classify_region(region)
    h, w = cls.shape
    vis = np.zeros_like(region)
    for cid, color in _CMAP.items():
        mask = (cls == cid)
        vis[mask] = color

    if overlay:
        region[:] = cv2.addWeighted(region, 0.35, vis, 0.65, 0)

    cv2.rectangle(img, (px, py), (px + pw, py + ph), C_PANEL, 2)
    icon_pos, icon_score, _k = _find_icon(bgr_full, px, py, pw, ph)  # 用原图检测图标（勿用已 addWeighted 标注的 img，否则白箭头被混色→检不出）
    if icon_pos:
        cv2.rectangle(img, (icon_pos[0] - 22, icon_pos[1] - 22),
                      (icon_pos[0] + 22, icon_pos[1] + 22), C_ICON, 2)
    return img, icon_pos, icon_score, cls


def make_class_only(cls, panel=FIXED_PANEL):
    """纯分类图：黑底上只显示分类色。淡色以便区分。"""
    px, py, pw, ph = panel
    canv = np.zeros((ph, pw, 3), dtype=np.uint8)
    for cid, color in _CMAP.items():
        m = (cls == cid)
        canv[m] = np.array(color, dtype=np.uint8) // 2  # 淡一点避免过艳
    return canv


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]

    if args and args[0] == "--capture":
        # 实时截屏标注
        from ui.capture import capture_monitor
        bgr = capture_monitor(1)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "实时标注.png"
        out_c = OUT_DIR / "实时标注_分类.png"
        anno, icon, score, cls = make_annotated(bgr)
        from PIL import Image
        Image.fromarray(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)).save(out)
        Image.fromarray(cv2.cvtColor(make_class_only(cls), cv2.COLOR_BGR2RGB)).save(out_c)
        print(f"已截屏并标注: {out}  (分类图: {out_c})")
        print(f"图标: {icon if icon else '未检测到(置信不足)'}  分数={score:.2f}")
        return

    if not args:
        print(__doc__)
        return
    path = args[0]
    bgr = load_bgr(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(path).stem
    out = OUT_DIR / f"{name}_标注.png"
    out_c = OUT_DIR / f"{name}_分类.png"
    anno, icon, score, cls = make_annotated(bgr)
    # 用 PIL 存（支持中文路径），cv2.imwrite 对中文路径会失败
    from PIL import Image
    Image.fromarray(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)).save(out)
    Image.fromarray(cv2.cvtColor(make_class_only(cls), cv2.COLOR_BGR2RGB)).save(out_c)
    print(f"标注图已保存: {out}")
    print(f"分类图已保存: {out_c}")
    print(f"图标: {icon if icon else '未检测到(置信不足)'}  分数={score:.2f}")


if __name__ == "__main__":
    main()

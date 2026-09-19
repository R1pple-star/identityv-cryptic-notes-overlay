# -*- coding: utf-8 -*-
"""
屏幕捕获
========
- capture_region():   截取屏幕区域，返回 numpy BGR 图
- capture_monitor():  截取整个显示器
- monitor_size():     显示器物理尺寸
"""
from __future__ import annotations

import numpy as np

# 常驻 mss 实例（懒初始化单例）。跟随轮询 250ms 一次，若每 tick `with mss.mss()`
# 现开现关，4次/秒的 GDI DC 开合循环会干扰英伟达截屏（软件常驻时 NVIDIA 截图失效，
# 关软件才恢复）。所有调用都在 Qt 主线程，无线程问题。
_SCT = None


def _sct():
    global _SCT
    if _SCT is None:
        import mss
        _SCT = mss.mss()
    return _SCT


def capture_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    """截取屏幕区域，返回 BGR ndarray（供 OpenCV 使用）。"""
    region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
    img = _sct().grab(region)
    # mss 返回 BGRA；转成 BGR
    bgra = np.asarray(img, dtype=np.uint8)
    return bgra[:, :, :3]


def monitor_size(index: int = 1) -> tuple[int, int]:
    """显示器物理尺寸 (宽, 高)。index 从 1 开始（1=主显示器）。

    复用常驻单例 —— **不要**在跟随循环里 `with mss.mss()` 现开现关：4 次/秒的
    GDI DC 开合会毁英伟达截图，单例之后才不成立。
    """
    m = _sct().monitors[index]
    return int(m["width"]), int(m["height"])


def capture_monitor(index: int = 1) -> np.ndarray:
    """截取整个显示器，返回 BGR ndarray。index 从 1 开始（1=主显示器）。"""
    mon = _sct().monitors[index]
    img = _sct().grab(mon)
    bgra = np.asarray(img, dtype=np.uint8)
    return bgra[:, :, :3]

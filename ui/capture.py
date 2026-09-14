# -*- coding: utf-8 -*-
"""
屏幕捕获 + 游戏窗口识别
=======================
- list_windows(): 枚举顶层窗口（标题 + 句柄），用于找到游戏窗口
- find_window():  按标题关键字找窗口
- get_window_rect(): 获取窗口矩形（left/top/width/height）
- capture():      用 mss 截取指定区域/显示器，返回 numpy BGR 图
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def list_windows() -> list[tuple[int, str]]:
    """返回 [(hwnd, title)]，按标题排序，过滤空标题。"""
    import win32gui
    out: list[tuple[int, str]] = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                out.append((hwnd, title))

    win32gui.EnumWindows(cb, None)
    return sorted(out, key=lambda x: x[1].lower())


def find_window(keyword: str) -> Optional[int]:
    """按标题关键字（不区分大小写）找第一个匹配的窗口，返回句柄。"""
    import win32gui
    for hwnd, title in list_windows():
        if keyword.lower() in title.lower():
            return hwnd
    return None


def get_window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """返回 (left, top, width, height)。"""
    import win32gui
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return left, top, right - left, bottom - top
    except Exception:  # noqa: BLE001
        return None


# 常驻 mss 实例（懒初始化单例）。跟随轮询 250ms 一次，旧版每 tick `with mss.mss()`
# 新建+销毁 DC 句柄，4次/秒的 GDI 开合循环会干扰英伟达截屏（2026-09-14 用户实测：
# 软件常驻时 NVIDIA 截图失效，关软件才恢复）。所有调用都在 Qt 主线程，无线程问题。
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


def capture_monitor(index: int = 1) -> np.ndarray:
    """截取整个显示器，返回 BGR ndarray。index 从 1 开始（1=主显示器）。"""
    mon = _sct().monitors[index]
    img = _sct().grab(mon)
    bgra = np.asarray(img, dtype=np.uint8)
    return bgra[:, :, :3]


def capture_window(hwnd: int) -> Optional[np.ndarray]:
    """按窗口矩形截取。窗口可能被遮挡/最小化，仅作参考。"""
    rect = get_window_rect(hwnd)
    if not rect:
        return None
    left, top, w, h = rect
    return capture_region(left, top, w, h)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print("当前可见窗口：")
    for hwnd, title in list_windows():
        print(f"  [{hwnd}] {title}")

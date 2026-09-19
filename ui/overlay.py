# -*- coding: utf-8 -*-
"""
透明悬浮窗模块
==============
一个 无边框 / 置顶 / 点击穿透 的透明窗口，用于把对齐后的完整地图投到游戏上。

要点：
- 点击穿透：鼠标事件直接穿透到下面的游戏，不挡操作
- 不抢焦点：不打断游戏输入
- 透明背景：只显示地图内容(墙体/房间)
"""
from __future__ import annotations

import ctypes

import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

# 投影窗从截屏画面排除（屏上正常显示，但 mss/热键全屏截屏/OBS/录屏拍不到）。
# 禁用 WDA_MONITOR(0x1)：全屏窗会整块变黑，热键全屏截屏全毁；用 WDA_EXCLUDEFROMCAPTURE(0x11)。
WDA_EXCLUDEFROMCAPTURE = 0x11


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()  # 深拷贝，避免底层数组被回收


class MapOverlay(QWidget):
    def __init__(self, geometry: QRect, exclude_capture: bool = True):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput  # 点击穿透
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(geometry)

        self._label = QLabel(self)
        self._label.setGeometry(0, 0, geometry.width(), geometry.height())
        self._label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._label.setScaledContents(False)

        # 截屏排除：防自截污染——投影墙体若进截屏会被 classify 成 structure 混进样本/探明掩膜，
        # 热键重匹配会吃到自己的投影（开合判定走导航列/雾不受影响，但**匹配**路径照样会被污染）。
        # 注意：winId() 已强制创建原生窗，此后不得再 setWindowFlags（重建 HWND 丢 affinity）。
        self.capture_excluded = self._set_capture_exclusion() if exclude_capture else False

    def _set_capture_exclusion(self, affinity: int = WDA_EXCLUDEFROMCAPTURE) -> bool:
        """把本窗从截屏画面排除。失败（offscreen 平台/旧系统）返回 False，绝不抛。"""
        try:
            return bool(ctypes.windll.user32.SetWindowDisplayAffinity(
                int(self.winId()), affinity))
        except Exception:  # noqa: BLE001
            return False

    def set_image(self, rgba: np.ndarray) -> None:
        """rgba: (h, w, 4) uint8。"""
        if rgba is None:
            self._label.clear()
            return
        qimg = rgba_to_qimage(rgba)
        self._label.setPixmap(QPixmap.fromImage(qimg))

    def clear(self) -> None:
        self._label.clear()

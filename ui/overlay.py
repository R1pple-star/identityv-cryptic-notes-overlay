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

import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()  # 深拷贝，避免底层数组被回收


class MapOverlay(QWidget):
    def __init__(self, geometry: QRect):
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

    def set_image(self, rgba: np.ndarray) -> None:
        """rgba: (h, w, 4) uint8。"""
        if rgba is None:
            self._label.clear()
            return
        qimg = rgba_to_qimage(rgba)
        self._label.setPixmap(QPixmap.fromImage(qimg))

    def clear(self) -> None:
        self._label.clear()

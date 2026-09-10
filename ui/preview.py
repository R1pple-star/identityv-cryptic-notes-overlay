# -*- coding: utf-8 -*-
"""
样本预览窗（阶段3）
==================
独立普通窗口，显示「软件拿来匹配的入口样本」+ 图标位置框，让玩家看到匹配
用的是哪块区域——选错了能立刻发现。贴主控制窗旁，可拖动。
手框纠错见 app.py 的 ClickPicker(rect 模式) + _manual_sample_pick。
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SamplePreview(QWidget):
    """显示一张 BGR 样本图（+ 可选图标框）。无边框置顶可拖动。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setWindowTitle("入口样本预览")
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background:#222; color:#888; font-size:11px;")
        self._label.setText("（无样本）")
        self._label.setFixedSize(280, 280)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        self.resize(282, 282)
        self._drag = None

    def set_sample(self, bgr: np.ndarray | None, icon_box=None):
        """bgr: HxWx3 BGR。icon_box: (x,y,w,h) 在 bgr 坐标，可选（画黄框）。"""
        if bgr is None:
            self._label.setText("（无样本）"); self._label.setPixmap(QPixmap()); return
        h, w = bgr.shape[:2]
        maxw, maxh = 280, 280
        sc = min(1.0, maxw / w, maxh / h) if w > 0 and h > 0 else 1.0
        dw, dh = max(1, int(w * sc)), max(1, int(h * sc))
        small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        if icon_box is not None:
            x, y, bw, bh = icon_box
            qx, qy = int(x * sc), int(y * sc)
            qw, qh = int(bw * sc), int(bh * sc)
            cv2.rectangle(small, (qx, qy), (qx + qw, qy + qh), (0, 255, 255), 2)
        qimg = QImage(small.tobytes(), dw, dh, 3 * dw,
                      QImage.Format.Format_RGB888).rgbSwapped()
        self._label.setPixmap(QPixmap.fromImage(qimg))

    # ---- 无边框拖动 ----
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

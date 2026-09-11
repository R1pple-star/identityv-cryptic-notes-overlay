# -*- coding: utf-8 -*-
"""
加页手记 · 摸金地图叠层工具（重构版）
======================================
热键(Ctrl+Shift+F) → 入口引索匹配定种子 → 两段式对齐 → 透明投影重合。

一次热键 = 一个任务 = 一个结果。自动跟随状态机已按计划推迟（见 CLAUDE.md /
重构计划-v2.md §7）；跟随实测可靠后再立项。废弃的整图识别(find_seed_submap /
find_seed_color / detect_direction)已删，全部走入口引索 + 两段式对齐。

双闸（见 CLAUDE.md）：
  入口置信闸 = 入口分 < score_confident(0.10) 且 overlap ≥ overlap_min(0.40)（种子ID可信）
  对齐显示闸 = 重合分 < align_score_max(0.30) 且 overlap ≥ 0.40（投影该显示）
"""
from __future__ import annotations

import csv
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QRubberBand, QSlider, QVBoxLayout, QWidget,
)

from core.alignment import (
    affine_from_points, auto_align_overlay, find_overlay_transform, map_to_overlay_rgba,
)
from core.entrance import (
    _crop_around_icon, build_entrance_transform, find_seed_by_entrance, load_index,
)
from core.map_library import MapLibrary
from core.vision import FIXED_PANEL, detect_fog_panel, load_bgr
from ui.capture import capture_monitor
from ui.hotkey import MOD_CONTROL, MOD_SHIFT, HotkeyManager, parse_hotkey
from ui.manage_materials import ManageMaterialsDialog
from ui.overlay import MapOverlay
from ui.preview import SamplePreview
from ui.settings import Settings, SettingsDialog, load as load_settings, save as save_settings

# DPI 缩放适配：让进程用物理像素（per-monitor DPI aware + Qt 禁用 high-DPI scaling），
# 使 mss 截屏、Qt 窗口坐标、FIXED_PANEL 三者统一于物理像素。否则 125%/150% 缩放下
# 截屏=逻辑像素而 FIXED_PANEL=物理坐标，错位致图标检偏、匹配退化(分0.000)、投影位置不对。
import ctypes
import os
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
except Exception:  # noqa: BLE001  旧 Windows 无 shcore
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parent
with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)
MAP_DIR = _CFG["paths"]["map_library"]
CAPTURES_DIR = ROOT / _CFG["paths"]["captures"]
EVAL_INBOX = ROOT / _CFG["paths"]["eval_inbox"]
SCORE_CONFIDENT = float(_CFG["match"]["score_confident"])
ALIGN_SCORE_MAX = float(_CFG["match"]["align_score_max"])
OVERLAP_MIN = float(_CFG["match"]["overlap_min"])

# 入口判别力降序：侧门/二楼房间形状各异（主要判别依据）；正门固定分不出种子。
ENTRANCE_TYPES = ("侧门", "二楼", "正门")


class ClickPicker(QDialog):
    """在图上点 N 个点 或 拖一个矩形（原图坐标）。

    mode='point'(默认): 点 n 个点，self.pts=[(x,y)...]。
    mode='rect': 拖一个矩形，self.rect=(x0,y0,x1,y1)（原图坐标，含起止）。
    点/框画在 pixmap 副本上，进度文本走独立 _status（不 setText 图 label，避免清 pixmap 黑屏）。
    """

    def __init__(self, bgr, n: int, title: str, parent=None, mode: str = "point"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.n = n
        self.mode = mode
        self.pts: list[tuple[int, int]] = []
        self.rect: tuple[int, int, int, int] | None = None
        self._scale = 1.0
        self._origin = None  # rect 起点（窗坐标）
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._label.setStyleSheet("background:#111;")
        self._status = QLabel(self)  # 进度文本独立 label，不碰图 label
        self._status.setStyleSheet("color:#ffcc66; padding:4px; background:#222;")
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(self._label); lay.addWidget(self._status)
        self._rubber = (QRubberBand(QRubberBand.Shape.Rectangle, self)
                        if mode == "rect" else None)
        self._orig_pix = None
        self._set_image(bgr)

    def _set_image(self, bgr):
        h, w = bgr.shape[:2]
        maxw, maxh = 1280, 720
        self._scale = min(1.0, maxw / w, maxh / h)
        dw, dh = int(w * self._scale), int(h * self._scale)
        small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        qimg = QImage(small.tobytes(), dw, dh, 3 * dw, QImage.Format.Format_RGB888).rgbSwapped()
        self._orig_pix = QPixmap.fromImage(qimg)
        self._label.setPixmap(self._orig_pix)
        self._label.setFixedSize(dw, dh)
        self._status.setText("点 %d 个点（顺序自定）" % self.n if self.mode == "point"
                             else "拖框选一个矩形区域")
        self.adjustSize()

    def _draw_points(self):
        if self._orig_pix is None:
            return
        pix = self._orig_pix.copy()
        painter = QPainter(pix)
        pen = QPen(QColor(255, 255, 0)); pen.setWidth(3); painter.setPen(pen)
        for (x, y) in self.pts:
            px, py = int(x * self._scale), int(y * self._scale)
            painter.drawEllipse(px - 6, py - 6, 12, 12)
        painter.end()
        self._label.setPixmap(pix)

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = e.position().toPoint()
        if self.mode == "rect":
            self._origin = p
            self._rubber.setGeometry(p.x(), p.y(), 0, 0)
            self._rubber.show()
            return
        x, y = int(p.x() / self._scale), int(p.y() / self._scale)
        self.pts.append((x, y))
        self._draw_points()
        self._status.setText(f"已点 {len(self.pts)}/{self.n}：{self.pts}")
        if len(self.pts) >= self.n:
            self.accept()

    def mouseMoveEvent(self, e):
        if self.mode == "rect" and self._origin is not None:
            p = e.position().toPoint()
            x0, y0 = min(self._origin.x(), p.x()), min(self._origin.y(), p.y())
            w, h = abs(p.x() - self._origin.x()), abs(p.y() - self._origin.y())
            self._rubber.setGeometry(x0, y0, w, h)

    def mouseReleaseEvent(self, e):
        if self.mode == "rect" and self._origin is not None:
            p = e.position().toPoint()
            x0, y0 = min(self._origin.x(), p.x()), min(self._origin.y(), p.y())
            x1, y1 = max(self._origin.x(), p.x()), max(self._origin.y(), p.y())
            self._origin = None
            rx0, ry0 = int(x0 / self._scale), int(y0 / self._scale)
            rx1, ry1 = int(x1 / self._scale), int(y1 / self._scale)
            if (rx1 - rx0) > 10 and (ry1 - ry0) > 10:
                self.rect = (rx0, ry0, rx1, ry1)
                self._status.setText(f"已框: {self.rect}（确认中）")
                self.accept()
            else:
                self._rubber.hide()
                self._status.setText("框太小，重拖")


class MainWindow(QWidget):
    def __init__(self, lib: MapLibrary, settings: Settings | None = None):
        super().__init__()
        self.lib = lib
        self.settings = settings if settings is not None else load_settings()
        self._shot = None           # 最近捕获的屏幕 BGR
        self._seed = None           # 当前选定种子（方向+门解析）
        self.overlay: MapOverlay | None = None
        self.preview: SamplePreview | None = None  # 入口样本预览窗（阶段3）
        self._last_shot_path = None  # 最近截图保存路径（log/回收用）

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setWindowTitle("加页手记 地图助手")
        self.setFixedWidth(300)

        # 状态灯：启动/热键成败一目了然（阶段1）
        self.led = QLabel("● 启动中…")
        self.led.setWordWrap(True)
        self.led.setStyleSheet("color:#ffcc66; font-weight:bold; padding:5px; "
                               "background:#222; border-radius:3px;")
        # 运行日志区：逐步显示 截屏→图标→匹配→对齐→投影（阶段1）
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setFixedHeight(150)
        self.log_view.setStyleSheet("background:#1a1a1a; color:#ccc; font-size:11px;")
        self.log_view.setVisible(self.settings.show_log)

        self.direction_combo = QComboBox(); self.direction_combo.addItems(lib.directions())
        self.door_combo = QComboBox()
        self.floor_combo = QComboBox(); self.floor_combo.addItems(["一楼", "二楼"])
        self.entrance_combo = QComboBox(); self.entrance_combo.addItems(list(ENTRANCE_TYPES))
        self.seed_label = QLabel("")
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        self.door_combo.currentIndexChanged.connect(self._resolve_seed)

        self.btn_realign = QPushButton("↻ 重新对齐(当前门)")
        self.btn_calib = QPushButton("✋ 3点标定(手动兜底)")
        self.btn_picksample = QPushButton("✂ 手框样本纠错")
        self.btn_mark_wrong = QPushButton("✗ 标记上次错→回收")
        self.btn_hide = QPushButton("👁 隐藏地图")
        self.btn_settings = QPushButton("⚙ 设置")
        self.btn_manage = QPushButton("🗂 素材管理")
        self.btn_quit = QPushButton("✕ 退出")
        self.btn_realign.clicked.connect(self._realign)
        self.btn_calib.clicked.connect(self._three_point_calib)
        self.btn_picksample.clicked.connect(self._manual_sample_pick)
        self.btn_mark_wrong.clicked.connect(self._mark_wrong)
        self.btn_hide.clicked.connect(self._hide_overlay)
        self.btn_settings.clicked.connect(self._open_settings)
        self.btn_manage.clicked.connect(self._manage_materials)
        self.btn_quit.clicked.connect(self._quit)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(int(self.settings.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        root = QVBoxLayout(); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(5)
        root.addWidget(self.led)
        for text, widget in [("方向（入口朝向）", self.direction_combo),
                             ("门特征", self.door_combo),
                             ("楼层", self.floor_combo),
                             ("入口(引索匹配用)", self.entrance_combo)]:
            root.addWidget(QLabel(text)); root.addWidget(widget)
        root.addWidget(self.seed_label)
        root.addWidget(QLabel("Ctrl+Shift+F = 入口匹配+对齐"))
        root.addWidget(self.btn_realign)
        root.addWidget(self.btn_calib)
        root.addWidget(self.btn_picksample)
        root.addWidget(self.btn_mark_wrong)
        root.addWidget(self.btn_hide)
        root.addWidget(self.btn_settings)
        root.addWidget(self.btn_manage)
        root.addWidget(self.btn_quit)
        root.addWidget(QLabel("地图透明度")); root.addWidget(self.opacity_slider)
        self.status = QLabel("就绪。Ctrl+Shift+F 入口匹配（需先按 g 打开地图、刚进入口）。")
        self.status.setStyleSheet("color:#aaa; word-wrap:break-word; font-size:11px;")
        root.addWidget(self.status)
        root.addWidget(QLabel("运行日志"))
        root.addWidget(self.log_view)
        root.addStretch(1)
        self.setLayout(root)
        self.move(8, 60)
        self._on_direction_changed()

    # ---- 无边框拖动 ----
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if getattr(self, "_drag_pos", None):
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    # ---- 方向/门/楼层 ----
    def _on_direction_changed(self):
        direction = self.direction_combo.currentText()
        doors = self.lib.doors_for_direction(direction)
        current = self.door_combo.currentText()
        self.door_combo.blockSignals(True)
        self.door_combo.clear(); self.door_combo.addItems(doors)
        if current in doors:
            self.door_combo.setCurrentText(current)
        self.door_combo.blockSignals(False)
        self._resolve_seed()

    def _resolve_seed(self):
        direction = self.direction_combo.currentText()
        door = self.door_combo.currentText()
        self._seed = self.lib.find_by_clue(direction, door)
        if self._seed is None:
            self.seed_label.setText("未找到该门"); self.seed_label.setStyleSheet("color:#ff6666;")
        else:
            self.seed_label.setText(f"→ 种子 {self._seed}"); self.seed_label.setStyleSheet("color:#66ff66;")

    def _current_map(self):
        if not getattr(self, "_seed", None):
            return None
        return self.lib.get(self._seed, self.floor_combo.currentText())

    # ---- 状态灯 / 运行日志（阶段1）----
    def set_led(self, ok: bool, text: str):
        """启动/热键成败状态灯。ok=True 绿，False 红。"""
        color = "#66ff66" if ok else "#ff6666"
        self.led.setText(f"● {text}")
        self.led.setStyleSheet(f"color:{color}; font-weight:bold; padding:5px; "
                               f"background:#222; border-radius:3px;")

    def _log_step(self, msg: str, level: str = "INFO"):
        """一步运行日志：append 进日志区 + 同步 status 一句话。level: INFO/OK/WARN/ERROR。"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        color = {"ERROR": "#ff6666", "WARN": "#ffcc66", "OK": "#66ff66"}.get(level, "#cccccc")
        self.log_view.appendHtml(f'<span style="color:{color}">[{ts}] {msg}</span>')
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.status.setText(msg if len(msg) <= 56 else msg[:53] + "…")

    # ---- 捕获 ----
    def _capture(self) -> bool:
        try:
            self._shot = capture_monitor(1)
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = CAPTURES_DIR / f"hotkey_{ts}.png"
            cv2.imwrite(str(path), self._shot)
            self._last_shot_path = path
            return True
        except Exception as e:  # noqa: BLE001
            self._log_step(f"截屏失败: {e}", "ERROR")
            return False

    # ---- 一次热键流程：入口引索匹配 → 两段式对齐 → 投影 ----
    def _entrance_pipeline(self):
        """热键回调入口。顶层兜底：任何步异常进日志区，不冒泡到 nativeEventFilter。"""
        try:
            self._entrance_pipeline_impl()
        except Exception:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            self._log_step("流程异常: " + tb.strip().splitlines()[-1], "ERROR")
            self._log_step(tb, "ERROR")

    def _entrance_pipeline_impl(self):
        if not self._capture():
            return
        self._log_step("截屏 OK")
        shot = self._shot
        et = self.entrance_combo.currentText()
        res, icon_pos, isc = find_seed_by_entrance(shot, self.lib, et, top_n=3)
        if icon_pos is None:
            self._log_step(f"入口图标未检出(分{isc:.2f}) → 手框样本/3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=False)
            return
        self._log_step(f"入口图标 @({icon_pos[0]},{icon_pos[1]}) 分{isc:.2f}")
        sample = _crop_around_icon(shot, FIXED_PANEL, icon_pos, self.settings.sample_half_frac)
        self._show_sample_preview(sample)  # 让玩家看到匹配用的样本
        if not res:
            self._log_step("入口匹配无结果 → 手框样本/3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=False)
            return
        best = res[0]
        self._log_step(f"入口匹配: 种子{best[1]}({best[2]}[{best[3]}]) 分{best[0]:.3f} top3="
                       + str([(r[1], round(r[0], 3)) for r in res[:3]]), "OK")
        self._after_match(shot, best, et, icon_pos, isc, res, corrected=False)

    def _after_match(self, shot, best, et, icon_pos, isc, res, corrected=False):
        """拿到 best 后：填 UI + 两段式对齐 + 投影 + 归档。供热键/手框样本复用。"""
        sc, seed, key, fl, _s, _mloc = best
        # 退化防御：top1 与 top2 分差<0.001（多种子同分0.000）→ 样本退化/图标假阳，不假阳报告
        if len(res) >= 2 and sc < 0.001 and (res[1][0] - sc) < 0.001:
            self._log_step(f"匹配退化(多种子同分 {sc:.3f}) → 图标可能假阳/贴边，请手框样本或3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=corrected)
            return
        self.floor_combo.blockSignals(True); self.floor_combo.setCurrentText(fl)
        self.floor_combo.blockSignals(False)
        bdir, door = key.split("-", 1)
        self.direction_combo.setCurrentText(bdir)
        doors = [self.door_combo.itemText(i) for i in range(self.door_combo.count())]
        if door in doors:
            self.door_combo.setCurrentText(door)

        align = self._two_stage_align(shot, best, et, icon_pos)
        ov = asc = None
        show_ok = False
        if align is not None:
            _M, asc, ov = align
            show_ok = (asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN)
            self._log_step(f"两段式对齐: 重合分{asc:.3f} 重叠{ov:.2f} "
                           + ("过闸✓" if show_ok else "不过闸"),
                           "OK" if show_ok else "WARN")
            if show_ok:
                info = self.lib.get(seed, fl)
                if info is not None:
                    rgba = map_to_overlay_rgba(str(info.path), align[0], *self._screen_size(), wall_alpha=self.settings.wall_alpha)
                    self._show_overlay(rgba)
                    self._log_step("投影已显示", "OK")
                else:
                    self._show_centered_if_any(seed, fl)
            else:
                self._show_centered_if_any(seed, fl)
        else:
            self._show_centered_if_any(seed, fl)
            self._log_step("对齐失败(探明不足)，居中显示", "WARN")

        confident = (sc < SCORE_CONFIDENT and ov is not None and ov >= OVERLAP_MIN)
        self._log(et, res, icon_pos, isc, align, corrected=corrected)
        ov_txt = f"{ov:.2f}" if ov is not None else "-"
        self.status.setText(
            f"入口{et}(图标{isc:.2f}) 入口分{sc:.3f} 重叠{ov_txt} → 种子{seed}({key}[{fl}])"
            + (" ✓确信已对齐" if (confident and show_ok)
               else " 不确信，可手动改门/3点标定"))

    def _two_stage_align(self, shot, best, entrance_type, icon_pos):
        """两段式对齐：第一段 M1(入口图标对应+匹配尺度) → 第二段 find_overlay_transform(hint_s) 精修；
        不过对齐显示闸则回退全搜。返回 (M, score, overlap) 或 None。"""
        _sc, seed, _key, fl, _s, _mloc = best
        info = self.lib.get(seed, fl)
        if info is None:
            return None
        ref = load_bgr(str(info.path))
        panel = detect_fog_panel(shot)
        if panel is None:
            self.status.setText("未检测到地图迷雾面板——请先按 g 打开地图")
            return None
        # 第一段：构造 M1 + 对齐尺度提示 hint_s（=入口匹配尺度 s）
        hint_s = None
        idx = load_index(seed)
        if idx is not None:
            m1 = build_entrance_transform(best, entrance_type, idx, icon_pos, panel)
            if m1 is not None:
                _M1, hint_s = m1
        # 第二段：hint 精修；过闸即用
        if hint_s is not None:
            align = find_overlay_transform(shot, ref, panel, hint_s=hint_s)
            if align is not None:
                _M, asc, ov = align
                if asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN:
                    return align
        # 回退：全搜（baseline verify_entrance_e2e 证 17.13/17.11 全搜可靠）
        return find_overlay_transform(shot, ref, panel)

    def _realign(self):
        """用当前选定的方向+门+楼层重新对齐（全搜，无入口 hint）。"""
        info = self._current_map()
        if info is None:
            self.status.setText("请先选好方向+门"); return
        if self._shot is None and not self._capture():
            return
        panel = detect_fog_panel(self._shot)
        if panel is None:
            self.status.setText("未检测到地图迷雾面板——请先按 g 打开地图"); return
        ref = load_bgr(str(info.path))
        align = find_overlay_transform(self._shot, ref, panel)
        if align is None:
            self.status.setText("对齐失败：探明不足"); return
        M, asc, ov = align
        if asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN:
            rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size(), wall_alpha=self.settings.wall_alpha)
            self._show_overlay(rgba)
            self.status.setText(f"已对齐(匹配{asc:.2f} 重叠{ov:.2f})：{info.key}")
        else:
            self._show_centered(ref)
            self.status.setText(f"对齐不可靠(匹配{asc:.2f} 重叠{ov:.2f})，居中显示：{info.key}")

    def _three_point_calib(self):
        """手动兜底：3 点标定（affine_from_points，3 下点击必对）。

        在截图上点 3 个特征点，再在参考图上点对应 3 点，构造 ref→screen 仿射变换投影。
        """
        if self._shot is None and not self._capture():
            return
        info = self._current_map()
        if info is None:
            self.status.setText("请先选好方向+门（确定参考图）"); return
        ref = load_bgr(str(info.path))
        pk1 = ClickPicker(self._shot, 3, "点 3 个游戏内特征点（顺序自定）", parent=self)
        if not pk1.exec():
            return
        pk2 = ClickPicker(ref, 3, "点参考图上对应的 3 点（同顺序）", parent=self)
        if not pk2.exec() or len(pk2.pts) < 3:
            return
        M = affine_from_points(pk2.pts, pk1.pts)  # src=ref, dst=screen
        if M is None:
            self.status.setText("3点标定失败（点不足）"); return
        rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size(), wall_alpha=self.settings.wall_alpha)
        self._show_overlay(rgba)
        self.status.setText(f"3点标定投影：{info.key}")

    def _manual_sample_pick(self):
        """手动框选入口样本：在最近截图上拖矩形，作为 sample 重跑匹配（纠错）。"""
        if self._shot is None and not self._capture():
            return
        pk = ClickPicker(self._shot, 1, "拖框选入口样本区域", parent=self, mode="rect")
        if not pk.exec() or not pk.rect:
            self._log_step("手框取消", "INFO"); return
        x0, y0, x1, y1 = pk.rect
        sample = self._shot[y0:y1, x0:x1].copy()
        self._show_sample_preview(sample)
        et = self.entrance_combo.currentText()
        self._log_step(f"手框样本 {x1-x0}x{y1-y0}，重跑匹配", "INFO")
        res, icon_pos, isc = find_seed_by_entrance(
            self._shot, self.lib, et, top_n=3, sample_crop=sample)
        if not res:
            self._log_step("手框样本仍无匹配 → 检查框选或用3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=True)
            return
        best = res[0]
        self._log_step(f"手框匹配: 种子{best[1]}({best[2]}[{best[3]}]) 分{best[0]:.3f}", "OK")
        self._after_match(self._shot, best, et, icon_pos, isc, res, corrected=True)

    # ---- 投影显示 ----
    def _screen_size(self):
        # 用 mss 主显示器物理尺寸（DPI aware 后=物理像素），与截屏/FIXED_PANEL 一致；
        # 不用 Qt primaryScreen.geometry()（DPI 缩放下可能返回逻辑像素致错位）
        import mss
        with mss.mss() as sct:
            m = sct.monitors[1]
            return int(m["width"]), int(m["height"])

    def _show_overlay(self, rgba):
        if self.overlay is None:
            sw, sh = self._screen_size()
            self.overlay = MapOverlay(QRect(0, 0, sw, sh))
        self.overlay.set_image(rgba)
        self.overlay.setWindowOpacity(self.opacity_slider.value() / 100.0)
        self.overlay.show()

    def _show_sample_preview(self, bgr):
        """显示匹配用的入口样本（让玩家看到选对没），贴主窗右侧。"""
        if self.preview is None:
            self.preview = SamplePreview()
            g = self.geometry()
            self.preview.move(g.right() + 8, g.top())
        self.preview.set_sample(bgr)
        self.preview.show()

    def _show_centered(self, ref):
        sw, sh = self._screen_size()  # _screen_size 返回 (width, height)
        rgba, (ox, oy) = auto_align_overlay(ref, FIXED_PANEL, rotate=0)
        rh, rw = rgba.shape[:2]
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(sw, ox + rw), min(sh, oy + rh)
        if x1 <= x0 or y1 <= y0:
            # 地图超出屏幕(屏幕非1920x1080/缩放/面板坐标不符)：直接显示不贴位
            self._show_overlay(rgba)
            return
        full = np.zeros((sh, sw, 4), dtype=np.uint8)  # (height, width, 4)
        full[y0:y1, x0:x1] = rgba[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        self._show_overlay(full)

    def _show_centered_if_any(self, seed, floor):
        info = self.lib.get(seed, floor)
        if info is not None:
            self._show_centered(load_bgr(str(info.path)))

    def _hide_overlay(self):
        if self.overlay is not None:
            self.overlay.hide()

    def _set_opacity(self, v):
        if self.overlay is not None:
            self.overlay.setWindowOpacity(v / 100.0)

    # ---- 失败回收（§6.3）：log.csv + inbox ----
    def _log(self, entrance_type, res, icon_pos, isc, align, corrected: bool):
        try:
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            log_path = CAPTURES_DIR / "log.csv"
            top3 = res[:3] if res else []
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self._last_shot_path.name if self._last_shot_path else "",
                entrance_type, f"{isc:.3f}",
                icon_pos[0] if icon_pos else "", icon_pos[1] if icon_pos else "",
                "|".join(f"{r[1]}:{r[0]:.3f}" for r in top3),
                f"{align[2]:.3f}" if align else "",      # overlap
                f"{align[1]:.3f}" if align else "",      # align score
                "1" if (align and align[1] < ALIGN_SCORE_MAX and align[2] >= OVERLAP_MIN) else "0",
                "1" if corrected else "0",
            ]
            new = not log_path.exists()
            with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "shot", "entrance", "icon_score", "ix", "iy",
                                "top3", "overlap", "align_score", "show_gate", "corrected"])
                w.writerow(row)
        except Exception as e:  # noqa: BLE001  CSV 归档失败不影响主流程，但提示用户
            try:
                self._log_step(f"日志归档失败: {e}", "WARN")
            except Exception:  # noqa: BLE001
                import sys
                print(f"[log] CSV 归档失败: {e}", file=sys.stderr)

    def _mark_wrong(self):
        """把上次截图复制到 eval/inbox/，供定期标注进 labels.csv。"""
        if self._last_shot_path is None or not self._last_shot_path.exists():
            self.status.setText("没有可回收的截图"); return
        EVAL_INBOX.mkdir(parents=True, exist_ok=True)
        dst = EVAL_INBOX / self._last_shot_path.name
        try:
            import shutil
            shutil.copy2(self._last_shot_path, dst)
            self.status.setText(f"已回收失败样本→ {dst.name}（待标注进 eval/labels.csv）")
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"回收失败: {e}")

    # ---- 设置（阶段2）----
    def _open_settings(self):
        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() and dlg.result_settings is not None:
            self.settings = dlg.result_settings
            save_settings(self.settings)
            self._apply_settings()

    def _apply_settings(self):
        """应用当前 settings：透明度/日志/墙体 alpha 即时；热键重注册。"""
        self.log_view.setVisible(self.settings.show_log)
        self.opacity_slider.setValue(int(self.settings.overlay_opacity * 100))
        self._set_opacity(self.opacity_slider.value())
        # wall_alpha 下次投影生效（map_to_overlay_rgba 读 self.settings.wall_alpha）
        hk = getattr(self, "_hotkeys", None)
        if hk is not None:
            try:
                hk.unregister_all()
                vk, mods = parse_hotkey(self.settings.hotkey)
                hk.register(vk, self._entrance_pipeline, mods)
                self.set_led(True, f"已启动 · {self.settings.hotkey} 已注册")
                self._log_step(f"热键已改为: {self.settings.hotkey}", "OK")
            except Exception as e:  # noqa: BLE001
                self.set_led(False, "热键失败")
                self._log_step(f"热键重注册失败: {e}（改回设置或重启）", "ERROR")

    # ---- 素材管理 / 退出 ----
    def _manage_materials(self):
        dlg = ManageMaterialsDialog(str(self.lib.base_dir), on_change=self._reload_lib, parent=self)
        dlg.exec()

    def _reload_lib(self):
        self.lib = MapLibrary.load(self.lib.base_dir)
        self._on_direction_changed()

    def _quit(self):
        if self.overlay is not None:
            self.overlay.close()
        if self.preview is not None:
            self.preview.close()
        hk = getattr(self, "_hotkeys", None)
        if hk is not None:
            try:
                hk.unregister_all()
            except Exception:  # noqa: BLE001
                pass
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    settings = load_settings()
    lib = MapLibrary.load(MAP_DIR)
    win = MainWindow(lib, settings)
    win.show()

    hotkeys = HotkeyManager()
    try:
        vk, mods = parse_hotkey(settings.hotkey)
        hotkeys.register(vk, win._entrance_pipeline, mods)
        win.set_led(True, f"已启动 · {settings.hotkey} 已注册")
        win._log_step(f"热键已注册: {settings.hotkey}", "OK")
        sw, sh = win._screen_size()
        win._log_step(f"屏幕 {sw}x{sh} 面板 {tuple(FIXED_PANEL)}（仅 1920x1080 精确）", "INFO")
        win._log_step("就绪：先按 g 打开游戏地图、刚进入口(图标在视野)，再 "
                      + settings.hotkey, "INFO")
    except Exception as e:  # noqa: BLE001
        win.set_led(False, "热键失败")
        win._log_step(f"热键注册失败: {e}（手动选门/3点标定仍可用）", "ERROR")
        QMessageBox.warning(win, "热键注册失败",
            f"{e}\n\n手动选门 + 3 点标定仍可用。\n（可在「设置」改热键）")
    win._hotkeys = hotkeys  # 保引用

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

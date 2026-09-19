# -*- coding: utf-8 -*-
"""
加页手记 · 摸金地图叠层工具（重构版）
======================================
热键(Ctrl+Shift+F) → 入口引索匹配定种子 → 两段式对齐 → 透明投影重合。

一次热键 = 一个任务 = 一个结果。自动跟随·第一步已落地：游戏内 G 开关小地图时
投影同步显隐（ui/follow.py 轮询 + 投影窗 WDA 截屏排除，见 CLAUDE.md）；其余跟随
状态机（连续再对齐/自动重匹配）仍推迟，待开合跟随实测可靠后再立项。废弃的整图
识别(find_seed_submap / find_seed_color / detect_direction)已删，全部走入口引索 + 两段式对齐。

双闸（见 CLAUDE.md）：
  入口置信闸 = 入口分 < score_confident(0.10) 且 overlap ≥ overlap_min(0.40)（种子ID可信）
  对齐显示闸 = 重合分 < align_score_max(0.30) 且 overlap ≥ 0.40（投影该显示）
"""
from __future__ import annotations

import csv
import html
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QRubberBand, QSlider, QVBoxLayout, QWidget,
)

from core.alignment import (
    affine_from_points, find_overlay_transform, map_to_overlay_rgba,
)
from core.entrance import (
    _crop_around_icon, sample_structure, build_entrance_transform, find_seed_by_entrance,
    load_index, score_desc,
)
from core.map_library import MapLibrary
from core.vision import (_find_icon, FOG_OPEN_MIN, NAV_NCC_MIN, detect_fog_panel, load_bgr,
                         map_is_open, map_open_from_roi, map_roi, panel_for_screen, roi_of,
                         zoom_scale_from_k, zoom_scale_from_roi)
from ui.capture import capture_monitor, capture_region, monitor_size
from ui.follow import (
    FOLLOW_IDLE_INTERVAL_MS, FOLLOW_INTERVAL_MS, FOLLOW_MAX_CAPTURE_ERRORS, FollowState,
)
from ui.hotkey import MOD_CONTROL, MOD_SHIFT, HotkeyManager, parse_hotkey
from ui.manage_materials import ManageMaterialsDialog
from ui.overlay import MapOverlay
from ui.preview import SamplePreview
from ui.settings import Settings, SettingsDialog, load as load_settings, save as save_settings
from ui.track import (TRACK_BEAT_MIN_SEC, TRACK_DEADBAND_S, TRACK_ICON_SCORE_MIN,
                      TRACK_IDLE_INTERVAL_MS, TRACK_LOG_MIN_SEC, TRACK_LOST_MAX,
                      TRACK_MOTION_MAD, TRACK_NEAR_R, TRACK_OK, TRACK_OVERLAP_MIN,
                      Tracker, ruler_slop, verdict)

# DPI 缩放适配：让进程用物理像素（per-monitor DPI aware + Qt 禁用 high-DPI scaling），
# 使 mss 截屏、Qt 窗口坐标、面板坐标(panel_for_screen) 三者统一于物理像素。否则 125%/150% 缩放下
# 截屏=逻辑像素而面板=物理坐标，错位致图标检偏、匹配退化(分0.000)、投影位置不对。
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
SAMPLE_MASK_MIN = float(_CFG["match"]["sample_mask_min"])
SAMPLE_LEAD_MIN = float(_CFG["match"].get("lead_min", 0.25))
DOMINANT_CLS_MAX = float(_CFG["match"].get("dominant_cls_max", 0.90))
# 扩大取样梯子（单类占比>DOMINANT_CLS_MAX 时依次尝试；只在当前档仍退化才降档）
SAMPLE_LADDER = (0.25, 0.32)
# 对齐失败居中兜底时，若样本单类占比仍高于此值 → 附「均匀区无锚点」说明
DOMINANT_NOTE_MIN = 0.75


def _dominant_frac(cls, mask):
    """mask 内 {1,2,3} 最大单类占比。>DOMINANT_CLS_MAX = 样本退化为单一均匀区。"""
    tot = max(1, int(mask.sum()))
    return max(float(((cls == c) & (mask > 0)).sum()) / tot for c in (1, 2, 3))


def _degenerate(res):
    """多种子同分退化（sc<0.001 且与次名分差<0.001），与 _after_match 退化闸同款。"""
    return len(res) >= 2 and res[0][0] < 0.001 and (res[1][0] - res[0][0]) < 0.001


def _lead_frac(res) -> float | None:
    """入口 top1 相对 top2 的领先幅度 ∈[0,1]（分越小越好 ⇒ (top2−top1)/top2）。

    只有一个结果 ⇒ 1.0（其余种子都弃权了，无次名可比；与 e34 口径一致）。无结果 ⇒ None。
    用途：结构闸的逃生门（见 config `[match].lead_min` 与 experiments/e34_evidence_gate.py）——
    「样本结构占比」只是"能不能判"的**代理**，而入口层判的是**墙**；用入口层自己的领先幅度更对症。
    """
    if not res:
        return None
    if len(res) < 2 or res[1][0] <= 1e-9:
        return 1.0
    return (res[1][0] - res[0][0]) / res[1][0]


# 入口判别力降序：侧门/二楼房间形状各异（主要判别依据）；正门固定分不出种子。
ENTRANCE_TYPES = ("侧门", "二楼", "正门")

# 状态行的**像素**预算（T2b）。主窗定宽 300 − 左右边距 16 − 圆点 10 − 间距 5 − 余量。
# 必须按像素截：一行放不下 56 个汉字（≈616px），旧的 `msg[:53]` 是按字数的，改成一行后
# 会在字中间被硬切（用户截图里「建议手动确」就是这么来的）。
STATUS_MAX_PX = 262

# 折叠后指引用户去手动纠错的统一前缀（T2b）。这几个按钮已不在主窗，文案里直接写
# 「手框样本/3点标定」会指向不存在的东西 —— 而日志窗默认关着，状态行只显示截断的一句话，
# 用户更没地方去找。改文案时别退回裸按钮名。
FIX_PATH = "⋯ 更多→手动纠错→"


class _ClickLabel(QLabel):
    """左键点击发 `clicked` 的 QLabel（状态行 / 「参考图」摘要行用）。

    QLabel 没有 clicked 信号，而为这一处去装 eventFilter 或改基类都不划算。
    """

    clicked = Signal()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


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
        # T3「换种子」的状态：入口层排名 + 本次会话的黑名单。
        # `_entrance_res` 是 `find_seed_by_entrance` 原样返回的排名 [(分,种子,key,楼层,s,mloc)...]；
        # `_entrance_et` 是产出它的入口类型（换种子后重跑对齐要用同一个入口取图标锚点）。
        self._entrance_res = None
        self._entrance_et = None
        self._seed_bl: set[int] = set()   # 只在「换种子」会话内累计；按热键清空（见 _entrance_pipeline_impl）
        # **实际试过的那个种子**（不是 `self._seed` —— 那是从下拉反解出来的）。
        # 换种子必须按这个拉黑：万一 `key→种子` 的反解与入口层排名里的种子对不上
        # （合成候选测试里就撞上过：key「北-1门」反解成种子10，而候选里是种子1），
        # 读 `self._seed` 会拉黑一个**不在候选表里**的号 ⇒ 候选表永不缩小 ⇒
        # 每次点击都重复试同一个候选、永远换不完。冒烟里有这条回归。
        self._tried_seed = None
        self.overlay: MapOverlay | None = None
        self.preview: SamplePreview | None = None  # 入口样本预览窗（阶段3）
        self._last_shot_path = None  # 最近截图保存路径（log/回收用）
        # 自动跟随·第一步（投影显隐跟随地图开合）：状态机 + 轮询定时器 + 面板缓存
        self._follow = FollowState()
        self._follow_timer: QTimer | None = None
        self._follow_panel = None
        self._follow_err = 0
        self._follow_poll_paused = False  # 手动隐藏导致的轮询暂停（一次性日志防刷屏）
        self._follow_idle = 0             # 自动隐藏期间的降频计数（每 N 格才真截一次）
        self._last_rgba = None            # 上次成功投影的整屏 RGBA（G 重开地图时原样放回）
        # 自动跟随·第二步（投影跟着地图平移/缩放，见 ui/track.py）：跟踪器 + 重对齐节流
        self._track = Tracker()
        self._track_arm = False           # 开态刚确立：跳一 tick 再开始跟踪（避开 G 开合动画过渡帧）
        self._track_idle = 0              # 投影没动时的降频计数
        self._track_log_t = 0.0           # 跟踪成功日志限流（拖动时别刷屏）
        self._track_beat_t = 0.0          # 跟踪**心跳**日志限流（每 tick 走哪条路，见 _track_beat）
        self._mad_peak = 0.0              # 上一次心跳以来 `帧差` 的**峰值**（心跳只记瞬时值会漏掉尖峰）
        self._track_n = 0                 # 上一次心跳以来 `_track_tick` 真跑了几次（分母）
        self._panel_prev = None           # 面板降采样上一帧（画面没动就整段跳过，省 ~70ms/次）
        self._track_err = 0               # 跟踪主体抛异常次数（见 _maybe_track 的兜底）
        self._screen_wh = None            # 屏幕尺寸缓存（避免每 tick 新建/销毁 mss DC 句柄）

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setWindowTitle("加页手记 地图助手")
        self.setFixedWidth(300)

        # 状态灯（阶段1）：**缩成状态行左边一个 10px 圆点**（T2b）。原来它是一整条
        # word-wrap 的文字（`● 已启动 · Ctrl+Shift+F 已注册` 能占两行），主窗被它顶掉一大截。
        # 颜色语义不变（绿=正常 / 红=失败），文案搬进 tooltip，点它 = 热键状态。
        self.led = QPushButton()
        self.led.setFixedSize(10, 10)
        self.led.setCursor(Qt.CursorShape.PointingHandCursor)
        self.led.clicked.connect(self._show_hotkey_status)
        self._led_text = "启动中…"
        # 运行日志区（阶段1）：**不再常驻主窗**，搬进懒创建的日志窗（T2b）。
        # ⚠️ 对象仍归 MainWindow —— `_log_step` 的写入目标一个字没改；日志窗关闭**不销毁**
        # `log_view`（QDialog 默认 close=hide），否则关着的那段日志全丢，而排查实机问题时
        # 那是唯一的东西。`self._log_window` 由 `_show_log_window` 惰性建。
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setStyleSheet("background:#1a1a1a; color:#ccc; font-size:11px;")
        self._log_window: QDialog | None = None
        self._ref_dialog: QDialog | None = None

        self.direction_combo = QComboBox(); self.direction_combo.addItems(lib.directions())
        self.door_combo = QComboBox()
        self.floor_combo = QComboBox(); self.floor_combo.addItems(["一楼", "二楼"])
        self.entrance_combo = QComboBox(); self.entrance_combo.addItems(list(ENTRANCE_TYPES))
        # 「参考图」摘要行（T2b）：主窗上唯一显示「当前选的是哪个种子」的地方，点开
        # → `_show_ref_picker()`（4 个下拉搬进对话框）。原独立的 `seed_label` 就是它。
        self.seed_label = _ClickLabel("参考图: …")
        self.seed_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.seed_label.setToolTip("点这里选 方向/门/楼层/入口（手动纠错用）")
        self.seed_label.clicked.connect(self._show_ref_picker)
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        self.door_combo.currentIndexChanged.connect(self._resolve_seed)
        # 楼层也接上：摘要行里带楼层，不然改了楼层这行就陈旧了（`_seed` 本身只由方向+门决定）
        self.floor_combo.currentIndexChanged.connect(self._resolve_seed)

        # ---- T2b 主窗 5 按钮（用户 09-19 定的清单，一个不折叠）----
        # 折叠进「⋯ 更多」的只有：手框样本 / 手动选点 / 参考图 / 日志 / 回收 / 素材 / 设置 / 退出。
        self.btn_onematch = QPushButton("🔴 一键匹配（=热键）")
        self.btn_onematch.setStyleSheet("font-weight:bold; padding:10px; background:#2a4a2a;")
        self.btn_onematch.clicked.connect(self._entrance_pipeline)
        self.btn_realign = QPushButton("▶ 按此种子重新对齐")
        self.btn_realign.setToolTip("当场截屏，用上面「参考图」选的那个种子重跑两段式对齐（=热键的同一条路）")
        self.btn_realign.clicked.connect(self._realign)
        self.btn_swap = QPushButton("🔄 换种子")
        self.btn_swap.setEnabled(False)   # 初始态；`_sync_swap_btn` 按入口层排名给可用性
        self.btn_hide = QPushButton("👁 隐藏地图")
        self.btn_options = QPushButton("⋯ 更多")
        self.btn_swap.clicked.connect(self._swap_seed)
        self.btn_hide.clicked.connect(self._hide_overlay)
        self.btn_options.clicked.connect(self._show_options_menu)
        # 「✕ 退出」不再需要控件：它已进「⋯ 更多」菜单，直接连 `self._quit`。

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(int(self.settings.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        # ---- T2b 主窗布局：≈190px（原 ≈520/700）。只有 5 个按钮 + 参考图行 + 状态行 + 滑块 ----
        root = QVBoxLayout(); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(5)
        root.addWidget(self.btn_onematch)
        row_align = QHBoxLayout(); row_align.setSpacing(5)
        row_align.addWidget(self.btn_realign, 3); row_align.addWidget(self.btn_swap, 2)
        root.addLayout(row_align)
        row_misc = QHBoxLayout(); row_misc.setSpacing(5)
        row_misc.addWidget(self.btn_hide, 1); row_misc.addWidget(self.btn_options, 1)
        root.addLayout(row_misc)
        root.addWidget(self.seed_label)
        # 状态行：**一行**，日志窗默认关着时它是主窗唯一的运行反馈 ⇒ 截断（按像素）+
        # 悬停看全文 + 按 level 染色 + 点击开日志窗。圆点在它左边。
        self.status = _ClickLabel("就绪。Ctrl+Shift+F 入口匹配（需先按 g 打开地图、刚进入口）。")
        self.status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status.setToolTip("点一下打开运行日志窗")
        self.status.clicked.connect(self._show_log_window)
        _sf = QFont(); _sf.setPixelSize(11)   # 与 status 的 `font-size:11px` 对齐
        self._status_fm = QFontMetrics(_sf)
        row_status = QHBoxLayout(); row_status.setSpacing(5)
        row_status.addWidget(self.led, 0, Qt.AlignmentFlag.AlignVCenter)
        row_status.addWidget(self.status, 1)
        root.addLayout(row_status)
        row_op = QHBoxLayout(); row_op.setSpacing(5)
        op_lab = QLabel("地图透明度"); op_lab.setStyleSheet("color:#aaa; font-size:11px;")
        row_op.addWidget(op_lab); row_op.addWidget(self.opacity_slider, 1)
        root.addLayout(row_op)
        self.setLayout(root)
        self.move(8, 60)
        self._sync_swap_btn()        # 初始禁用 + 说明"还没匹配过"
        self._on_direction_changed()
        if self.settings.auto_follow:
            self._start_follow()
        if self.settings.show_log:   # 语义已改为「启动时是否打开日志窗」（T2b）
            self._show_log_window()

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
        floor = self.floor_combo.currentText()
        self._seed = self.lib.find_by_clue(direction, door)
        # T2b：这行现在是主窗的「参考图」摘要（点开 = 方向/门/楼层/入口对话框），
        # 所以带上方向-门-楼层，让人不打开对话框也知道当前选的是哪张参考图。
        if self._seed is None:
            self.seed_label.setText(f"参考图: {direction}-{door} 未找到 ▾")
            self.seed_label.setStyleSheet("color:#ff6666; font-size:11px;")
        else:
            self.seed_label.setText(f"参考图: {direction}-{door} {floor} → 种子{self._seed} ▾")
            self.seed_label.setStyleSheet("color:#88dd88; font-size:11px;")

    def _current_map(self):
        if not getattr(self, "_seed", None):
            return None
        return self.lib.get(self._seed, self.floor_combo.currentText())

    def _select_ref(self, key: str, floor: str):
        """把「参考图」那套下拉切到 `key`（形如 `北-1沙发门`）+ `floor`，并刷新摘要行。

        热键命中（`_after_match`）与「🔄 换种子」共用 —— 两处都必须把主窗那行摘要
        一起带走，否则界面上还显示着被换掉/被覆盖的那个种子。
        """
        self.floor_combo.blockSignals(True)
        self.floor_combo.setCurrentText(floor)
        self.floor_combo.blockSignals(False)
        bdir, door = key.split("-", 1)
        self.direction_combo.setCurrentText(bdir)
        doors = [self.door_combo.itemText(i) for i in range(self.door_combo.count())]
        if door in doors:
            self.door_combo.setCurrentText(door)
        self._resolve_seed()   # 上面若因值相同没触发信号，这里兜一次底

    def _sync_swap_btn(self):
        """「🔄 换种子」可用性（T3）。

        有入口层排名就有候选可换 —— **包括投影被 `_refuse` 掉的时候**（种子已知、只是
        没过闸），那恰恰是最需要换种子的场合。候选被拉黑光了就置灰，等下次热键清空。
        """
        res = self._entrance_res or []
        left = [r for r in res if int(r[1]) not in self._seed_bl]
        self.btn_swap.setEnabled(bool(left))
        if not res:
            tip = "还没匹配过 —— 先按热键（或「🔴 一键匹配」）拿到入口层候选"
        elif left:
            tip = (f"入口层候选还剩 {len(left)}/{len(res)}（已排除 {sorted(self._seed_bl) or '无'}）\n"
                   "点一下 = 把当前种子拉黑 + 试下一个候选")
        else:
            tip = "没有别的候选了 —— 按热键重新匹配（会自动清空排除名单）"
        self.btn_swap.setToolTip(tip)

    # ---- 状态灯 / 运行日志（阶段1）----
    def set_led(self, ok: bool, text: str):
        """启动/热键成败状态灯。ok=True 绿，False 红。

        T2b：从「占一整行的文字条」缩成状态行左边一个 10px 圆点，文案搬进 tooltip。
        **调用点一个没删** —— 启动失败 / 热键被占用这类反馈全靠它，只是渲染方式变了。
        """
        self._led_text = text
        color = "#66ff66" if ok else "#ff6666"
        self.led.setStyleSheet(
            f"QPushButton{{border:none; border-radius:5px; background:{color};}}")
        self.led.setToolTip(f"● {text}\n（点一下看热键状态）")

    def _log_step(self, msg: str, level: str = "INFO"):
        """一步运行日志：append 进日志区 + 同步 status **那一行**。level: INFO/OK/WARN/ERROR。

        T2b：日志区已搬进日志窗且默认关着 ⇒ **status 那一行是主窗唯一的运行反馈**，
        所以这里做三件事补偿：
        ① 按**像素**截断（`STATUS_MAX_PX`）而不是按字数 —— 一行放不下 56 个汉字，
           旧的 `msg[:53]` 在单行布局下会在字中间硬切；
        ② 全文进 tooltip（悬停看全）；
        ③ 按 level 染色（黄/红 = 坏消息），日志窗关着时也能一眼看出这句是警告。
        """
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        color = {"ERROR": "#ff6666", "WARN": "#ffcc66", "OK": "#66ff66"}.get(level, "#cccccc")
        # msg 必须转义：appendHtml 把整串当 HTML 解析，消息里的 `<`（「导航列NCC-0.03(<0.45)」
        # 「入口领先仅15%(<0.25)」）会被当成标签开头，**把后面整段吃掉**（2026-09-18 bug②：
        # 日志里所有截断点都恰好紧跟一个 `<`）——正是诊断最需要看的那几条。
        self.log_view.appendHtml(f'<span style="color:{color}">[{ts}] {html.escape(msg)}</span>')
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._set_status(msg, level)

    def _set_status(self, msg: str, level: str = "INFO"):
        """写状态行**那一行**：按像素截断 + tooltip 全文 + 按 level 染色。

        ⚠️ 别再直接写 `self.status.setText(...)` —— 那是绕过截断/tooltip/配色的写法：
        单行布局下会硬切在字中间，而且 tooltip 还留着**上一条**的内容（hover 出来是错的）。
        全部走这里。
        """
        color = {"ERROR": "#ff6666", "WARN": "#ffcc66", "OK": "#66ff66"}.get(level, "#aaaaaa")
        self.status.setStyleSheet(f"color:{color}; font-size:11px;")
        self.status.setToolTip(msg)
        self.status.setText(self._status_fm.elidedText(
            msg, Qt.TextElideMode.ElideRight, STATUS_MAX_PX))

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
        # T3：按热键（=「🔴 一键匹配」）就是"重新自动来一次" ⇒ 清空换种子黑名单。
        # 用户原话：「按"自动匹配"时就把黑名单重置防止匹配不到种子」—— 不清的话按着按着
        # 候选就被自己拉黑光了，反而匹配不到。
        self._seed_bl.clear()
        if not self._capture():
            return
        self._log_step("截屏 OK")
        shot = self._shot
        panel = detect_fog_panel(shot)
        if panel is None:
            self._log_step(f"屏幕 {shot.shape[1]}×{shot.shape[0]} 未适配（非16:9且未校准；"
                           "config.toml [panel.rects] 可加校准）", "ERROR")
            return
        # 地图画面闸（CLAUDE.md 待办 3）：大厅/结算/桌面等非地图画面按下热键，跑完入口匹配
        # 只会给出一个**无意义却看着挺像**的种子。判据与跟随同源（导航列 NCC 或 雾兜底），
        # 见 core/vision.py 的 NAV_BAND 长注。此处只记日志、不动投影（跟随自己会隐）。
        opened, why = map_is_open(shot, panel)
        if not opened:
            self._log_step(f"不是地图画面（{why}）→ 先按 g 打开游戏地图、刚进入口再按热键", "WARN")
            return
        et = self.entrance_combo.currentText()
        res, icon_pos, isc, icon_k = find_seed_by_entrance(shot, self.lib, et, panel=panel, top_n=3)
        if icon_pos is None:
            self._log_step(f"入口图标未检出(分{isc:.2f}) → {FIX_PATH}手框样本/3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=False)
            return
        self._log_step(f"入口图标 @({icon_pos[0]},{icon_pos[1]}) 分{isc:.2f} k={icon_k}")
        sample = _crop_around_icon(shot, panel, icon_pos, self.settings.sample_half_frac,
                                   icon_k=icon_k)
        # 快速失败闸：样本结构太少=入口周围未探明/迷雾占屏，跑匹配只会出
        # 多种子同分0.000的误导结果（实测坏样本mask≤15.6%、好样本≥24.7%，见 config）
        # **逃生门（2026-09-18）**：结构占比只是"能不能判"的**代理**，而入口层判的是**墙**
        # （被拒样本里仍含 581~1348 个墙像素）。入口匹配已经跑完了(res) ⇒ 直接用**入口层
        # 自己的领先幅度**判：top1 领先次名 ≥ `lead_min` 就放行。
        # 依据 experiments/e34_evidence_gate.py（44 张主集）：领先≥0.25 时**出图且对 14→21
        # （+7）、出图但错仍是 3 张没多**；领先≥0.15 就开始崩（错 3→10）。
        _cls, smask = sample_structure(sample)
        if smask.mean() < SAMPLE_MASK_MIN:
            lead = _lead_frac(res)
            if lead is None or lead < SAMPLE_LEAD_MIN:
                why_lead = ("入口无结果" if lead is None
                            else f"入口领先仅{lead:.0%}")
                self._log_step(f"样本结构仅{smask.mean() * 100:.0f}%（入口周围未探明）且{why_lead}"
                               f"（<{SAMPLE_LEAD_MIN:.0%}）→ 请在刚进入口、周围已探明时再按", "WARN")
                self._log(et, res, icon_pos, isc, None, corrected=False)
                return
            self._log_step(f"样本结构仅{smask.mean() * 100:.0f}%，但入口 top1 领先次名 {lead:.0%}"
                           f"（≥{SAMPLE_LEAD_MIN:.0%}）→ 放行", "INFO")
        self._show_sample_preview(sample)  # 让玩家看到匹配用的样本
        dom = _dominant_frac(_cls, smask)
        # 扩大取样梯子：单类占比过高 = 入口周围是单一均匀区（大片走廊/雾），分类
        # SQDIFF「样本覆盖处类别全等」⇒ 多种子精确同分（2026-09-14 实测 98% cls3 →
        # 种子2/18 同分 0.000）。扩大取样纳入更多结构可拉开分差（145440@0.25 →
        # 种子2 0.0069 vs 次名 0.045）。只在「当前档仍退化」时降档、首个非退化即停
        # ——0.32 档可能把并列翻成错误种子的假干净（实测 144715@0.32 翻成种子18）。
        if dom > DOMINANT_CLS_MAX:
            retried = False
            for hf in SAMPLE_LADDER:
                if res and not _degenerate(res):
                    break
                sample2 = _crop_around_icon(shot, panel, icon_pos, hf, icon_k=icon_k)
                _cls2, smask2 = sample_structure(sample2)
                if smask2.mean() < SAMPLE_MASK_MIN:
                    continue  # 更大的框反而更空（罕见）：跳过该档
                res2, _ip, _isc, _ik = find_seed_by_entrance(
                    shot, self.lib, et, panel=panel, top_n=3, sample_crop=sample2)
                if not res2:
                    continue
                sample, res, _cls, smask, dom = (sample2, res2, _cls2, smask2,
                                                 _dominant_frac(_cls2, smask2))
                retried = True
            self._show_sample_preview(sample)
            if _degenerate(res):
                self._log_step(
                    f"样本单类占比{dom * 100:.0f}%（大片均匀走廊/未探明），扩大取样后仍无判别结构 → "
                    f"多探明周围结构后再按 / {FIX_PATH}手框含墙角结构的小块 / 3点标定", "WARN")
                self._log(et, res, icon_pos, isc, None, corrected=False)
                return
            if retried:
                self._log_step(f"扩大取样重试({sample.shape[1]}px)：单类降至{dom * 100:.0f}%，分差已拉开", "INFO")
        if not res:
            self._log_step(f"入口匹配无结果 → {FIX_PATH}手框样本/3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=False)
            return
        best = res[0]
        self._log_step(f"入口匹配: 种子{best[1]}({best[2]}[{best[3]}]) {score_desc(best[0])} top3="
                       + str([(r[1], score_desc(r[0])) for r in res[:3]]), "OK")
        self._after_match(shot, best, et, icon_pos, isc, res, corrected=False, dom_frac=dom)

    def _after_match(self, shot, best, et, icon_pos, isc, res, corrected=False, dom_frac=None):
        """拿到 best 后：填 UI + 两段式对齐 + 投影 + 归档。供热键/手框样本复用。

        dom_frac: 匹配样本的单类占比（>DOMINANT_NOTE_MIN 且对齐失败时，居中兜底附
        「均匀区无锚点」说明——种子ID可信但面板没有可对齐的结构，非对齐算法失灵。"""
        sc, seed, key, fl, _s, _mloc = best
        # T3：把入口层排名与产出它的入口类型存下来 —— 「🔄 换种子」靠它取下一个候选。
        # 必须在**所有**早退分支之前存：退化/哨兵分那两类虽然没投影，但用户正需要换种子。
        self._entrance_res = res
        self._entrance_et = et
        self._tried_seed = int(seed)
        self._sync_swap_btn()
        corr_note = ("（面板为大段均匀区，无锚点可对齐——种子ID可信，地图居中仅供参考；"
                     "精确重合请3点标定）" if (dom_frac is not None and dom_frac > DOMINANT_NOTE_MIN) else "")
        # 退化防御：top1 与 top2 分差<0.001（多种子同分0.000）→ 样本无判别结构/图标假阳，不假阳报告
        if _degenerate(res):
            self._log_step(f"匹配退化(多种子同分 {sc:.3f}) → 样本无判别结构（均匀区/假阳图标），"
                           f"请{FIX_PATH}手框含墙角结构的小块或3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=corrected)
            return
        if sc >= 1e8:  # 哨兵分兜底（find_seed_by_entrance 已跳过无尺度种子，此处防御）
            self._log_step("匹配无有效尺度（样本过大/引索缺）→ 框小一点(入口局部结构)或3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=corrected)
            return
        self._select_ref(key, fl)

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
                self._seed_track(shot, seed, fl, info.path, align[0])
                self._log_step("投影已显示", "OK")
            else:
                self._refuse(f"种子{seed}({key}[{fl}])", f"引索里没有该种子的 {fl} 参考图")
        else:
            why = (f"对齐没过显示闸(重合{asc:.3f} 重叠{ov:.2f})" if align is not None
                   else "对齐失败(探明不足)")
            self._refuse(f"种子{seed}({key}[{fl}])", why + corr_note)

        confident = (sc < SCORE_CONFIDENT and ov is not None and ov >= OVERLAP_MIN)
        self._log(et, res, icon_pos, isc, align, corrected=corrected)
        ov_txt = f"{ov:.2f}" if ov is not None else "-"
        self._set_status(
            f"入口{et}(图标{isc:.2f}) {score_desc(sc)} 重叠{ov_txt} → 种子{seed}({key}[{fl}])"
            + (" ✓确信已对齐" if (confident and show_ok)
               else (" 种子可信但未对齐（未投影）" if confident
                     else " 不确信，可手动改门/3点标定")))

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
            self._set_status("屏幕分辨率未适配（非16:9且未校准）——3点标定仍可用")
            return None
        # 第一段：构造 M1 + 对齐尺度提示 hint_s（=入口匹配尺度 s）+ 图标锚点 hint_icon
        # （锚点约束第二段平移搜索——防自相似迷宫幽灵相位，见 find_overlay_transform 注）
        hint_s = None
        hint_icon = None
        idx = load_index(seed)
        if idx is not None:
            entry = idx.get(entrance_type)
            if entry is not None:
                hint_icon = (entry["cx"], entry["cy"], icon_pos[0], icon_pos[1])
            m1 = build_entrance_transform(best, entrance_type, idx, icon_pos, panel)
            if m1 is not None:
                _M1, hint_s = m1
        # 第二段：hint 精修；过闸即用
        if hint_s is not None or hint_icon is not None:
            align = find_overlay_transform(shot, ref, panel, hint_s=hint_s,
                                           hint_icon=hint_icon)
            if align is not None:
                _M, asc, ov = align
                if asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN:
                    return align
        # 回退：全搜（baseline verify_entrance_e2e 证 17.13/17.11 全搜可靠）
        return find_overlay_transform(shot, ref, panel)

    def _realign_hints(self, shot, panel) -> tuple:
        """「按此种子对齐」的 hint：与热键路径同源 —— `hint_s` 尺度提示 + `hint_icon` 图标锚点。

        `hint_s` 取**缩放滑条读数**（`zoom_scale_from_roi`，±0.01，全量程已标定）而不是
        「挑分最小的尺度」：分对尺度是单调偏低的（技术备忘⑥），挑分必然挑到更小的那个。
        读不到滑条（窗口化布局，导航列 NCC 掉到 0.34 < 闸 0.45）时回退图标尺子 `0.374/k`。
        """
        s_ui, _why = zoom_scale_from_roi(roi_of(shot, panel), panel)
        icon_pos, isc, k = _find_icon(shot, *panel)
        if s_ui is None and icon_pos is not None:
            s_k = zoom_scale_from_k(k) if (isc or 0) >= TRACK_ICON_SCORE_MIN else None
            s_ui = s_k
        return s_ui, icon_pos

    def _align_to(self, seed, floor: str, et: str, label: str) -> bool:
        """**当场截屏** → 用指定种子走两段式对齐 → 过闸投影。返回是否投影成功。

        `_realign`（用户手选种子）与 `_swap_seed`（换种子候选）共用这一条路 ——
        保证「按此种子重新对齐」和「换种子」用的是**同一个对齐方式**（T1.5 的教训：
        用户原话「把匹配时的对齐方式用在其他跟随和对齐的时候」）。

        2026-09-19 修的两处（用户报「点『按此种子对齐』按钮对齐也是不准的」）都在这条路上：

        ① **必须当场重新截屏**。旧写法 `if self._shot is None and not self._capture()`
           只在从未截过屏时抓一张 ⇒ 按钮对齐的其实是**上一次热键那张旧图**，投影于是落在
           「地图当时所在」的位置。实机日志的指纹很干净：`手动重对齐: 南-三缺一门
           重合0.05 重叠1.00 已投影` 在 4 分钟里出现 **8 次**（15:40:00/06/14/18、
           15:42:43/54/57/59），**分数与重叠一字不差** —— 中间地图已挪过好几处，只有
           「同一张旧图 + 同一个变换」才会给出逐位相同的结果。

        ② **把热键路径的对齐方式搬过来**。旧写法是无锚点无尺度提示的全搜，而全搜在**这一帧**
           上就给出骗人的答案：实测 `captures/hotkey_20260919_154124.png` 全搜得
           s≈0.39 / 分0.038 / 重叠1.00（**过闸**），真值是 s≈0.83（图标尺子 0.374/k，k=0.45）
           —— 技术备忘① 的「缩模板骗分」在尺度轴上重演：模板缩小 ⇒ 不一致像素被一起缩掉
           ⇒ 分更低。热键路径靠 `hint_s`（入口匹配尺度）+ `hint_icon`（图标钉死平移）双重
           约束才稳，这里同样给（见 `_realign_hints`）。两者缺失只是退化成「少一层约束」；
           都不过对齐显示闸就**回退全搜**，等于旧行为，不会更差。
        """
        info = self.lib.get(seed, floor)
        if info is None:
            self._set_status(f"引索里没有 种子{seed} 的 {floor} 参考图"); return False
        if not self._capture():
            return False
        shot = self._shot
        panel = detect_fog_panel(shot)
        if panel is None:
            self._set_status("屏幕分辨率未适配（非16:9且未校准）"); return False
        # 复用热键路径的 `_two_stage_align`（hint 精修 → 过闸即用 → 回退全搜），零重复。
        # `best` 是入口层结果的元组 (score, seed, key, floor, s, mloc)：这里没有入口层匹配
        # （种子是手选/换来的），就把**滑条/尺子量出来的尺度**放进 `s` 槽 —— 它在这一段的
        # 唯一用途就是当 `hint_s`（见 build_entrance_transform），语义正好对得上。
        hint_s, icon_pos = self._realign_hints(shot, panel)
        best = (0.0, seed, info.key, info.floor, hint_s, None)
        align = self._two_stage_align(shot, best, et, icon_pos)
        if align is None:
            self._set_status("对齐失败：探明不足"); return False
        M, asc, ov = align
        if asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN:
            rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size(),
                                       wall_alpha=self.settings.wall_alpha)
            self._show_overlay(rgba)
            self._seed_track(shot, seed, info.floor, info.path, M)
            self._set_status(f"已对齐(匹配{asc:.2f} 重叠{ov:.2f})：{info.key}")
            self._log_step(f"{label}: {info.key} 重合{asc:.2f} 重叠{ov:.2f} 已投影", "OK")
            return True
        self._refuse(info.key, f"{label}没过闸(重合{asc:.2f} 重叠{ov:.2f})")
        self._set_status(f"未投影：对齐不可靠(匹配{asc:.2f} 重叠{ov:.2f})｜{info.key}")
        return False

    def _realign(self):
        """「▶ 按此种子重新对齐」：用**主窗「参考图」行选的那个种子**重跑对齐。"""
        info = self._current_map()
        if info is None:
            self._set_status("请先选好方向+门"); return
        self._align_to(self._seed, info.floor, self.entrance_combo.currentText(), "手动重对齐")

    def _swap_seed(self):
        """「🔄 换种子」：把当前种子拉黑，用入口层排名里的下一个候选取代，重跑对齐。

        **点一下 = 拉黑当前 + 试下一个**（用户 2026-09-19 定的触发模型）。点这个按钮的
        理由本身就是"这个种子不对"，若改成"等对齐失败才拉黑"，用户得点两次才见效。

        候选 = `find_seed_by_entrance` 的入口层排名（`self._entrance_res`，热键路径存下的）。
        入口层 top-1 只有 ≈48%，已知稳定失败场景就是「北-1沙发门(23) 被判成 北-1门(10)」
        —— 这时排名里通常紧跟着正确的那一个。

        黑名单只在本会话内累计，**按热键清空**（见 `_entrance_pipeline_impl`）。
        """
        res = self._entrance_res or []
        if not res:
            self._set_status("还没有入口层排名 —— 先按热键匹配一次"); return
        # 拉黑「上一次实际试过的种子」，**不是** `self._seed`（下拉反解）—— 见 `_tried_seed` 的长注。
        bl_target = self._tried_seed if self._tried_seed is not None else self._seed
        if bl_target is not None:
            self._seed_bl.add(int(bl_target))
        cand = [r for r in res if int(r[1]) not in self._seed_bl]
        if not cand:
            self._sync_swap_btn()
            self._set_status(f"没有别的候选了（入口 top{len(res)} 已被排除光）—— "
                             "按热键重新匹配（会清空排除名单）")
            self._log_step(f"换种子: 候选耗尽 top{len(res)}={[int(r[1]) for r in res]} "
                           f"已排除={sorted(self._seed_bl)}", "WARN")
            return
        sc, seed, key, fl, _s, _mloc = cand[0]
        self._tried_seed = int(seed)
        self._log_step(f"换种子: 种子{bl_target}→种子{seed}（{key}[{fl}]，入口分{sc:.2f}，"
                       f"已排除={sorted(self._seed_bl)}）", "INFO")
        self._select_ref(key, fl)          # 摘要行 + 下拉一起带走，否则界面还显示被换掉的种子
        self._sync_swap_btn()
        self._align_to(seed, fl, self._entrance_et or self.entrance_combo.currentText(), "换种子")


    def _three_point_calib(self):
        """手动兜底：3 点标定（affine_from_points，3 下点击必对）。

        在截图上点 3 个特征点，再在参考图上点对应 3 点，构造 ref→screen 仿射变换投影。
        """
        if self._shot is None and not self._capture():
            return
        info = self._current_map()
        if info is None:
            self._set_status("请先选好方向+门（确定参考图）"); return
        ref = load_bgr(str(info.path))
        pk1 = ClickPicker(self._shot, 3, "点 3 个游戏内特征点（顺序自定）", parent=self)
        if not pk1.exec():
            return
        pk2 = ClickPicker(ref, 3, "点参考图上对应的 3 点（同顺序）", parent=self)
        if not pk2.exec() or len(pk2.pts) < 3:
            return
        M = affine_from_points(pk2.pts, pk1.pts)  # src=ref, dst=screen
        if M is None:
            self._set_status("3点标定失败（点不足）"); return
        rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size(), wall_alpha=self.settings.wall_alpha)
        self._show_overlay(rgba)
        # 3 点标定走的是 affine_from_points —— **可能带旋转**，参数化与跟踪用的相似变换
        # 不同（跟踪只认 s/tx/ty），喂进去会算错。故显式断根，该投影不参与跟随重对齐。
        self._track.reset()
        self._set_status(f"3点标定投影：{info.key}")

    def _manual_sample_pick(self):
        """手动框选入口样本：在最近截图上拖矩形，作为 sample 重跑匹配（纠错）。"""
        if self._shot is None and not self._capture():
            return
        pk = ClickPicker(self._shot, 1, "拖框选入口样本区域", parent=self, mode="rect")
        if not pk.exec() or not pk.rect:
            self._log_step("手框取消", "INFO"); return
        x0, y0, x1, y1 = pk.rect
        sample = self._shot[y0:y1, x0:x1].copy()
        # 尺度硬上限：SCALES_ENT 最小 0.35，样本边 × s 须 ≤ 引索裁图 228px → 228/0.35≈650
        # 超限全尺度放不下 → 各种子直接跳过、res 恒空（2026-09-14 实测手框 408px「仍无匹配」即此因，
        # 380px 旧上限是 0.6 下限时代所设）。
        if max(sample.shape[:2]) > 650:
            self._log_step(f"手框 {x1-x0}x{y1-y0} 超过650px（引索尺度下限 0.35×228）→ "
                           "请框含墙角/房间边缘结构的区域", "WARN")
            return
        self._show_sample_preview(sample)
        et = self.entrance_combo.currentText()
        self._log_step(f"手框样本 {x1-x0}x{y1-y0}，重跑匹配", "INFO")
        panel = detect_fog_panel(self._shot)
        if panel is None:
            self._log_step("屏幕分辨率未适配（非16:9且未校准）", "ERROR"); return
        res, icon_pos, isc, _ik = find_seed_by_entrance(
            self._shot, self.lib, et, panel=panel, top_n=3, sample_crop=sample,
            sample_origin=(x0, y0))
        if not res:
            self._log_step(f"手框样本仍无匹配 → 换一处含墙角/房间边缘的区域再框，或用{FIX_PATH}3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=True)
            return
        best = res[0]
        self._log_step(f"手框匹配: 种子{best[1]}({best[2]}[{best[3]}]) 分{best[0]:.3f}", "OK")
        _cls, smask = sample_structure(sample)
        self._after_match(self._shot, best, et, icon_pos, isc, res, corrected=True,
                          dom_frac=_dominant_frac(_cls, smask))

    # ---- 投影显示 ----
    def _screen_size(self):
        # 用 mss 主显示器物理尺寸（DPI aware 后=物理像素），与截屏/面板坐标一致；
        # 不用 Qt primaryScreen.geometry()（DPI 缩放下可能返回逻辑像素致错位）。
        # **缓存在 self._screen_wh**：跟随跟踪每 tick 都要烘焙投影，若每次现开 mss
        # 就是 2026-09-14 那个「GDI DC 开合 4 次/秒毁英伟达截图」的坑（见 ui/capture）。
        if self._screen_wh is None:
            self._screen_wh = monitor_size(1)
        return self._screen_wh

    def _paint_overlay(self, rgba):
        """把整屏 RGBA 贴到投影窗（建窗 / 设透明度 / 显示）——**不含跟随状态机副作用**。

        `_last_rgba` 在这里更新（而不是各调用方各自赋值）：它只有一个含义「当前屏幕上那张
        投影」，任何一次贴图都该刷新它，否则「G 重开地图」放回的会是更旧的一张。
        """
        self._last_rgba = rgba
        if self.overlay is None:
            sw, sh = self._screen_size()
            self.overlay = MapOverlay(QRect(0, 0, sw, sh))
            if not self.overlay.capture_excluded:
                self._log_step("投影未能排除截屏（跟随/重匹配可能受自污染）", "WARN")
        self.overlay.set_image(rgba)
        self.overlay.setWindowOpacity(self.opacity_slider.value() / 100.0)
        self._set_overlay_visible(True)

    def _show_overlay(self, rgba):
        """新投影（热键 / 手动重对齐 / 3点标定）——会重置跟随状态机。"""
        self._paint_overlay(rgba)
        self._follow.reset()  # 新投影=「我现在要投影」：清跟随挂起与确认态，重新 adopt

    def _update_overlay(self, rgba):
        """跟随跟踪刷新投影——与 `_show_overlay` 的唯一区别：**不碰跟随状态机**。

        走 `_show_overlay` 会把 `_follow` 重置掉（那是"新投影"的语义），而跟踪是**同一个**
        投影在挪位置，重置会让开合状态机每 tick 重新 adopt。
        """
        self._paint_overlay(rgba)

    def _seed_track(self, shot, seed, floor, ref_path, M):
        """把刚显示出来的投影登记给跟踪器（跟随要靠它做局部重对齐）。"""
        panel = detect_fog_panel(shot)
        if panel is None or M is None:
            self._track.reset()
            return
        self._track.seed_from(panel, seed, floor, str(ref_path), M, 1.0 / float(M[0, 0]))
        self._track_idle = 0


    def _restore_overlay(self) -> bool:
        """地图被 G 重新打开 ⇒ 直接用**上次渲染好的投影**恢复显示，不重跑匹配。

        投影早已按上次的变换烘焙成整屏 RGBA，原样放回即与关图前逐像素一致 —— 比重新跑
        一遍入口匹配+两段式对齐快得多，也不会因为门口探明没变而白跑一遍。
        ⚠️ 关图期间地图若平移/缩放过，恢复的就是**旧位置**：要更新请按热键重匹配。
        上次没成功投影过（被判过拒答/藏图）⇒ 不恢复，只提示按热键（宁可什么都不给）。
        """
        if self._last_rgba is None:
            self._log_step("地图已开，但上次没有成功投影 ⇒ 不自动恢复（请按热键匹配）", "WARN")
            return False
        self._show_overlay(self._last_rgba)
        self._log_step("地图已开 → 投影已恢复（沿用上次的变换；地图若已移动请按热键重匹配）", "OK")
        return True

    def _show_sample_preview(self, bgr):
        """显示匹配用的入口样本（让玩家看到选对没），贴主窗右侧。

        自动消失秒数每次从 settings 现读（T2a）：预览窗弹完默认 5s 自己消失，防它一直
        挂在屏幕上挡画面。0 = 不自动消失（旧行为）。
        """
        if self.preview is None:
            self.preview = SamplePreview()
            g = self.geometry()
            self.preview.move(g.right() + 8, g.top())
        self.preview.set_sample(bgr, autohide_sec=self.settings.sample_preview_sec)
        self.preview.show()

    def _refuse(self, label, why):
        """不确信 / 对齐没过闸 ⇒ **一张图都不显示**（待办 1，2026-09-16 晚）。

        旧行为是 `_show_centered_if_any` → `auto_align_overlay` 把参考图按
        `min(pw/cw, ph/ch)` **缩放铺满迷雾面板**：比例与游戏内毫无关系，却画得工整、
        结构清楚，看起来就像一张"已经对齐好的地图"—— 用户报的「乱给一张素材、
        比例都不对」就是它，不是对齐结果。宁可什么都不给，也不给一张像是对的的假图：
        假图会让人以为算法定位到了别处，比空白有害得多。

        种子ID仍写进状态栏与日志（那个数是有意义的）；想看参考图请显式用主窗
        「▶ 按此种子重新对齐」，或「⋯ 更多→手动纠错→手动选点重合」（3 点标定）。
        """
        self._set_overlay_visible(False)
        self._last_rgba = None  # 清掉：跟随「G 重开地图」不许把上一张被否掉的投影放回来
        self._track.reset()     # 跟踪也要断根：没有可信投影可跟
        self._log_step(f"不投影：{why}｜入口判定 {label}（位置未验证，仅供参考）", "WARN")

    def _hide_overlay(self):
        if self.overlay is None:
            return
        visible = not self.overlay.isVisible()
        if self.settings.auto_follow:
            if visible:
                self._follow.resume()
            else:
                self._follow.suspend()
        self._set_overlay_visible(visible)
        if visible:
            self._log_step("地图已显示（跟随已恢复）")
        else:
            self._log_step("地图已隐藏（跟随暂停：G 不再自动唤起；点此按钮或按热键重新匹配可恢复）")

    def _set_overlay_visible(self, visible: bool):
        """投影显隐统一入口：show/hide + btn_hide 文案（_show_overlay/_hide_overlay/跟随共用）。"""
        if self.overlay is None:
            return
        if visible:
            self.overlay.show()
            self.btn_hide.setText("👁 隐藏地图")
        else:
            self.overlay.hide()
            self.btn_hide.setText("👁 显示地图")

    def _set_opacity(self, v):
        if self.overlay is not None:
            self.overlay.setWindowOpacity(v / 100.0)

    # ---- 自动跟随·第一步：投影显隐跟随游戏内地图开合 ----
    def _start_follow(self):
        """启动跟随轮询（分辨率未适配则拒绝并提示）。"""
        if self._follow_timer is not None and self._follow_timer.isActive():
            return
        panel = panel_for_screen(*self._screen_size())
        if panel is None:
            self._log_step("分辨率未适配，自动跟随不可用（非16:9，可config校准）", "ERROR")
            return
        self._follow_panel = panel
        self._follow.reset()
        self._follow_err = 0
        if self._follow_timer is None:
            self._follow_timer = QTimer(self)
            self._follow_timer.setInterval(FOLLOW_INTERVAL_MS)
            self._follow_timer.timeout.connect(self._follow_tick)
        self._follow_timer.start()
        self._log_step(f"自动跟随已开启：面板 {tuple(panel)} @ {FOLLOW_INTERVAL_MS}ms（地图关后降频到 "
                       f"{FOLLOW_IDLE_INTERVAL_MS}ms 继续等 G 开回来，手动隐藏才停），"
                       f"开闸: 导航列≥{NAV_NCC_MIN} 或 雾≥{FOG_OPEN_MIN}", "OK")
        self._log_step("跟随第二步：投影已按热键出来后会跟着地图平移/缩放（近处小窗→单尺度重对齐；"
                       "连丢两次则隐藏投影，提示重按热键）")

    def _stop_follow(self):
        if self._follow_timer is not None and self._follow_timer.isActive():
            self._follow_timer.stop()
            self._log_step("自动跟随已关闭")
        self._follow.reset()
        self._panel_prev = None

    def _follow_tick(self):
        if not self.settings.auto_follow or self._follow_panel is None:
            return
        # 模态对话框/QMenu 是嵌套事件循环，QTimer 照常触发——期间不判定
        if QApplication.activeModalWidget() or QApplication.activePopupWidget():
            return
        if self.overlay is None:
            self._follow.idle_reset()
            return
        # 两种「投影不在」必须分开（2026-09-17，用户报「关了按 G 也不会再开」）：
        #   手动隐藏 = 明确的用户意图 ⇒ 停一切屏幕检测；
        #   跟随判出地图关 = 只是「现在看不到」⇒ 降频续看，等 G 把地图开回来。
        # 旧版两者都早退 ⇒ 关一次地图之后 G 再也唤不回投影，只能点按钮或按热键重匹配。
        visible = self.overlay.isVisible()
        if not visible and self._follow.suspended:
            if not self._follow_poll_paused:
                self._follow_poll_paused = True
                self._log_step("投影已手动隐藏，跟随检测暂停（点『显示地图』或按热键恢复）")
            self._follow.idle_reset()
            return
        if visible:
            self._follow_poll_paused = False
            self._follow_idle = 0
        else:
            self._follow_idle += 1
            if self._follow_idle % max(1, FOLLOW_IDLE_INTERVAL_MS // FOLLOW_INTERVAL_MS):
                return
        try:
            # 截「面板 ∪ 右侧导航列」——一次截屏同时够算导航列 NCC 与面板雾特征
            rx0, ry0, rx1, ry1 = map_roi(self._follow_panel)
            roi = capture_region(rx0, ry0, rx1 - rx0, ry1 - ry0)
        except Exception as e:  # noqa: BLE001
            self._follow_err += 1
            if self._follow_err == 1:
                self._log_step(f"跟随截屏异常: {e}", "WARN")
            elif self._follow_err >= FOLLOW_MAX_CAPTURE_ERRORS:
                self._log_step(f"跟随截屏连续失败 {self._follow_err} 次，停止跟随", "ERROR")
                self._stop_follow()
            return
        self._follow_err = 0
        # 判据：侧栏导航列 NCC（主）或 雾占比（兜底）——结构占比已退出判定，见 vision.NAV_BAND
        is_open, why, nav_ok = map_open_from_roi(roi, self._follow_panel)
        evt = self._follow.feed(is_open)
        if evt is None:
            # 稳定态：地图开着 + 投影显示 ⇒ 让它跟着地图平移/缩放（自动跟随·第二步）
            if visible and self._follow.confirmed is True:
                self._maybe_track(roi, is_open, nav_ok)
            elif visible:
                # 心跳（§4 第 3 条静默路径）：投影显示着却没进跟踪链 —— `confirmed` 每次热键
                # 出图都被 `_show_overlay` 的 `reset()` 清成 None，要连续 2 帧同判才重立。
                self._track_beat(f"投影显示但不开跟踪（confirmed={self._follow.confirmed} "
                                 f"track_active={self._track.active}）")
            return  # 静默：稳定态/防抖中不刷日志
        _kind, state = evt
        if self._follow.suspended:
            self._track_arm = False
            self._log_step(f"跟随暂停中（地图{'开' if state else '关'}），不动投影")
            return
        restored = True
        if state and not self.overlay.isVisible():
            restored = self._restore_overlay()   # G 把地图开回来 ⇒ 放回上次的投影
        elif not state and self.overlay.isVisible():
            self._set_overlay_visible(False)
        # 开合刚翻转 ⇒ 跳一 tick 再跟踪：G 开合有动画，过渡帧的面板不是稳态
        self._track_arm = True
        self._track_idle = 0
        if restored:
            self._log_step(f"跟随: 地图{'开' if state else '关'} → "
                           f"投影{'显示' if state else '隐藏'}（{why}）")

    # ---- 自动跟随·第二步：投影跟着地图平移/缩放（设计见 ui/track.py 模块头）----
    def _maybe_track(self, roi, is_open, nav_ok=True):
        """跟踪节流：画面没动就整段跳过；静止时降频，动了回全速。

        `is_open` = 本 tick 的**原始**开合读数（未过防抖）。它可能比 `_follow.confirmed`
        更早看到「G 关了地图」——跟踪要的正是这个更早的信号，见 `_track_tick`。
        `nav_ok` = 这个「开」是导航列主判据说的（不是雾兜底说的），一路传到 `_track_lost`：
        只有主判据说开，判丢才允许销毁 `_last_rgba`。
        """
        if self._track_arm:
            self._track_arm = False     # 开态刚确立，本 tick 只做准备
            self._track_beat("开态刚确立 → 本 tick 只做准备（不跟踪）", force=True)
            return
        if not self._track.active:
            self._track_beat("跟踪器未登记（没有可信投影）→ 不跟踪", force=True)
            return
        if self._track_idle:
            self._track_idle += 1
            step = max(1, TRACK_IDLE_INTERVAL_MS // FOLLOW_INTERVAL_MS)
            if self._track_idle % step:
                return
        try:
            self._track_tick(roi, is_open, nav_ok)
        except Exception as e:  # noqa: BLE001
            # 跟踪主体**必须自己报错**，不许静默死掉。
            # 2026-09-19 查出的实机病根就是这个：`ui.track` 的 `TRACK_DEADBAND_S` /
            # `TRACK_OVERLAP_MIN` 与 `ui.follow` 的 `FOLLOW_MAX_CAPTURE_ERRORS` **没进
            # app.py 的 import 列表** ⇒ 自 `1b888e7`（跟随·第二步）起每次过闸都在写第一行
            # 日志之前抛 `NameError`，traceback 只进 stderr（用户看不到），心跳里只剩
            # 「帧差…画面没动」⇒ 实机 56 次过闸、零条输出。技术备忘③ 同款教训：
            # **走不到那条分支**等于没有记录，先让失败可见。
            self._track_err += 1
            if self._track_err <= 3 or self._track_err % 50 == 0:
                self._log_step(f"跟踪内部异常 #{self._track_err}"
                               f"（{type(e).__name__}: {e}）→ 本 tick 丢弃，跟随继续", "ERROR")

    def _track_beat(self, msg: str, force: bool = False):
        """跟踪**心跳**：把「这一 tick 走了哪条路」写进日志（限流 `TRACK_BEAT_MIN_SEC`）。

        2026-09-18 立：实机日志 126 秒 / ~500 个 tick 里 `投影跟着地图更新` **0 条**，而三条
        静默路径（帧差闸跳过 / 采纳但落在死区内 / `_maybe_track` 根本没被调到）在日志上完全
        同形，只能靠猜。心跳把三者分开：帧差闸那条会留下「画面没动」，采纳那条会留下
        「小窗/全平移 + 分/ov」，一条都不出说明是第三条（接线问题）。

        ⚠️ **限流会把要看的尖峰采样掉**（09-18 第二版教训）：tick 250ms、心跳 1.5s ⇒ 每 6 个
        tick 只记 1 个，而用户拖缩放往往就 1 秒（4 个 tick）。第二份实机日志里 35 条心跳
        **全是** `帧差0.00`，看着像「检测不到屏幕变化」，其实只是那 4 个 tick 没被采到。
        ⇒ 两条对策：`帧差` 那条记**本轮峰值**（窗口内最大值，尖峰跑不掉）；过闸/刚确立这类
        **稀有且关键**的事件 `force=True` 绕过限流（它们一次只该出几条）。
        """
        now = time.monotonic()
        if not force and now - self._track_beat_t < TRACK_BEAT_MIN_SEC:
            return
        self._track_beat_t = now
        self._mad_peak, self._track_n = 0.0, 0
        self._log_step("跟随心跳: " + msg)

    def _track_tick(self, roi, is_open, trusted=True):
        """一次重对齐尝试：近处小窗 → 单尺度全平移。**不做全域尺度搜索**（1.5s，会冻 UI）。

        `trusted` = 本 tick「地图开着」是**导航列主判据**说的（`nav_ok`）。雾兜底说的「开」
        不算数 —— 它会把关闭动画帧也判成开，而那种帧里导航列根本不在（滑条读数必失败）⇒
        判丢 ⇒ 若顺手清掉 `_last_rgba`，G 开回来就没图可放（见 `_track_lost`）。
        """
        tr = self._track
        if not is_open:
            # 本帧判「地图关」（G 关闭动画的过渡帧；状态机要连续 2 帧才确认，这半秒里
            # confirmed 仍是 True ⇒ 本函数会被调到）。此时**不跟踪、更不判丢**：
            # 过渡帧导航列 NCC 掉到 -0.03 ⇒ 滑条/图标尺子全失效 ⇒ 若照常判丢，连丢 2 次就
            # 会把投影隐藏 + 清 `_last_rgba`，于是 G 开回来时无图可放回（2026-09-18 bug①：
            # 19:41:32 跟丢 → 19:41:33 投影隐藏 → 19:41:58「上次没有成功投影」）。
            return
        px, py, pw, ph = self._follow_panel
        rx, ry, _x1, _y1 = map_roi(self._follow_panel)
        region = roi[py - ry:py - ry + ph, px - rx:px - rx + pw]   # 零拷贝视图
        if region.size == 0 or region.shape[0] < ph * 0.5 or region.shape[1] < pw * 0.5:
            self._track_beat(f"面板不在 ROI 内（{region.shape}）→ 跳过", force=True)
            return
        # 画面没动 ⇒ 地图没动 ⇒ 投影原样有效，一次匹配都不用跑（投影窗已排除截屏，
        # 面板像素只可能来自游戏本身；玩家的黄点只占几个像素，降采样后淹没在噪声里）。
        small = cv2.resize(region, (64, 34), interpolation=cv2.INTER_AREA).astype(np.int16)
        prev, self._panel_prev = self._panel_prev, small
        mad = float(np.abs(small - prev).mean()) if prev is not None else None
        # `mad is None` = 本次跟随的**第一** tick（`_panel_prev` 刚被 `_start_follow` 清成 None）。
        # 它不过闸（没有上一帧可比，就当"动过"），于是会走到下面那条路 —— 那里所有日志都用
        # `{mad_s}` 而不是 `{mad_s}`：`None:.2f` 会抛 TypeError（2026-09-19 一并修）。
        mad_s = "--" if mad is None else format(mad, ".2f")
        # 心跳限流 1.5s = 每 6 个 tick 才记 1 条，而一次拖缩放往往只有 1 秒（4 个 tick）
        # ⇒ 只记瞬时值必然漏掉尖峰（09-18 第二份日志 35 条心跳全是 0.00 就是这么来的）。
        # 记**本轮峰值**：只要这 1.5s 里有过 4.30，就一定会出现在日志里。
        self._mad_peak = max(self._mad_peak, mad or 0.0)
        if mad is not None and mad < TRACK_MOTION_MAD:
            self._track_beat(f"帧差{mad_s}<闸{TRACK_MOTION_MAD} 画面没动 → 跳过（不重对齐）"
                             f"｜本轮峰值{self._mad_peak:.2f} / {self._track_n} tick")
            tr.note_ok()
            return
        self._track_n += 1
        try:
            ref = load_bgr(tr.ref_path)
        except Exception as e:  # noqa: BLE001
            self._log_step(f"跟踪：参考图读不出（{e}）→ 停止跟踪", "WARN")
            tr.reset(); return

        # 尺度：两个「量出来的」源，优先级 滑条(1ms,±0.01) > 图标尺子(200ms,±0.05/k)。
        # 对齐分选不出尺度（vision.ZOOM_CURVE_C 长注 + CLAUDE.md ⑥）。
        s_ui, why_s = zoom_scale_from_roi(roi, self._follow_panel)
        s_src = "滑条"
        if s_ui is None:
            # 备用源：图标尺子 s≈0.374/k。要 200ms，故只在滑条读不到时才量。
            _ip, isc_i, k_i = _find_icon(roi, px - rx, py - ry, pw, ph)
            s_k = zoom_scale_from_k(k_i) if (isc_i or 0) >= TRACK_ICON_SCORE_MIN else None
            if s_k is None:
                why_s = f"{why_s}；图标尺子也不可用(分{isc_i:.2f})"
            elif abs(s_k - tr.s) > ruler_slop(s_k, k_i):
                s_ui, s_src = s_k, "图标尺子"      # 变化超出尺子自身精度 ⇒ 确实缩放了
            else:
                s_ui, s_src = tr.s, "沿用上次"      # 尺子与旧尺度一致 ⇒ 保留更精确的旧值
        scale_moved = s_ui is not None and abs(s_ui - tr.s) > TRACK_DEADBAND_S

        # 1) 尺度没变：先试上次位置附近的小窗（半径 < 半个迷宫格周期 ⇒ 窗内不可能有幽灵相位）
        sc_near = None
        if not scale_moved:
            near = find_overlay_transform(None, ref, self._follow_panel, region=region,
                                          hint_s=tr.s, hint_icon=tr.pin(), fast=True,
                                          hint_radius=TRACK_NEAR_R, ref_key=tr.ref_path)
            sc_near = near[1] if near is not None else None
            if near is not None and sc_near < TRACK_OK and near[2] >= TRACK_OVERLAP_MIN:
                self._track_beat(f"帧差{mad_s} 尺度源{s_src} s={tr.s:.3f} "
                                 f"→ 小窗(±{TRACK_NEAR_R}px) 分{sc_near:.3f} ov{near[2]:.2f} 采纳",
                                 force=True)
                self._track_adopt(near[0], tr.s, "平移")
                return
        # 尺度变了就不走小窗：在**错尺度**上小窗也能找到低分位置（实测假接受，图标偏 78~180px）

        # 1.5) 一个尺度源都没有 ⇒ **不做全平移，就地判丢**。没有可信尺度时，全平移的全局极小
        #      会落在错位置上，而且**错尺度分更低**（2026-09-18 实测：真尺度 0.42 全局极小
        #      0.086，错尺度 0.30 反而 0.126、图标偏 840px）⇒ 没有任何分数闸能分辨。
        #      小窗（±12px）是有界的、且要过 TRACK_OK 才采纳，所以上面试完就可以收手了；
        #      连丢 TRACK_LOST_MAX 次会隐藏投影 —— 与「宁可什么都不给」一致（待办 1）。
        if s_ui is None:
            self._track_beat(f"帧差{mad_s} 尺度源无（{why_s}）→ 判丢", force=True)
            self._track_lost(f"没有可信的尺度（{why_s}）", is_open, trusted)
            return

        # 2) 上一帧位置对不上了 ⇒ 换尺度做单尺度全平移。候选**按优先级**逐个试、谁先过闸用谁；
        #    **绝不"挑分最小的那个尺度"** —— 分对 s 单调偏低（技术备忘⑥），滑条值 0.35 与旧值
        #    0.31 同场竞逐时挑分必选 0.31，于是又滑回旧尺度、投影偏 26px。
        cands = [s_ui]
        if abs(s_ui - tr.s) > 0.02:
            cands.append(tr.s)      # 兜底：滑条读数万一读错（圆点串到别的行），别把旧尺度丢了
        for s in cands:
            r = find_overlay_transform(None, ref, self._follow_panel, region=region,
                                       hint_s=s, fast=True, ref_key=tr.ref_path)
            if r is None:
                continue
            v = verdict(sc_near, r[1], r[2])
            if v == "accept":
                self._track_beat(f"帧差{mad_s} 尺度源{s_src} s={s:.3f} "
                                 f"→ 全平移 分{r[1]:.3f} ov{r[2]:.2f}（旧分"
                                 f"{'None' if sc_near is None else f'{sc_near:.3f}'}）采纳",
                                 force=True)
                self._track_adopt(r[0], float(s), f"重对齐 尺度{s:.2f}"
                                  f"（{s_src if s == s_ui else '沿用上次'}）")
                return
            if v == "keep":
                self._track_beat(f"帧差{mad_s} 尺度源{s_src} s={s:.3f} "
                                 f"→ 全平移 分{r[1]:.3f} ov{r[2]:.2f} 不如旧位置 ⇒ 保持不动",
                                 force=True)
                tr.note_ok(); return
        msg = (f"证据不足（候选尺度 {[round(c, 3) for c in cands]} 都不过闸；{why_s}）")
        self._track_beat(f"帧差{mad_s} 尺度源{s_src} → 全平移 候选"
                         f"{[round(c, 3) for c in cands]} 全不过闸（旧分"
                         f"{'None' if sc_near is None else f'{sc_near:.3f}'}）→ 判丢", force=True)
        self._track_lost(msg, is_open, trusted)

    def _track_adopt(self, M, s, why):
        """接受新变换：过死区才重烘焙投影（省掉一次 warpAffine 1920×1080 + QPixmap）。"""
        tr = self._track
        moved = tr.advanced(M, s)
        tr.adopt(M, s)
        self._track_idle = 0            # 动过 ⇒ 恢复全速
        if not moved:
            return
        try:
            rgba = map_to_overlay_rgba(tr.ref_path, M, *self._screen_size(),
                                       wall_alpha=self.settings.wall_alpha)
        except Exception as e:  # noqa: BLE001
            self._log_step(f"跟踪：重烘焙投影失败（{e}）", "WARN"); return
        self._update_overlay(rgba)      # 不碰跟随状态机（同一个投影在挪位置）
        now = time.monotonic()
        if now - self._track_log_t >= TRACK_LOG_MIN_SEC:
            self._track_log_t = now
            self._log_step(f"跟随: 投影跟着地图更新（{why}）")

    def _track_lost(self, why, is_open=True, trusted=True):
        """判丢：连忍 `TRACK_LOST_MAX` 次才动投影 —— 判丢后**隐藏**而不是留着错位置。

        `_last_rgba` 是「G 重开地图放回哪张」的**唯一**来源，只在**地图确实开着**时清。
        两个条件都要：
        - `is_open` = 本 tick 的原始开合读数。G 的关闭动画帧也会走到判丢（见 `_track_tick`
          开头），那种「丢」是假的，把图清了就再也放不回来（2026-09-18 bug①）。
        - `trusted` = 这个「开」是**导航列主判据**说的。雾兜底是低精度高召回，关闭动画帧里
          导航列 NCC 掉到 −0.01（列根本不在）而雾 0.41 ⇒ 照样判「开」⇒ 光靠 `is_open`
          拦不住。实机 22:52:35 就是这么丢的：判丢 → 清图 → 22:52:42「上次没有成功投影
          ⇒ 不自动恢复」。所以雾兜底说的「开」只敢隐藏投影，不敢销毁 `_last_rgba`。
        """
        tr = self._track
        n = tr.note_lost()
        self._track_idle = 0
        if n < TRACK_LOST_MAX:
            return
        self._set_overlay_visible(False)
        if is_open and trusted:
            self._last_rgba = None  # 真丢（主判据说地图开着）⇒ 不留旧的，G 重开也别把错位置放回来
        tr.reset()
        self._log_step(f"跟丢（{why}）→ 已隐藏投影，请按热键重新匹配", "WARN")

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
            self._set_status("没有可回收的截图"); return
        EVAL_INBOX.mkdir(parents=True, exist_ok=True)
        dst = EVAL_INBOX / self._last_shot_path.name
        try:
            import shutil
            shutil.copy2(self._last_shot_path, dst)
            self._set_status(f"已回收失败样本→ {dst.name}（待标注进 eval/labels.csv）")
        except Exception as e:  # noqa: BLE001
            self._set_status(f"回收失败: {e}")

    # ---- 日志窗 / 参考图对话框（T2b：从主窗搬出去的常驻控件）----
    def _show_log_window(self):
        """运行日志窗。**关窗不销毁 `log_view`** ⇒ 关着的那段日志不会丢，再打开还在。"""
        if self._log_window is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("运行日志")
            dlg.resize(640, 340)
            lay = QVBoxLayout(dlg); lay.setContentsMargins(6, 6, 6, 6)
            lay.addWidget(self.log_view)      # 重挂父控件，但对象所有权仍在 MainWindow
            row = QHBoxLayout()
            btn_clear = QPushButton("清空"); btn_clear.clicked.connect(self.log_view.clear)
            row.addWidget(btn_clear); row.addStretch(1)
            lay.addLayout(row)
            self._log_window = dlg
        self._log_window.show(); self._log_window.raise_(); self._log_window.activateWindow()

    def _show_ref_picker(self):
        """「参考图」对话框：方向/门/楼层/入口 4 个下拉（T2b 从主窗搬进来）。

        ⚠️ **combo 对象仍归 MainWindow，只是换了父布局容器** —— `currentIndexChanged` 上挂的
        `_on_direction_changed` / `_resolve_seed` 一个都没断，`_realign` 读的仍是本体
        (`self.direction_combo.currentText()`)。对话框只负责显示，关掉不销毁 combo，
        所以下次再点开还是同一批对象、连接照旧。（这是本次改造最容易回归的点。）
        """
        if self._ref_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("参考图（手动纠错用）")
            lay = QVBoxLayout(dlg); lay.setContentsMargins(8, 8, 8, 8)
            for text, widget in [("方向（入口朝向）", self.direction_combo),
                                 ("门特征", self.door_combo),
                                 ("楼层", self.floor_combo),
                                 ("入口(引索匹配用)", self.entrance_combo)]:
                lay.addWidget(QLabel(text)); lay.addWidget(widget)
            tip = QLabel("提示：选完关掉本窗，主窗「▶ 按此种子重新对齐」即用这个种子。")
            tip.setStyleSheet("color:#888; font-size:11px;"); tip.setWordWrap(True)
            lay.addWidget(tip)
            row = QHBoxLayout()
            btn_ok = QPushButton("确定"); btn_ok.clicked.connect(dlg.accept)
            row.addWidget(btn_ok); row.addStretch(1)
            lay.addLayout(row)
            dlg.resize(300, 260)
            self._ref_dialog = dlg
        self._ref_dialog.show(); self._ref_dialog.raise_(); self._ref_dialog.activateWindow()

    # ---- 选项菜单（UI改造B2：低频折叠；T2b 收成三组）----
    def _show_options_menu(self):
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        fix = menu.addMenu("🛠 手动纠错")
        fix.addAction("✂ 手框样本匹配", self._manual_sample_pick)
        fix.addAction("✋ 手动选点重合", self._three_point_calib)
        fix.addAction("🎯 参考图（方向/门/楼层/入口）…", self._show_ref_picker)
        diag = menu.addMenu("🧪 诊断")
        diag.addAction("📜 运行日志", self._show_log_window)
        diag.addAction("✗ 标记上次错→回收", self._mark_wrong)
        diag.addAction("🗂 素材管理", self._manage_materials)
        diag.addAction("⚙ 设置", self._open_settings)
        diag.addAction("ℹ 热键状态", self._show_hotkey_status)
        menu.addSeparator()
        menu.addAction("✕ 退出", self._quit)
        menu.exec(QCursor.pos())

    def _show_hotkey_status(self):
        hk = getattr(self, "_hotkeys", None)
        reg = hk is not None and bool(getattr(hk, "_callbacks", {}))
        QMessageBox.information(self, "热键状态",
            f"当前热键: {self.settings.hotkey}\n注册状态: "
            + ("已注册" if reg else "未注册（可能被占用，去设置改键）")
            + f"\n\n状态灯: ● {self._led_text}")

    # ---- 设置（阶段2）----
    def _open_settings(self):
        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() and dlg.result_settings is not None:
            self.settings = dlg.result_settings
            save_settings(self.settings)
            self._apply_settings()

    def _apply_settings(self):
        """应用当前 settings：透明度/日志窗/墙体 alpha 即时；热键重注册。"""
        # T2b：show_log 语义已改为「启动时是否打开日志窗」——运行中改设置就即时开/关。
        # 注意不能再去 `log_view.setVisible(...)`：它现在是日志窗的子控件，直接隐藏它
        # 只会让日志窗里空一块，窗口还杵在那儿。
        if self.settings.show_log:
            self._show_log_window()
        elif self._log_window is not None:
            self._log_window.hide()
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
        if self.settings.auto_follow:
            self._start_follow()
        else:
            self._stop_follow()

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
        for dlg in (self._log_window, self._ref_dialog):
            if dlg is not None:
                dlg.close()
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
        p = panel_for_screen(sw, sh)
        win._log_step(f"屏幕 {sw}x{sh} 面板 {tuple(p) if p else '未适配(非16:9，可config校准)'}"
                      + ("" if (sw, sh) == (1920, 1080) else "（非基准分辨率，外推/校准值，未实测）"), "INFO")
        win._log_step("就绪：先按 g 打开游戏地图、刚进入口(图标在视野)，再 "
                      + settings.hotkey, "INFO")
    except Exception as e:  # noqa: BLE001
        win.set_led(False, "热键失败")
        win._log_step(f"热键注册失败: {e}（{FIX_PATH}参考图 手动选门 / 3点标定仍可用）", "ERROR")
        QMessageBox.warning(win, "热键注册失败",
            f"{e}\n\n{FIX_PATH}参考图 手动选门 + 3 点标定仍可用。\n（可在「设置」改热键）")
    win._hotkeys = hotkeys  # 保引用

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

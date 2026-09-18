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
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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
from core.vision import (FOG_OPEN_MIN, NAV_NCC_MIN, detect_fog_panel, load_bgr,
                         map_is_open, map_open_from_roi, map_roi, panel_for_screen,
                         zoom_scale_from_roi)
from ui.capture import capture_monitor, capture_region, monitor_size
from ui.follow import (
    FOLLOW_IDLE_INTERVAL_MS, FOLLOW_INTERVAL_MS, FollowState,
)
from ui.hotkey import MOD_CONTROL, MOD_SHIFT, HotkeyManager, parse_hotkey
from ui.manage_materials import ManageMaterialsDialog
from ui.overlay import MapOverlay
from ui.preview import SamplePreview
from ui.settings import Settings, SettingsDialog, load as load_settings, save as save_settings
from ui.track import (TRACK_IDLE_INTERVAL_MS, TRACK_LOG_MIN_SEC, TRACK_LOST_MAX,
                      TRACK_MOTION_MAD, TRACK_NEAR_R, TRACK_OK, Tracker, verdict)

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
        self._panel_prev = None           # 面板降采样上一帧（画面没动就整段跳过，省 ~70ms/次）
        self._screen_wh = None            # 屏幕尺寸缓存（避免每 tick 新建/销毁 mss DC 句柄）

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

        # ---- UI改造B1：主界面只留高频按钮；低频(素材/标记错/设置/热键状态)进「⋯选项」菜单 ----
        self.btn_onematch = QPushButton("🔴 一键匹配（=热键）")
        self.btn_onematch.setStyleSheet("font-weight:bold; padding:8px; background:#2a4a2a;")
        self.btn_onematch.clicked.connect(self._entrance_pipeline)
        self.btn_realign = QPushButton("▶ 按此种子对齐（用上面选的门）")
        self.btn_picksample = QPushButton("✂ 手框样本匹配")
        self.btn_calib = QPushButton("✋ 手动选点重合")
        self.btn_hide = QPushButton("👁 隐藏地图")
        self.btn_options = QPushButton("⋯ 选项")
        self.btn_quit = QPushButton("✕ 退出")
        self.btn_realign.clicked.connect(self._realign)
        self.btn_picksample.clicked.connect(self._manual_sample_pick)
        self.btn_calib.clicked.connect(self._three_point_calib)
        self.btn_hide.clicked.connect(self._hide_overlay)
        self.btn_options.clicked.connect(self._show_options_menu)
        self.btn_quit.clicked.connect(self._quit)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(int(self.settings.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        root = QVBoxLayout(); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(5)
        root.addWidget(self.led)
        root.addWidget(self.btn_onematch)
        root.addWidget(QLabel("── 手动纠错（自动出错时用）──"))
        for text, widget in [("方向（入口朝向）", self.direction_combo),
                             ("门特征", self.door_combo),
                             ("楼层", self.floor_combo),
                             ("入口(引索匹配用)", self.entrance_combo)]:
            root.addWidget(QLabel(text)); root.addWidget(widget)
        root.addWidget(self.seed_label)
        root.addWidget(self.btn_realign)
        root.addWidget(self.btn_picksample)
        root.addWidget(self.btn_calib)
        misc = QHBoxLayout()
        misc.addWidget(self.btn_hide); misc.addWidget(self.btn_options); misc.addWidget(self.btn_quit)
        root.addLayout(misc)
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
        if self.settings.auto_follow:
            self._start_follow()

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
            self._log_step(f"入口图标未检出(分{isc:.2f}) → 手框样本/3点标定", "WARN")
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
                    "多探明周围结构后再按 / 手框含墙角结构的小块 / 3点标定", "WARN")
                self._log(et, res, icon_pos, isc, None, corrected=False)
                return
            if retried:
                self._log_step(f"扩大取样重试({sample.shape[1]}px)：单类降至{dom * 100:.0f}%，分差已拉开", "INFO")
        if not res:
            self._log_step("入口匹配无结果 → 手框样本/3点标定", "WARN")
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
        corr_note = ("（面板为大段均匀区，无锚点可对齐——种子ID可信，地图居中仅供参考；"
                     "精确重合请3点标定）" if (dom_frac is not None and dom_frac > DOMINANT_NOTE_MIN) else "")
        # 退化防御：top1 与 top2 分差<0.001（多种子同分0.000）→ 样本无判别结构/图标假阳，不假阳报告
        if _degenerate(res):
            self._log_step(f"匹配退化(多种子同分 {sc:.3f}) → 样本无判别结构（均匀区/假阳图标），"
                           "请手框含墙角结构的小块或3点标定", "WARN")
            self._log(et, res, icon_pos, isc, None, corrected=corrected)
            return
        if sc >= 1e8:  # 哨兵分兜底（find_seed_by_entrance 已跳过无尺度种子，此处防御）
            self._log_step("匹配无有效尺度（样本过大/引索缺）→ 框小一点(入口局部结构)或3点标定", "WARN")
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
        self.status.setText(
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
            self.status.setText("屏幕分辨率未适配（非16:9且未校准）——3点标定仍可用")
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

    def _realign(self):
        """用当前选定的方向+门+楼层重新对齐（全搜，无入口 hint）。"""
        info = self._current_map()
        if info is None:
            self.status.setText("请先选好方向+门"); return
        if self._shot is None and not self._capture():
            return
        panel = detect_fog_panel(self._shot)
        if panel is None:
            self.status.setText("屏幕分辨率未适配（非16:9且未校准）"); return
        ref = load_bgr(str(info.path))
        align = find_overlay_transform(self._shot, ref, panel)
        if align is None:
            self.status.setText("对齐失败：探明不足"); return
        M, asc, ov = align
        if asc < ALIGN_SCORE_MAX and ov >= OVERLAP_MIN:
            rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size(), wall_alpha=self.settings.wall_alpha)
            self._show_overlay(rgba)
            self._seed_track(self._shot, self._seed, info.floor, info.path, M)
            self.status.setText(f"已对齐(匹配{asc:.2f} 重叠{ov:.2f})：{info.key}")
            self._log_step(f"手动重对齐: {info.key} 重合{asc:.2f} 重叠{ov:.2f} 已投影", "OK")
        else:
            self._refuse(info.key, f"手动重对齐没过闸(重合{asc:.2f} 重叠{ov:.2f})")
            self.status.setText(f"未投影：对齐不可靠(匹配{asc:.2f} 重叠{ov:.2f})｜{info.key}")

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
        # 3 点标定走的是 affine_from_points —— **可能带旋转**，参数化与跟踪用的相似变换
        # 不同（跟踪只认 s/tx/ty），喂进去会算错。故显式断根，该投影不参与跟随重对齐。
        self._track.reset()
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
            self._log_step("手框样本仍无匹配 → 换一处含墙角/房间边缘的区域再框，或用3点标定", "WARN")
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
        """显示匹配用的入口样本（让玩家看到选对没），贴主窗右侧。"""
        if self.preview is None:
            self.preview = SamplePreview()
            g = self.geometry()
            self.preview.move(g.right() + 8, g.top())
        self.preview.set_sample(bgr)
        self.preview.show()

    def _refuse(self, label, why):
        """不确信 / 对齐没过闸 ⇒ **一张图都不显示**（待办 1，2026-09-16 晚）。

        旧行为是 `_show_centered_if_any` → `auto_align_overlay` 把参考图按
        `min(pw/cw, ph/ch)` **缩放铺满迷雾面板**：比例与游戏内毫无关系，却画得工整、
        结构清楚，看起来就像一张"已经对齐好的地图"—— 用户报的「乱给一张素材、
        比例都不对」就是它，不是对齐结果。宁可什么都不给，也不给一张像是对的的假图：
        假图会让人以为算法定位到了别处，比空白有害得多。

        种子ID仍写进状态栏与日志（那个数是有意义的）；想看参考图请显式用
        「▶ 按此种子对齐」（全搜）或「✋ 手动选点重合」（3 点标定）。
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
        is_open, why = map_open_from_roi(roi, self._follow_panel)
        evt = self._follow.feed(is_open)
        if evt is None:
            # 稳定态：地图开着 + 投影显示 ⇒ 让它跟着地图平移/缩放（自动跟随·第二步）
            if visible and self._follow.confirmed is True:
                self._maybe_track(roi)
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
    def _maybe_track(self, roi):
        """跟踪节流：画面没动就整段跳过；静止时降频，动了回全速。"""
        if self._track_arm:
            self._track_arm = False     # 开态刚确立，本 tick 只做准备
            return
        if not self._track.active:
            return
        if self._track_idle:
            self._track_idle += 1
            step = max(1, TRACK_IDLE_INTERVAL_MS // FOLLOW_INTERVAL_MS)
            if self._track_idle % step:
                return
        self._track_tick(roi)

    def _track_tick(self, roi):
        """一次重对齐尝试：近处小窗 → 单尺度全平移。**不做全域尺度搜索**（1.5s，会冻 UI）。"""
        tr = self._track
        px, py, pw, ph = self._follow_panel
        rx, ry, _x1, _y1 = map_roi(self._follow_panel)
        region = roi[py - ry:py - ry + ph, px - rx:px - rx + pw]   # 零拷贝视图
        if region.size == 0 or region.shape[0] < ph * 0.5 or region.shape[1] < pw * 0.5:
            return
        # 画面没动 ⇒ 地图没动 ⇒ 投影原样有效，一次匹配都不用跑（投影窗已排除截屏，
        # 面板像素只可能来自游戏本身；玩家的黄点只占几个像素，降采样后淹没在噪声里）。
        small = cv2.resize(region, (64, 34), interpolation=cv2.INTER_AREA).astype(np.int16)
        prev, self._panel_prev = self._panel_prev, small
        if prev is not None and float(np.abs(small - prev).mean()) < TRACK_MOTION_MAD:
            tr.note_ok()
            return
        try:
            ref = load_bgr(tr.ref_path)
        except Exception as e:  # noqa: BLE001
            self._log_step(f"跟踪：参考图读不出（{e}）→ 停止跟踪", "WARN")
            tr.reset(); return

        # 尺度：滑条实测值优先。对齐分选不出尺度（vision.ZOOM_TABLE_* 长注 + CLAUDE.md ⑥）
        s_ui, why_s = zoom_scale_from_roi(roi, self._follow_panel)
        scale_moved = s_ui is not None and abs(s_ui - tr.s) > TRACK_DEADBAND_S

        # 1) 尺度没变：先试上次位置附近的小窗（半径 < 半个迷宫格周期 ⇒ 窗内不可能有幽灵相位）
        sc_near = None
        if not scale_moved:
            near = find_overlay_transform(None, ref, self._follow_panel, region=region,
                                          hint_s=tr.s, hint_icon=tr.pin(), fast=True,
                                          hint_radius=TRACK_NEAR_R, ref_key=tr.ref_path)
            sc_near = near[1] if near is not None else None
            if near is not None and sc_near < TRACK_OK and near[2] >= TRACK_OVERLAP_MIN:
                self._track_adopt(near[0], tr.s, "平移")
                return
        # 尺度变了就不走小窗：在**错尺度**上小窗也能找到低分位置（实测假接受，图标偏 78~180px）

        # 2) 上一帧位置对不上了 ⇒ 换尺度做单尺度全平移。候选**按优先级**逐个试、谁先过闸用谁；
        #    **绝不"挑分最小的那个尺度"** —— 分对 s 单调偏低（技术备忘⑥），滑条值 0.35 与旧值
        #    0.31 同场竞逐时挑分必选 0.31，于是又滑回旧尺度、投影偏 26px。
        cands = [s_ui] if s_ui is not None else []
        if s_ui is None or abs(s_ui - tr.s) > 0.02:
            cands.append(tr.s)
        for s in cands:
            r = find_overlay_transform(None, ref, self._follow_panel, region=region,
                                       hint_s=s, fast=True, ref_key=tr.ref_path)
            if r is None:
                continue
            v = verdict(sc_near, r[1], r[2])
            if v == "accept":
                self._track_adopt(r[0], float(s), f"重对齐 尺度{s:.2f}"
                                                    f"{'（滑条）' if s == s_ui else '（沿用上次）'}")
                return
            if v == "keep":
                tr.note_ok(); return
        self._track_lost(f"证据不足（候选尺度 {[round(c, 3) for c in cands]} 都不过闸）")

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

    def _track_lost(self, why):
        """判丢：连忍 TRACK_LOST_MAX 次才动投影 —— 判丢后**隐藏**而不是留着错位置。"""
        tr = self._track
        n = tr.note_lost()
        self._track_idle = 0
        if n < TRACK_LOST_MAX:
            return
        self._set_overlay_visible(False)
        self._last_rgba = None      # 不留旧的：G 重开地图时也不许把错位置放回来
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
            self.status.setText("没有可回收的截图"); return
        EVAL_INBOX.mkdir(parents=True, exist_ok=True)
        dst = EVAL_INBOX / self._last_shot_path.name
        try:
            import shutil
            shutil.copy2(self._last_shot_path, dst)
            self.status.setText(f"已回收失败样本→ {dst.name}（待标注进 eval/labels.csv）")
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"回收失败: {e}")

    # ---- 选项菜单（UI改造B2：低频折叠）----
    def _show_options_menu(self):
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("🗂 素材管理", self._manage_materials)
        menu.addAction("✗ 标记上次错→回收", self._mark_wrong)
        menu.addAction("⚙ 设置", self._open_settings)
        menu.addAction("ℹ 热键状态", self._show_hotkey_status)
        menu.exec(QCursor.pos())

    def _show_hotkey_status(self):
        hk = getattr(self, "_hotkeys", None)
        reg = hk is not None and bool(getattr(hk, "_callbacks", {}))
        QMessageBox.information(self, "热键状态",
            f"当前热键: {self.settings.hotkey}\n注册状态: "
            + ("已注册" if reg else "未注册（可能被占用，去设置改键）"))

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
        win._log_step(f"热键注册失败: {e}（手动选门/3点标定仍可用）", "ERROR")
        QMessageBox.warning(win, "热键注册失败",
            f"{e}\n\n手动选门 + 3 点标定仍可用。\n（可在「设置」改热键）")
    win._hotkeys = hotkeys  # 保引用

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

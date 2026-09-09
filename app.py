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
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QLabel, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from core.alignment import (
    affine_from_points, auto_align_overlay, find_overlay_transform, map_to_overlay_rgba,
)
from core.entrance import build_entrance_transform, find_seed_by_entrance, load_index
from core.map_library import MapLibrary
from core.vision import FIXED_PANEL, detect_fog_panel, load_bgr
from ui.capture import capture_monitor
from ui.hotkey import MOD_CONTROL, MOD_SHIFT, HotkeyManager
from ui.manage_materials import ManageMaterialsDialog
from ui.overlay import MapOverlay

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
    """在图上点 N 个点。显示图缩放到窗口，点击坐标映射回原图。返回 [(x,y)...] 或 []。"""

    def __init__(self, bgr, n: int, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.n = n
        self.pts: list[tuple[int, int]] = []
        self._scale = 1.0
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lay = QVBoxLayout(self); lay.addWidget(self._label)
        self._set_image(bgr)

    def _set_image(self, bgr):
        h, w = bgr.shape[:2]
        maxw, maxh = 1280, 720
        self._scale = min(1.0, maxw / w, maxh / h)
        dw, dh = int(w * self._scale), int(h * self._scale)
        small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        qimg = QImage(small.tobytes(), dw, dh, 3 * dw, QImage.Format.Format_RGB888).rgbSwapped()
        self._label.setPixmap(QPixmap.fromImage(qimg))
        self._label.setFixedSize(dw, dh)
        self.adjustSize()

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = e.position().toPoint()
        x, y = int(p.x() / self._scale), int(p.y() / self._scale)
        self.pts.append((x, y))
        self._label.setText(f"已点 {len(self.pts)}/{self.n}：{self.pts}")
        if len(self.pts) >= self.n:
            self.accept()


class MainWindow(QWidget):
    def __init__(self, lib: MapLibrary):
        super().__init__()
        self.lib = lib
        self._shot = None           # 最近捕获的屏幕 BGR
        self._seed = None           # 当前选定种子（方向+门解析）
        self.overlay: MapOverlay | None = None
        self._last_shot_path = None  # 最近截图保存路径（log/回收用）

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setWindowTitle("加页手记 地图助手")
        self.setFixedWidth(236)

        self.direction_combo = QComboBox(); self.direction_combo.addItems(lib.directions())
        self.door_combo = QComboBox()
        self.floor_combo = QComboBox(); self.floor_combo.addItems(["一楼", "二楼"])
        self.entrance_combo = QComboBox(); self.entrance_combo.addItems(list(ENTRANCE_TYPES))
        self.seed_label = QLabel("")
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        self.door_combo.currentIndexChanged.connect(self._resolve_seed)

        self.btn_realign = QPushButton("↻ 重新对齐(当前门)")
        self.btn_calib = QPushButton("✋ 3点标定(手动兜底)")
        self.btn_mark_wrong = QPushButton("✗ 标记上次错→回收")
        self.btn_hide = QPushButton("👁 隐藏地图")
        self.btn_manage = QPushButton("🗂 素材管理")
        self.btn_quit = QPushButton("✕ 退出")
        self.btn_realign.clicked.connect(self._realign)
        self.btn_calib.clicked.connect(self._three_point_calib)
        self.btn_mark_wrong.clicked.connect(self._mark_wrong)
        self.btn_hide.clicked.connect(self._hide_overlay)
        self.btn_manage.clicked.connect(self._manage_materials)
        self.btn_quit.clicked.connect(self._quit)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100); self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        root = QVBoxLayout(); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(5)
        for text, widget in [("方向（入口朝向）", self.direction_combo),
                             ("门特征", self.door_combo),
                             ("楼层", self.floor_combo),
                             ("入口(引索匹配用)", self.entrance_combo)]:
            root.addWidget(QLabel(text)); root.addWidget(widget)
        root.addWidget(self.seed_label)
        root.addWidget(QLabel("Ctrl+Shift+F = 入口匹配+对齐"))
        root.addWidget(self.btn_realign)
        root.addWidget(self.btn_calib)
        root.addWidget(self.btn_mark_wrong)
        root.addWidget(self.btn_hide)
        root.addWidget(self.btn_manage)
        root.addWidget(self.btn_quit)
        root.addWidget(QLabel("地图透明度")); root.addWidget(self.opacity_slider)
        self.status = QLabel("就绪。Ctrl+Shift+F 入口匹配（需先按 g 打开地图、刚进入口）。")
        self.status.setStyleSheet("color:#aaa; word-wrap:break-word; font-size:11px;")
        root.addWidget(self.status); root.addStretch(1)
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
            self.status.setText(f"捕获失败: {e}")
            return False

    # ---- 一次热键流程：入口引索匹配 → 两段式对齐 → 投影 ----
    def _entrance_pipeline(self):
        if not self._capture():
            return
        shot = self._shot
        et = self.entrance_combo.currentText()
        res, icon_pos, isc = find_seed_by_entrance(shot, self.lib, et, top_n=3)
        if not res:
            self.status.setText(
                f"入口匹配失败：入口图标未检出(分{isc:.2f})。请确认入口在视野内，"
                f"或手动选门/3点标定。")
            self._log(et, res, icon_pos, isc, None, corrected=False)
            return
        best = res[0]
        sc, seed, key, fl, _s, _mloc = best
        # 填 UI（种子/方向/门/楼层）
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
            if show_ok:
                info = self.lib.get(seed, fl)
                if info is not None:
                    rgba = map_to_overlay_rgba(str(info.path), align[0], *self._screen_size())
                    self._show_overlay(rgba)
            else:
                self._show_centered_if_any(seed, fl)
        else:
            self._show_centered_if_any(seed, fl)

        confident = (sc < SCORE_CONFIDENT and ov is not None and ov >= OVERLAP_MIN)
        self._log(et, res, icon_pos, isc, align, corrected=False)
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
            rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size())
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
        rgba = map_to_overlay_rgba(str(info.path), M, *self._screen_size())
        self._show_overlay(rgba)
        self.status.setText(f"3点标定投影：{info.key}")

    # ---- 投影显示 ----
    def _screen_size(self):
        g = QApplication.primaryScreen().geometry()
        return g.width(), g.height()

    def _show_overlay(self, rgba):
        if self.overlay is None:
            self.overlay = MapOverlay(QApplication.primaryScreen().geometry())
        self.overlay.set_image(rgba)
        self.overlay.setWindowOpacity(self.opacity_slider.value() / 100.0)
        self.overlay.show()

    def _show_centered(self, ref):
        h, w = self._screen_size()
        rgba, (ox, oy) = auto_align_overlay(ref, FIXED_PANEL, rotate=0)
        full = np.zeros((h, w, 4), dtype=np.uint8)
        rh, rw = rgba.shape[:2]
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(w, ox + rw), min(h, oy + rh)
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
        except Exception:  # noqa: BLE001  日志失败不影响主流程
            pass

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
        hk = getattr(self, "_hotkeys", None)
        if hk is not None:
            try:
                hk.unregister_all()
            except Exception:  # noqa: BLE001
                pass
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    lib = MapLibrary.load(MAP_DIR)
    win = MainWindow(lib)
    win.show()

    hotkeys = HotkeyManager()
    try:
        hotkeys.register(ord("F"), win._entrance_pipeline, MOD_CONTROL | MOD_SHIFT)
        win.status.setText("就绪。Ctrl+Shift+F 入口匹配+对齐（需先按 g 打开地图、刚进入口）。")
    except Exception as e:  # noqa: BLE001
        win.status.setText(f"热键注册失败: {e}")
    win._hotkeys = hotkeys  # 保引用

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

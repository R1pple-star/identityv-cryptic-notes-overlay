# -*- coding: utf-8 -*-
"""
用户设置（持久化到 settings.json）
================================
软件内调整：悬浮窗透明度、地图墙体透明度、热键、日志显隐、样本裁剪尺寸。
与 config.toml（算法常量，eval 读它）分离——本文件只管用户偏好，
不影响匹配/对齐算法。不放匹配/对齐阈值（守 CLAUDE.md 阈值稳定精神）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSlider, QVBoxLayout,
)

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "settings.json"


@dataclass
class Settings:
    overlay_opacity: float = 0.7      # 悬浮窗整体透明度 0.2-1.0
    wall_alpha: int = 160             # 地图墙体 alpha（map_to_overlay_rgba 原硬编码 160）
    hotkey: str = "Ctrl+Shift+F"      # 热键串，启动时 parse 成 (vk, mods)
    show_log: bool = True             # 运行日志区显隐
    sample_half_frac: float = 0.18    # 入口样本裁剪半边比例（icon 贴边时内部自动增大到 0.25）
    auto_follow: bool = True          # 自动跟随·第一步：G 开关游戏地图时投影同步显隐


def load() -> Settings:
    """读 settings.json；缺字段用默认；文件不存在/损坏用全默认，绝不崩。"""
    s = Settings()
    try:
        if SETTINGS_PATH.exists():
            d = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for k in ("overlay_opacity", "wall_alpha", "hotkey", "show_log",
                      "sample_half_frac", "auto_follow"):
                if k in d:
                    setattr(s, k, d[k])
    except Exception:  # noqa: BLE001
        pass
    return s


def save(s: Settings) -> None:
    """写 settings.json（utf-8，缩进）。写失败不崩（用户偏好非关键）。"""
    try:
        SETTINGS_PATH.write_text(
            json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


class SettingsDialog(QDialog):
    """设置面板。确定→设 result_settings 并 accept；取消→不动。"""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(360, 320)
        self.result_settings: Settings | None = None

        self.op_slider = QSlider(Qt.Orientation.Horizontal); self.op_slider.setRange(20, 100)
        self.op_slider.setValue(int(settings.overlay_opacity * 100))
        self.op_label = QLabel(f"悬浮窗透明度 {self.op_slider.value()}%")
        self.op_slider.valueChanged.connect(
            lambda v: self.op_label.setText(f"悬浮窗透明度 {v}%"))

        self.wall_slider = QSlider(Qt.Orientation.Horizontal); self.wall_slider.setRange(80, 255)
        self.wall_slider.setValue(settings.wall_alpha)
        self.wall_label = QLabel(f"地图墙体透明度 {settings.wall_alpha}")
        self.wall_slider.valueChanged.connect(
            lambda v: self.wall_label.setText(f"地图墙体透明度 {v}"))

        self.hk_edit = QLineEdit(settings.hotkey)
        self.hk_edit.setPlaceholderText("如 Ctrl+Shift+F / Ctrl+Alt+G / F8")

        self.log_chk = QCheckBox("显示运行日志区")
        self.log_chk.setChecked(settings.show_log)

        self.follow_chk = QCheckBox("自动跟随地图开合（G 开关小地图时投影同步显隐）")
        self.follow_chk.setChecked(settings.auto_follow)

        self.sample_spin = QDoubleSpinBox()
        self.sample_spin.setRange(0.05, 0.40); self.sample_spin.setSingleStep(0.01)
        self.sample_spin.setValue(settings.sample_half_frac)

        btn_ok = QPushButton("确定"); btn_ok.clicked.connect(self._ok)
        btn_cancel = QPushButton("取消"); btn_cancel.clicked.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.op_label); lay.addWidget(self.op_slider)
        lay.addWidget(self.wall_label); lay.addWidget(self.wall_slider)
        lay.addWidget(QLabel("热键（修饰键+主键，用 + 连接）")); lay.addWidget(self.hk_edit)
        lay.addWidget(self.log_chk)
        lay.addWidget(self.follow_chk)
        lay.addWidget(QLabel("入口样本裁剪比例（手框样本范围参考）"))
        lay.addWidget(self.sample_spin)
        row = QHBoxLayout(); row.addWidget(btn_ok); row.addWidget(btn_cancel); row.addStretch(1)
        lay.addLayout(row)

    def _ok(self):
        from ui.hotkey import parse_hotkey
        hk = self.hk_edit.text().strip()
        try:
            parse_hotkey(hk)  # 验证格式
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "热键格式错误", f"{e}\n示例：Ctrl+Shift+F")
            return
        self.result_settings = Settings(
            overlay_opacity=self.op_slider.value() / 100,
            wall_alpha=self.wall_slider.value(),
            hotkey=hk,
            show_log=self.log_chk.isChecked(),
            sample_half_frac=float(self.sample_spin.value()),
            auto_follow=self.follow_chk.isChecked(),
        )
        self.accept()

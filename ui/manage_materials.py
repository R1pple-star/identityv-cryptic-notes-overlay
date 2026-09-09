# -*- coding: utf-8 -*-
"""
素材管理对话框
==============
让用户自己增删改「完整地图」素材。素材就是素材目录下的 PNG 文件，
文件名遵循约定：{种子号} {方向}-{门}{楼层}.png
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.map_library import DIRECTIONS, FLOORS, parse_filename


class ManageMaterialsDialog(QDialog):
    def __init__(self, dir_path: str, on_change=None, parent=None):
        super().__init__(parent)
        self.dir_path = Path(dir_path)
        self.on_change = on_change
        self.setWindowTitle("素材管理")
        self.resize(820, 520)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["种子", "方向-门", "楼层", "文件名"])
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 70)
        self.table.setColumnWidth(3, 300)

        self.btn_add = QPushButton("添加素材")
        self.btn_add.clicked.connect(self._add)
        self.btn_del = QPushButton("删除选中")
        self.btn_del.clicked.connect(self._delete)
        self.btn_reload = QPushButton("刷新")
        self.btn_reload.clicked.connect(self._reload)

        tip = QLabel("约定文件名：{种子号} {方向}-{门}{楼层}.png   示例：10 北-1门一楼.png")
        tip.setStyleSheet("color:#aaa;")

        btns = QHBoxLayout()
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_reload)
        btns.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addWidget(tip)
        lay.addWidget(self.table, 1)
        lay.addLayout(btns)

        self._reload()

    def _list_files(self) -> list[Path]:
        if not self.dir_path.is_dir():
            return []
        return sorted([p for p in self.dir_path.iterdir() if p.suffix.lower() == ".png"])

    def _reload(self):
        self.table.setRowCount(0)
        for p in self._list_files():
            info = parse_filename(p.name)
            if info is None:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(info.seed)))
            self.table.setItem(r, 1, QTableWidgetItem(f"{info.direction}-{info.door}"))
            self.table.setItem(r, 2, QTableWidgetItem(info.floor))
            self.table.setItem(r, 3, QTableWidgetItem(p.name))
        if self.on_change:
            self.on_change()

    def _add(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择地图图片", "", "图片 (*.png *.jpg *.jpeg)")
        if not path:
            return
        seed, ok = QInputDialog.getInt(self, "添加素材", "种子号 (1-999):", 1, 1, 999)
        if not ok:
            return
        direction, ok = QInputDialog.getItem(self, "添加素材", "方向(左右南北):",
                                             list(DIRECTIONS), 0, False)
        if not ok:
            return
        door, ok = QInputDialog.getText(self, "添加素材", "门特征(如 红门/T门/1门):")
        if not ok or not door.strip():
            return
        floor, ok = QInputDialog.getItem(self, "添加素材", "楼层:",
                                         list(FLOORS), 2, False)
        if not ok:
            return
        target_name = f"{seed} {direction}-{door.strip()}{floor}.png"
        target = self.dir_path / target_name
        if target.exists():
            QMessageBox.warning(self, "已存在", f"文件已存在：{target_name}")
            return
        try:
            shutil.copy2(path, target)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "失败", f"复制失败：{e}")
            return
        self._reload()
        QMessageBox.information(self, "完成", f"已添加：{target_name}")

    def _delete(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选中一行")
            return
        fname = self.table.item(row, 3).text()
        if QMessageBox.question(self, "确认", f"删除素材 {fname} ?") != QMessageBox.StandardButton.Yes:
            return
        try:
            (self.dir_path / fname).unlink()
        except OSError as e:
            QMessageBox.critical(self, "失败", f"删除失败：{e}")
            return
        self._reload()

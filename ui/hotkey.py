# -*- coding: utf-8 -*-
"""
全局快捷键模块
==============
用 Windows 原生 RegisterHotKey + Qt 原生事件过滤器实现全局热键。
热键回调运行在 Qt 主线程的事件循环里，可直接更新 UI（无跨线程问题）。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

WM_HOTKEY = 0x0312

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004

_user32 = ctypes.windll.user32


def _register_hotkey(hid: int, vk: int, mods: int = 0) -> bool:
    return bool(_user32.RegisterHotKey(None, hid, mods, vk))


def _unregister_hotkey(hid: int) -> bool:
    return bool(_user32.UnregisterHotKey(None, hid))


class GlobalHotkeyFilter(QAbstractNativeEventFilter):
    """拦截 WM_HOTKEY 消息，转成回调。回调在主线程执行。"""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def nativeEventFilter(self, eventType, message):
        try:
            msg = wintypes.MSG.from_address(int(message))
        except Exception:  # noqa: BLE001
            return False, 0
        if msg.message == WM_HOTKEY:
            self._callback(msg.wParam)
            return True, 0
        return False, 0


class HotkeyManager:
    """管理一个或多个全局热键。"""

    def __init__(self):
        self._ids: dict[int, int] = {}  # hotkey_id -> virtual_key
        self._filter: GlobalHotkeyFilter | None = None
        self._next_id = 1

    def register(self, vk: int, callback, mods: int = 0) -> int:
        """注册一个热键，返回 hotkey_id。mods 为 MOD_* 组合（如 MOD_CONTROL|MOD_SHIFT）。"""
        hid = self._next_id
        self._next_id += 1
        self._ids[hid] = (vk, mods)
        # 先建好过滤器再注册（回调里需要能查到 id）
        if self._filter is None:
            self._filter = GlobalHotkeyFilter(self._on_hotkey)
            from PySide6.QtWidgets import QApplication
            QApplication.instance().installNativeEventFilter(self._filter)
        if not _register_hotkey(hid, vk, mods):
            self._ids.pop(hid, None)
            raise RuntimeError(f"热键注册失败（vk={vk}），可能已被占用")
        # 记录回调
        self._callbacks = getattr(self, "_callbacks", {})
        self._callbacks[hid] = callback
        return hid

    def _on_hotkey(self, hid: int):
        cb = getattr(self, "_callbacks", {}).get(hid)
        if cb:
            cb()

    def unregister_all(self):
        for hid in self._ids:
            try:
                _unregister_hotkey(hid)
            except Exception:  # noqa: BLE001
                pass
        self._ids.clear()


if __name__ == "__main__":
    # 自检：注册 F8，打印说明（不实际触发）
    import win32con
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    mgr = HotkeyManager()
    hid = mgr.register(win32con.VK_F8, lambda: print("F8 触发"))
    print("已注册 F8 热键，id =", hid)

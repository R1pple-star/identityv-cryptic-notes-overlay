# -*- coding: utf-8 -*-
"""
全局快捷键模块
==============
用 Windows 原生 RegisterHotKey + Qt 原生事件过滤器实现全局热键。
热键回调运行在 Qt 主线程的事件循环里，可直接更新 UI（无跨线程问题）。

注册失败（如热键已被占用）抛 RuntimeError，含 Win GetLastError 中文含义，
供 app.py 醒目提示（状态灯 + 弹窗），不再静默吞。回调异常也兜底，不冒泡到
nativeEventFilter（否则 Qt 静默吞，用户毫无感知）。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

WM_HOTKEY = 0x0312

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

# use_last_error=True 才能用 ctypes.get_last_error() 取到真实的 Win GetLastError。
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
_user32.RegisterHotKey.restype = ctypes.c_int
_user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.UnregisterHotKey.restype = ctypes.c_int


def _winerr_msg(code: int) -> str:
    """常见 RegisterHotKey 错误码中文含义（不全，够用）。"""
    return {
        1409: "热键已被其他程序占用",
        1639: "修饰键参数无效",
        87: "参数错误",
        5: "拒绝访问",
    }.get(code, f"Win 错误码 {code}")


def _register_hotkey(hid: int, vk: int, mods: int = 0) -> tuple[bool, int]:
    """注册；返回 (ok, lasterr)。失败时 lasterr = GetLastError。"""
    ok = bool(_user32.RegisterHotKey(None, hid, mods, vk))
    return ok, (0 if ok else ctypes.get_last_error())


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
        self.last_callback_error: str | None = None  # 回调异常兜底（双保险）

    def register(self, vk: int, callback, mods: int = 0) -> int:
        """注册一个热键，返回 hotkey_id。失败抛 RuntimeError（含 Win 错误含义）。

        mods 为 MOD_* 组合（如 MOD_CONTROL | MOD_SHIFT）。
        """
        hid = self._next_id
        self._next_id += 1
        self._ids[hid] = (vk, mods)
        # 先建好过滤器再注册（回调里需要能查到 id）
        if self._filter is None:
            self._filter = GlobalHotkeyFilter(self._on_hotkey)
            from PySide6.QtWidgets import QApplication
            QApplication.instance().installNativeEventFilter(self._filter)
        ok, err = _register_hotkey(hid, vk, mods)
        if not ok:
            self._ids.pop(hid, None)
            raise RuntimeError(f"热键注册失败：{_winerr_msg(err)}")
        # 记录回调
        self._callbacks = getattr(self, "_callbacks", {})
        self._callbacks[hid] = callback
        return hid

    def _on_hotkey(self, hid: int):
        cb = getattr(self, "_callbacks", {}).get(hid)
        if cb:
            try:
                cb()
            except Exception:  # noqa: BLE001  回调异常不冒泡到 nativeEventFilter（Qt 会静默吞）
                import sys
                import traceback
                self.last_callback_error = traceback.format_exc()
                # 主兜底在 app._entrance_pipeline 的顶层 try/except；这里只防它之外的异常
                print("[hotkey] 回调异常:", self.last_callback_error, file=sys.stderr)

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

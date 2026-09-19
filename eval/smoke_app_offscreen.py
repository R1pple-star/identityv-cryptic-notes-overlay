# -*- coding: utf-8 -*-
"""离屏冒烟：offscreen 起 MainWindow，验证面板适配 + 跟随两步的接线不崩。

覆盖：窗口构造 / `_screen_size`（走常驻 mss 单例）/ `_paint_overlay`（投影窗）/
`_seed_track`（跟踪登记）/ `_track_adopt`（重烘焙）/ `_track_lost`（判丢隐藏 + 清旧图）/
`zoom_scale_from_roi` / `detect_fog_panel` 各分辨率分支。
不注册热键、不进 exec()。

用法: python eval/smoke_app_offscreen.py
"""
import os
import sys
import tomllib
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app as appmod  # noqa: E402
from core.vision import detect_fog_panel, load_bgr  # noqa: E402

ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"  {'✓' if cond else '✗'} {name} {detail}")


def n_recv(obj, sig: str) -> int:
    """该信号上挂了几个接收者。PySide6 的 `receivers()` 只认签名串（`2clicked(bool)`）。"""
    try:
        return int(obj.receivers(sig))
    except Exception:  # noqa: BLE001
        return -1


def main():
    QApplication([])
    settings = appmod.load_settings()
    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    lib = appmod.MapLibrary.load(cfg["paths"]["map_library"])
    win = appmod.MainWindow(lib, settings)
    win.show()
    sw, sh = win._screen_size()
    print(f"窗口构造 OK, 屏幕={sw}x{sh}")
    check("_screen_size 走双调用一致（缓存）", (sw, sh) == win._screen_size())

    # 投影窗：贴一张全屏 RGBA
    rgba = np.zeros((sh, sw, 4), np.uint8)
    rgba[100:300, 200:400, 3] = 160
    win._paint_overlay(rgba)
    check("_paint_overlay 建窗并显示", win.overlay is not None and win.overlay.isVisible())
    check("_paint_overlay 记录了 _last_rgba", win._last_rgba is rgba)

    # 跟踪登记：真实截图 + 真实对齐 → _seed_track
    shot_path = ROOT / "captures" / "hotkey_20260916_221826.png"
    if shot_path.exists():
        shot = load_bgr(str(shot_path))
        panel = detect_fog_panel(shot)
        from core.alignment import find_overlay_transform
        info = lib.get(2, "一楼")
        ref = load_bgr(str(info.path))
        r = find_overlay_transform(shot, ref, panel)
        if r is not None:
            win._seed_track(shot, 2, "一楼", info.path, r[0])
            check("_seed_track 建立跟踪", win._track.active,
                  f"s={win._track.s:.3f} pin={None if win._track.pin() is None else '有'}")
            M2 = np.array([[r[0][0, 0], 0, r[0][0, 2] + 20.0],
                           [0, r[0][1, 1], r[0][1, 2]]])
            before = win._track.M
            win._track_adopt(M2, win._track.s, "冒烟")
            check("_track_adopt 过死区后换了变换", win._track.M is M2 and before is not M2)
            win._track_adopt(M2, win._track.s, "冒烟")   # 同值再一次：死区内
            check("_track_adopt 死区内不重复烘焙", win._track.lost == 0)
            win._track_lost("冒烟"); win._track_lost("冒烟")
            check("_track_lost 连丢 2 次 ⇒ 隐藏投影并断根",
                  (not win.overlay.isVisible()) and win._last_rgba is None
                  and not win._track.active, f"lost={win._track.lost}")
            # 2026-09-18 bug①：G 关闭动画帧引起的判丢**不许清 _last_rgba**（地图重开要放回）
            win._paint_overlay(rgba)
            win._track_lost("冒烟", False); win._track_lost("冒烟", False)
            check("_track_lost 地图判关时判丢 ⇒ 隐藏但留住 _last_rgba（G 重开可恢复）",
                  (not win.overlay.isVisible()) and win._last_rgba is rgba)
            # 2026-09-18 bug① 的另一半：本 tick 判「地图关」⇒ 连跟踪都不做（更不判丢）。
            # G 关闭动画那半秒里 confirmed 仍是 True ⇒ _maybe_track 照旧会调 _track_tick。
            win._seed_track(shot, 2, "一楼", info.path, r[0])
            win._panel_prev = None
            win._track_tick("不是 ROI", False)      # 若没早退，这里会因 roi 类型不对而抛异常
            check("_track_tick 地图判关 ⇒ 本 tick 直接返回（不跟踪、不判丢、不动作）",
                  win._track.lost == 0 and win._track.active and win._panel_prev is None)
            # 2026-09-18 bug① 第二半：**雾兜底**说的「开」也不算数 —— 关闭动画帧里导航列
            # NCC 掉到 −0.01（列根本不在）而雾 0.41 ⇒ map_open_from_roi 照样判「开」，
            # 只靠 is_open 拦不住（实机 22:52:35 判丢 → 清图 → 22:52:42「上次没有成功投影」）。
            win._paint_overlay(rgba)
            win._track_lost("冒烟", True, False); win._track_lost("冒烟", True, False)
            check("_track_lost 只有雾兜底说开时判丢 ⇒ 隐藏但留住 _last_rgba（G 重开可恢复）",
                  (not win.overlay.isVisible()) and win._last_rgba is rgba)
            # 主判据说开才是真丢 ⇒ 必须清（否则会把错位置放回来）
            win._paint_overlay(rgba)
            win._track_lost("冒烟", True, True); win._track_lost("冒烟", True, True)
            check("_track_lost 主判据说开时判丢 ⇒ 隐藏并清 _last_rgba（真丢不留旧位置）",
                  (not win.overlay.isVisible()) and win._last_rgba is None)
        else:
            print("  [跳过] 该截图对齐无结果")
    else:
        print("  [跳过] 缺 captures/hotkey_20260916_221826.png")

    # 滑条读数（该帧是全屏）
    if shot_path.exists():
        from core.vision import roi_of, zoom_scale_from_roi
        s_ui, why = zoom_scale_from_roi(roi_of(load_bgr(str(shot_path)), panel), panel)
        check("zoom_scale_from_roi 全屏帧读得出尺度", s_ui is not None, f"s={s_ui} {why}")

    # detect_fog_panel 各分辨率（合成空图只验证分辨率分支）
    for wh in [(1920, 1080), (2560, 1440), (3440, 1440)]:
        fake = np.zeros((wh[1], wh[0], 3), np.uint8)
        print(f"  {wh} -> {detect_fog_panel(fake)}")

    # ---- T2b：主窗重排（5 按钮 + 参考图行 + 状态行 + 滑块）----
    # 覆盖：尺寸 KPI / 5 按钮接线 / combo 搬进对话框后仍是本体（信号没断）/ 日志窗关开不丢日志 /
    #      LED 不占行 / 状态行按像素截断 + 配色 + tooltip / WARN 染色。
    hint = win.sizeHint()
    print(f"主窗 sizeHint = {hint.width()}x{hint.height()}（T2b 目标 ≤210）")
    check("主窗高度 ≤ 210px（T2b 主 KPI）", hint.height() <= 210, f"h={hint.height()}")
    for name in ("btn_onematch", "btn_realign", "btn_swap", "btn_hide", "btn_floor", "btn_options"):
        b = getattr(win, name, None)
        check(f"主窗按钮 {name} 在且已接线",
              b is not None and n_recv(b, "2clicked(bool)") > 0)
    check("btn_realign 文案 = 按此种子重新对齐（用户 09-19 纠正过的字面）",
          "重新对齐" in win.btn_realign.text())
    check("btn_swap 初始置灰（还没匹配过，没有候选）", not win.btn_swap.isEnabled())
    # 参考图对话框：combo 必须是**本体**（`is` 判等），否则 _realign 读的还是主窗那批空壳
    win._show_ref_picker()
    d = win._ref_dialog
    in_dlg = any(d.findChild(type(c), None) is c for c in (win.direction_combo,))
    check("参考图对话框里就是 direction_combo 本体（信号连接没断）", in_dlg)
    d.close(); win._show_ref_picker()
    check("参考图对话框复开复用同一对象（combo 不会被销毁）", win._ref_dialog is d)
    d.close()
    # 改对话框里的门 → _resolve_seed 仍算出种子，且主窗摘要行跟着变
    win.direction_combo.setCurrentText(win.direction_combo.itemText(0))
    win._resolve_seed()
    check("主窗「参考图」摘要行跟种子走",
          win.seed_label.text().startswith("参考图: ") and "▾" in win.seed_label.text(),
          win.seed_label.text()[:28])
    # 日志窗：关掉再开，日志不丢（对象没被销毁）
    win._show_log_window()
    lw = win._log_window
    win._log_step("T2b 冒烟：日志窗保活测试", "INFO")
    n_before = len(win.log_view.toPlainText())
    lw.close(); win._show_log_window()
    check("日志窗 关→开 复用同一对象且日志未丢",
          win._log_window is lw and len(win.log_view.toPlainText()) == n_before)
    lw.close()
    # LED：不占行（10px），文案在 tooltip 里，set_led 仍能用
    win.set_led(False, "冒烟红灯")
    check("LED 缩成 10px 圆点且文案进 tooltip",
          win.led.width() <= 12 and "冒烟红灯" in win.led.toolTip())
    win.set_led(True, "冒烟绿灯")
    # 状态行：按**像素**截断（不是字数）+ tooltip 全文 + WARN 染色
    long_msg = "入口领先仅15%（<0.25）" * 5
    win._log_step(long_msg, "WARN")
    shown = win.status.text()
    check("状态行按像素截断（单行放得下）",
          win._status_fm.horizontalAdvance(shown) <= 262 + 2, f"{len(shown)}字 {win.status.text()[-1]}")
    check("状态行 tooltip = 全文", win.status.toolTip() == long_msg)
    check("WARN 状态行染色（不是 INFO 灰）", "#ffcc66" in win.status.styleSheet())
    check("状态行可点（点了开日志窗）", n_recv(win.status, "2clicked()") > 0)

    # ---- T3：换种子 + 会话内黑名单 ----
    # 用**合成**的入口层排名测「点一下点谁」，把 `_align_to` 打桩成记录器 ——
    # 真对齐本身由 e48 / eval_match 覆盖，这里只测候选与黑名单的逻辑。
    # 候选的 key/floor 从**真引索**取，保证 `key → 种子` 能反解回同一个号（与热键路径一致）。
    cands = []
    for i, sd in enumerate((23, 1, 20)):
        inf = lib.get(sd, "一楼")
        if inf is not None:
            cands.append((0.48 - 0.01 * i, sd, inf.key, inf.floor, 0.4, None))
    check("T3 合成候选 3 个（key/floor 取自真引索，能反解回同一号）",
          len(cands) == 3, str([(c[1], c[2]) for c in cands]))
    win._entrance_res = cands
    win._entrance_et = "侧门"
    win._seed_bl.clear()
    win._seed = 23
    win._tried_seed = 23
    win._sync_swap_btn()
    check("T3 有入口层排名 ⇒ 换种子按钮可用", win.btn_swap.isEnabled())
    calls = []
    real_align_to = win._align_to
    win._align_to = lambda seed, floor, et, label: calls.append((seed, floor, et, label)) or True
    try:
        win._swap_seed()
        check("T3 点一次 ⇒ 拉黑当前种子(23) 并试下一个候选(1)",
              win._seed_bl == {23} and calls[-1][:2] == (1, "一楼"),
              f"bl={sorted(win._seed_bl)} calls={calls}")
        check("T3 主窗「参考图」行跟着换（不再是种子23）",
              win.seed_label.text().startswith("参考图: ") and "种子23" not in win.seed_label.text(),
              win.seed_label.text()[:26])
        check("T3 换完还有候选 ⇒ 按钮仍可用", win.btn_swap.isEnabled())
        win._swap_seed()
        check("T3 连点两次 ⇒ 23 与 1 都拉黑，第三个候选(20)顶上",
              win._seed_bl == {1, 23} and calls[-1][0] == 20, f"bl={sorted(win._seed_bl)}")
        win._swap_seed()
        check("T3 候选耗尽 ⇒ 不再调对齐且按钮置灰（排除表含第三个）",
              len(calls) == 2 and not win.btn_swap.isEnabled() and 20 in win._seed_bl,
              f"calls={len(calls)} bl={sorted(win._seed_bl)}")
    finally:
        win._align_to = real_align_to
    # ★ 回归：`key → 种子` 反解**对不上**时也必须换得完。
    # 这里故意造一个反解不出同一号的 key（「北-1门」实际反解成种子10，候选里却是种子1）——
    # 若黑名单读 `self._seed`（下拉反解），就会拉黑一个不在候选表里的号，候选表永不缩小，
    # 每次点击都重复试同一个候选。黑名单按 `_tried_seed` 记才对。（这条是写 T3 时真撞上的。）
    win._entrance_res = [(0.48, 23, "北-1沙发门", "一楼", 0.4, None),
                         (0.47, 1, "北-1门", "一楼", 0.4, None)]
    win._seed_bl.clear(); win._tried_seed = 23
    calls.clear()
    win._align_to = lambda seed, floor, et, label: calls.append(seed) or True
    try:
        win._swap_seed()   # 试种子1（下拉会被反解成别的号，但黑名单不该受影响）
        win._swap_seed()   # 应判「候选耗尽」，而不是再试一遍种子1
        check("T3 ★ key→种子 反解对不上时也换得完（黑名单按实际试过的种子记）",
              win._seed_bl == {23, 1} and calls == [1] and not win.btn_swap.isEnabled(),
              f"bl={sorted(win._seed_bl)} calls={calls}")
    finally:
        win._align_to = real_align_to
    # 热键路径清空黑名单。`_capture` 打桩成 False ⇒ 清完之后立刻早退，
    # 既不真截屏（不往 captures/ 里扔垃圾图），又恰好把「先清、后截」这个顺序钉住。
    real_capture = win._capture
    win._capture = lambda: False
    try:
        win._entrance_pipeline_impl()
    finally:
        win._capture = real_capture
    check("T3 按热键 ⇒ 黑名单清空（用户在按之前就清了，不依赖后面走多远）",
          win._seed_bl == set(), f"bl={sorted(win._seed_bl)}")

    # ---- T4：换楼层（困难模式 一楼⇄二楼 直切 + 入口锚点同步）----
    # 同样把 `_align_to` 打桩成记录器 —— 真对齐由 e48 / eval_match 覆盖，这里测
    # 「切到哪层、锚点用哪个入口、下拉/摘要行/按钮文案跟不跟」。
    inf23 = lib.get(23, "一楼")
    win._select_ref(inf23.key, "一楼")          # 确定性起点：种子23 在一楼
    win._entrance_et = "侧门"; win.entrance_combo.setCurrentText("侧门")
    win._sync_floor_btn()
    check("T4 一楼时按钮指向→二楼且可用", "→二楼" in win.btn_floor.text() and win.btn_floor.isEnabled(),
          win.btn_floor.text())
    fcalls = []
    win._align_to = lambda seed, floor, et, label: fcalls.append((seed, floor, et, label)) or True
    try:
        win._switch_floor()
        check("T4 一楼→二楼：_align_to(种子23, 二楼, 锚点=二楼, 换楼层) 且入口下拉同步",
              fcalls[-1] == (23, "二楼", "二楼", "换楼层")
              and win.floor_combo.currentText() == "二楼"
              and win.entrance_combo.currentText() == "二楼",
              f"calls={fcalls} floor={win.floor_combo.currentText()} et={win.entrance_combo.currentText()}")
        check("T4 切层后摘要行带二楼、按钮翻成→一楼",
              "二楼" in win.seed_label.text() and "→一楼" in win.btn_floor.text(),
              win.seed_label.text()[:26])
        win._switch_floor()                     # 回一楼：_entrance_et=侧门 ⇒ 锚点沿用侧门
        check("T4 二楼→一楼：_align_to(…, 一楼, 锚点=侧门) 且 combo 复位",
              fcalls[-1][1] == "一楼" and fcalls[-1][2] == "侧门"
              and win.floor_combo.currentText() == "一楼"
              and win.entrance_combo.currentText() == "侧门",
              f"calls={fcalls}")
        win._entrance_et = "二楼"               # 热键在二楼入口匹配过 ⇒ 回一楼锚点兜底正门
        win._switch_floor(); win._switch_floor()
        check("T4 _entrance_et=二楼 时回一楼 ⇒ 锚点兜底「正门」（二楼锚点配一楼图必错）",
              fcalls[-1][1] == "一楼" and fcalls[-1][2] == "正门", f"calls={fcalls}")
        # 换种子必须**保楼层**：二楼上点换种子，不能被候选自带的 fl(一楼) 拽回去
        win._entrance_et = "侧门"
        win._entrance_res = [(0.48, 23, inf23.key, "一楼", 0.4, None),
                             (0.47, 1, lib.get(1, "一楼").key, "一楼", 0.4, None)]
        win._seed_bl.clear(); win._tried_seed = 23
        win._sync_swap_btn()
        win._switch_floor()                     # 现在在二楼
        win._swap_seed()
        check("T4 ★ 二楼上点换种子 ⇒ 仍对齐二楼、锚点=二楼（不被候选的 fl 拽回一楼）",
              fcalls[-1][:3] == (1, "二楼", "二楼") and win.floor_combo.currentText() == "二楼",
              f"calls={fcalls}")
    finally:
        win._align_to = real_align_to
    win._seed = None; win._sync_floor_btn()
    check("T4 种子未定 ⇒ 换楼层按钮置灰", not win.btn_floor.isEnabled())
    win._select_ref(inf23.key, "一楼")          # 复位，别影响后面的段落

    # ---- T2a：样本预览窗自动消失（2026-09-19）----
    # 覆盖：到点消失 / 光标还在窗上时续期 / 拖动时续期 / 0 秒=不消失（旧行为）/
    #      hide→show 复用同一 HWND（钉死 WDA_EXCLUDEFROMCAPTURE 不丢）。
    bgr = np.zeros((120, 160, 3), np.uint8)
    win.settings.sample_preview_sec = 5.0
    win._show_sample_preview(bgr)
    pv = win.preview
    check("样本预览：set_sample 后倒计时启动且窗可见",
          pv is not None and pv._timer.isActive() and pv.isVisible())
    wid = int(pv.winId())
    pv.hide(); pv.show()
    check("预览窗 hide→show 复用同一 HWND（不丢截屏排除 affinity）", int(pv.winId()) == wid)
    check("_mouse_inside 返回 bool（真实实现，非打桩）", isinstance(pv._mouse_inside(), bool))
    real_inside = pv._mouse_inside
    try:
        pv._mouse_inside = lambda: True          # 打桩：光标还在窗上
        pv._on_timeout()
        check("到点但光标还在窗上 ⇒ 续期不隐藏", pv.isVisible() and pv._timer.isActive())
        pv._mouse_inside = lambda: False
        pv._drag = (0, 0)                        # 正在拖动
        pv._on_timeout()
        check("到点但正在拖动 ⇒ 续期不隐藏", pv.isVisible() and pv._timer.isActive())
        pv._drag = None
        pv._on_timeout()
        check("到点且光标不在窗上、没在拖 ⇒ 隐藏", not pv.isVisible())
    finally:
        pv._mouse_inside = real_inside
    pv._on_close_clicked()
    check("✕ 关闭 ⇒ 隐藏且停表", (not pv.isVisible()) and (not pv._timer.isActive()))
    win.settings.sample_preview_sec = 0.0
    win._show_sample_preview(bgr)
    check("sample_preview_sec=0 ⇒ 不启动倒计时（旧行为：一直留着）",
          pv.isVisible() and not pv._timer.isActive())
    pv.hide()

    print(f"SMOKE {sum(ok)}/{len(ok)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

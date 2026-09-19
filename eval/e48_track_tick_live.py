# -*- coding: utf-8 -*-
"""跟踪主体（`app._maybe_track` → `app._track_tick`）的**真代码**回归。

**为什么必须造这个工具**：跟随·第二步原有的三个回归里，`e33_follow_track.py` 与
`e45_zoom_follow_replay.py` 都是**复刻** `_track_tick` 的逻辑（各自维护一份副本、各自
`from ui.track import …`），而 `smoke_app_offscreen.py` 那句
`win._track_tick("不是 ROI", False)` 被 `is_open=False` 在第一行早退 ⇒
**真代码的跟踪主体从来没有被执行过一次**。

代价在 2026-09-19 现形：`ui.track.TRACK_DEADBAND_S` / `TRACK_OVERLAP_MIN` 与
`ui.follow.FOLLOW_MAX_CAPTURE_ERRORS` **都没进 app.py 的 import 列表** ⇒ 自 `1b888e7`
（跟随·第二步）起，每次「帧差过闸」都在写第一行日志之前抛 `NameError`
（`app.py:993`），traceback 只进 stderr。实机日志（15:37–15:47）的症状是
**56 次过闸、零条输出**（14 条心跳的 `本轮峰值≥1.2 / N tick` 相加），而三条静默路径
在日志上完全同形 ⇒ 只能靠猜。这与技术备忘③ 是同一个坑：
**「离线通过、实机不行」时先问「实机走的是不是另一条分支」**。

本脚本用**用户当场那批截图**（`captures/hotkey_20260919_1541xx.png` = 15:41/15:42 那三次
热键）喂真代码：播种走与热键同源的 `hint_icon` 图标锚定，之后逐帧调 `_maybe_track`，
断言 ①零异常 ②真的跑到了判据（留下 `小窗`/`全平移` 心跳）③本次跟随的第一 tick
（`mad is None`）也不炸。

用法: python eval/e48_track_tick_live.py
"""
import os
import sys
import tomllib
import traceback
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app as appmod  # noqa: E402
from core.alignment import find_overlay_transform  # noqa: E402
from core.entrance import find_seed_by_entrance, load_index  # noqa: E402
from core.vision import _find_icon, detect_fog_panel, load_bgr, map_roi  # noqa: E402

SEED, FLOOR = 7, "一楼"
FRAMES = ["hotkey_20260919_154124.png", "hotkey_20260919_154201.png",
          "hotkey_20260919_154208.png"]

LOGS = []
BAD = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name} {detail}")
    if not cond:
        BAD.append(name)


def capture_like_app(shot, panel):
    """复刻 `ui.capture.capture_region` 的返回：`bgra[:, :, :3]` —— **步长 4 的视图**。

    必须保持一致：真实路径给出的就是这种非连续视图。用连续 BGR 测等于换了输入
    （`_find_icon` / `to_match3` / `matchTemplate` 吃的都是它）。
    """
    x0, y0, x1, y1 = map_roi(panel)
    bgra = cv2.cvtColor(shot[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
    roi = bgra[:, :, :3]
    assert roi.strides[1] == 4, roi.strides
    return roi


def seed_transformer(shot, ref, panel, idx):
    """与 `app._entrance_pipeline_impl` 同源的播种：图标锚定钉平移 + 全尺度扫。"""
    ip, isc, _k = _find_icon(shot, *panel)
    # 与 `_entrance_pipeline_impl` 同源：图标锚定（钉平移、全尺度窗扫）。播种必须落在**有效**
    # 对齐上（分<0.30 且 ov≥0.40），否则 `tr.s` 是错的 ⇒ 每 tick 都 `scale_moved` ⇒
    # 只走全平移、`小窗` 分支永远没被覆盖。锚定都不行时退回全搜（= 实机的隐式兜底）。
    cands = []
    for ent, h in idx.items():
        if not isinstance(h, dict) or "cx" not in h:   # 引索里还有 width/height 这类标量键
            continue
        a = find_overlay_transform(shot, ref, panel, hint_icon=(h["cx"], h["cy"], ip[0], ip[1]))
        if a is not None:
            cands.append((a[1], a))
    full = find_overlay_transform(shot, ref, panel)
    if full is not None:
        cands.append((full[1], full))
    valid = [c for c in cands if c[1][1] < 0.30 and c[1][2] >= 0.40]
    best = min(valid or cands, key=lambda c: c[0])[1]
    return best, (ip[0], ip[1], isc)


def main():
    QApplication([])
    settings = appmod.load_settings()
    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    lib = appmod.MapLibrary.load(cfg["paths"]["map_library"])
    win = appmod.MainWindow(lib, settings)
    win.show()
    win._log_step = lambda msg, level="INFO": (LOGS.append(msg), print(f"  [LOG] {msg}"))

    shot0 = load_bgr(str(ROOT / "captures" / FRAMES[0]))
    panel = detect_fog_panel(shot0)
    check("面板 = 实机面板", panel == (668, 166, 1064, 569), f"{panel}")
    win._follow_panel = panel

    ref_path = lib.get(SEED, FLOOR).path
    ref = load_bgr(str(ref_path))
    idx = load_index(SEED)
    a, ip = seed_transformer(shot0, ref, panel, idx)
    print(f"  播种：图标 @{ip[0]},{ip[1]} 分{ip[2]:.2f}｜对齐 s={a[0][0, 0]:.3f} "
          f"分{a[1]:.3f} ov{a[2]:.2f}（实机 15:41 那次：图标@(932,426) 分0.79 "
          f"重合分0.038 重叠1.00）")
    win._seed_track(shot0, SEED, FLOOR, ref_path, a[0])

    print("\n1) 走完整接线 `_maybe_track`（不是直接调 _track_tick，让 _track_arm/active 也进范围）")
    win._follow.confirmed = True
    win._track_arm = False
    n_exc = 0
    for f in FRAMES[1:]:
        piece = load_bgr(str(ROOT / "captures" / f))
        roi = capture_like_app(piece, panel)
        for label, prev in (("首 tick（mad=None）", None),
                            ("刷新后 tick", np.zeros((34, 64, 3), np.int16))):
            win._panel_prev = prev      # 首 tick 复现 `_start_follow` 的清零；后者强制过闸
            win._track_beat_t = 0.0
            print(f"\n  --- {f} / {label} ---")
            try:
                win._maybe_track(roi, True, True)
            except Exception:
                n_exc += 1
                print("  ‼ 抛异常：")
                traceback.print_exc()

    check("跟踪主体零异常（旧版这里每次过闸都 NameError: TRACK_DEADBAND_S）", n_exc == 0,
          f"n_exc={n_exc} app._track_err={win._track_err}")
    check("app 的异常兜底没被触发", win._track_err == 0, f"_track_err={win._track_err}")

    print("\n2) 判据真的跑到了")
    decided = [m for m in LOGS if ("小窗" in m or "全平移" in m or "判丢" in m)]
    for m in decided:
        print(f"     {m}")
    check("留下 ≥1 条「小窗/全平移/判丢」心跳（= 主体跑到了判据，不是死在半路）",
          len(decided) >= 1, f"{len(decided)} 条")
    check("没有「跟踪内部异常」日志", not any("跟踪内部异常" in m for m in LOGS))

    # ---- 3) 「按此种子对齐」（`_realign`，2026-09-19 bug③）的真代码 ----
    # 用户原话：「点『按此种子对齐』按钮对齐也是不准的，也许我们可以把匹配时的对齐方式
    # 用在其他跟随和对齐的时候。」两处病根：①拿的是上一次热键的**旧图**；②无锚点无尺度
    # 提示的全搜 ⇒ 缩模板骗分（实测本批帧全搜得 s≈0.39 而真值 s≈0.83，分 0.038 照样过闸）。
    print("\n3) 手动重对齐（`_realign`）：旧图+全搜 vs 当场截图+两段式")
    # 引用必须换成这三帧真正的种子（实机 15:41/15:42 都是「种子7 南-三缺一门[一楼]」）。
    win._seed = SEED
    win.floor_combo.setCurrentText(FLOOR)

    # 参照物 = **热键路径本尊**：`_two_stage_align(shot, 入口层结果, 真入口, 图标位)`。
    # 实机 15:41 那次它给出「重合分0.038 重叠1.00」＝用户认可的"对"。断言 `_realign` 在同一帧
    # 复现同一位置 —— 这就是用户原话「把匹配时的对齐方式用在其他对齐的时候」的字面验收。
    #
    # 真入口 = **让入口层自己说**（top1 是种子7 且墙重合最好者），与 `_realign` 内部路径无关，
    # 不循环。⚠️ 参照必须**逐帧各算**：地图在帧间挪过，拿第一帧的位置去比第二帧只会比出位移。
    ent_types = [k for k, v in idx.items() if isinstance(v, dict) and "cx" in v]
    cp = (ref.shape[1] / 2.0, ref.shape[0] / 2.0)     # 参考图中心：独立于任何锚点

    def pick_entrance(sh, pl):
        """入口层定真入口：top1 是种子7 且墙重合最低者。返回 (ent, 入口层结果, 图标位)。"""
        best = None
        for et in ent_types:
            res, ipos, _i, _k = find_seed_by_entrance(sh, lib, et, panel=pl, top_n=3)
            if res and res[0][1] == SEED and (best is None or res[0][0] < best[1][0]):
                best = (et, res[0], ipos)
        return best

    def center(M):
        M = np.asarray(M)
        return (float(M[0, 0] * cp[0] + M[0, 2]), float(M[1, 1] * cp[1] + M[1, 2]))

    stale = load_bgr(str(ROOT / "captures" / FRAMES[0]))
    print(f"   {'帧':<30}{'旧（旧图+全搜）':>18}{'新（当场截图+两段式）':>22}{'参照=热键本尊':>16}")
    n_fresh = n_close = n_ref = 0
    for f in FRAMES:
        sh = load_bgr(str(ROOT / "captures" / f))
        pl = detect_fog_panel(sh)
        hit = pick_entrance(sh, pl)
        if hit is None:
            print(f"   {f}: 入口层没把种子7 排第一 ⇒ 跳过"); continue
        et, best_ent, icon_ent = hit
        n_ref += 1
        hot = win._two_stage_align(sh, best_ent, et, icon_ent)     # 热键路径本尊（该帧）
        if hot is None:
            print(f"   {f}: 热键路径无结果 ⇒ 跳过"); continue
        hot_p = center(hot[0])
        r_old = find_overlay_transform(stale, ref, pl)             # 旧行为那两句
        c_old = center(r_old[0]) if r_old is not None else (float("nan"),) * 2
        d_old = float(np.hypot(c_old[0] - hot_p[0], c_old[1] - hot_p[1]))
        win.entrance_combo.setCurrentText(et)
        win._shot = stale                                          # 先塞旧图
        win._capture = lambda sh=sh: (setattr(win, "_shot", sh), True)[1]
        win._realign()
        n_fresh += win._shot is sh                                 # ① 真的重新截屏了
        c_new = center(win._track.M) if win._track.active else (float("nan"),) * 2
        d_new = float(np.hypot(c_new[0] - hot_p[0], c_new[1] - hot_p[1]))
        close = d_new <= 25.0                                      # 项目既定的「对齐良定」容差
        n_close += close
        print(f"   {f:<30}{f'{d_old:.0f}px':>18}"
              f"{(f'{d_new:.0f}px{chr(10003) if close else chr(10007)}'):>22}"
              f"{f'{hot[1]:.3f}/{hot[2]:.2f} 入口={et}':>16}")
    check("① 每次都当场重新截屏（旧写法只在 `_shot is None` 时截）",
          n_fresh == len(FRAMES), f"{n_fresh}/{len(FRAMES)}")
    check("② `_realign` 复现该帧热键路径的位置（参考图中心差 ≤25px）；旧行为那列作对照",
          n_ref > 0 and n_close == n_ref, f"{n_close}/{n_ref}")

    print(f"\n{'=' * 60}\nE48 {'PASS' if not BAD else 'FAIL ' + str(BAD)}（异常 {n_exc}）")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())

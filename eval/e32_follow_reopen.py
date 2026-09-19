# -*- coding: utf-8 -*-
"""跟随开合链路的离线断言（2026-09-17 改动）：真截图上跑「开→关→再开」全序列。

复刻 app._follow_tick 的判定链（map_open_from_roi → FollowState.feed），用真实像素验证：
  1. 开态图判「开」、关态图判「关」（判据本身）；
  2. 序列 adopt(开) → flip(关) → flip(开)，即 G 关掉再开回来时**状态机能回到开**
     （旧版 app 在这中间把轮询停了 ⇒ 永远收不到最后那个 flip(开)，用户报的就是这个）；
  3. 两次「开」之间换一张不同的开态图也能 flip 回来（不依赖同一张图）。

用法: python eval/e32_follow_reopen.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.vision import (FIXED_PANEL, load_bgr, map_open_from_roi, panel_for_screen,
                         roi_of)
from ui.follow import FOLLOW_CONFIRM_N, FollowState

NVIDIA = Path(r"D:\Videos\NVIDIA\IdentityV")
C = ROOT / "captures"
OPEN1 = C / "hotkey_20260916_221826.png"
OPEN2 = C / "hotkey_20260916_222657.png"
CLOSED1 = NVIDIA / "IdentityV Screenshot 2026.08.25 - 20.02.47.44.png"   # 游戏世界·木楼梯
CLOSED2 = NVIDIA / "IdentityV Screenshot 2026.09.14 - 14.46.48.17.png"   # 游戏世界·走廊绿植

PANEL = panel_for_screen(1920, 1080)
ok = []


def judge(p):
    shot = load_bgr(str(p))
    is_open, why, _nav_ok = map_open_from_roi(roi_of(shot, PANEL), PANEL)
    return is_open, why


def check(name, cond, detail=""):
    ok.append(cond)
    print(f"  {'✓' if cond else '✗'} {name} {detail}")


print("1) 判据本身（单张）")
for p, want in ((OPEN1, True), (OPEN2, True), (CLOSED1, False), (CLOSED2, False)):
    got, why = judge(p)
    check(f"{p.name[-30:]:<32} 期望{'开' if want else '关'}", got == want, f"→ {why}")

print("\n2) 开→关→再开 全序列（每态连喂 FOLLOW_CONFIRM_N 帧，模拟防抖）")
st = FollowState()
seq = [(OPEN1, "开①"), (CLOSED1, "关①"), (OPEN2, "开②")]
events = []
for p, tag in seq:
    is_open, why = judge(p)
    for _ in range(FOLLOW_CONFIRM_N):
        e = st.feed(is_open)
        if e is not None:
            events.append((e[0], e[1]))
    print(f"    喂 {tag:<5} 判{'开' if is_open else '关'}（{why}）")
check("事件序列 = adopt(开) → flip(关) → flip(开)",
      events == [("adopt", True), ("flip", False), ("flip", True)], f"实得 {events}")
check("末态 confirmed = 开（G 开回来能被察觉）", st.confirmed is True)

print("\n3) 关态连喂不抖动（同一关态图重复喂，不应反复翻转）")
st2 = FollowState()
ev2 = []
for _ in range(10):
    for _ in range(FOLLOW_CONFIRM_N):
        e = st2.feed(judge(CLOSED1)[0])
    if e is not None:
        ev2.append(e)
check("全程只 adopt 一次、无 flip", ev2 == [("adopt", False)], f"实得 {ev2}")

print(f"\n跟随开合链路: {sum(ok)}/{len(ok)} 通过")
sys.exit(0 if all(ok) else 1)

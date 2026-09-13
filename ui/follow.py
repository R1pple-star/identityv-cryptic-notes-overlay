# -*- coding: utf-8 -*-
"""
自动跟随·第一步：投影显隐跟随游戏内地图开合
==========================================
纯逻辑状态机（零 Qt 依赖，离线断言见 experiments/e3_follow_state_offline.py）。
app.py 用 QTimer 喂帧：tick 只截面板区 → map_is_open 取 content 占比
> FOLLOW_OPEN_THRESH 判开/关 → feed() → 连续 FOLLOW_CONFIRM_N 帧同判才确立/翻转
（吸收 G 开合动画的过渡帧）。

暂停语义（suspended）：
- 手动点 btn_hide（跟随开启时）→ suspend()，用户意图保持；
- 状态翻转（flip）自动清挂起，跟随接管（两方向手动覆盖均平滑衔接）；
- 首态确立（adopt）不清挂起（防跟随刚启动就覆盖用户意图）；
- 热键/重对齐出新投影 → reset() 全清（热键=「我现在要投影」）。

阈值依据（experiments/e2_map_open_calib.py, 2026-09-13）：开态 53 张 content
min=0.327 / 中位 0.469；关态 4 张 max=0.123；分离带 [0.123, 0.327] 宽 0.204 →
取中点 0.225。关态样本少（仅 4 张同场景），待补拍 captures/follow_closed/ 后复标。
"""
from __future__ import annotations

FOLLOW_INTERVAL_MS = 250          # 轮询周期（只截面板小图，主线程占空比低）
FOLLOW_CONFIRM_N = 2              # 连续同判帧数（防抖；最坏响应 ≈ 2 tick + 相位 ≈ 0.75s）
FOLLOW_OPEN_THRESH = 0.225        # 开闸阈值（e2 标定，见模块头）
FOLLOW_MAX_CAPTURE_ERRORS = 10    # 连续截屏异常上限，达到即停跟随


class FollowState:
    """feed() 返回 None=无事件；("adopt", state)=首态确立；("flip", state)=状态翻转。"""

    def __init__(self, confirm_n: int = FOLLOW_CONFIRM_N):
        self.confirm_n = max(1, int(confirm_n))
        self.confirmed: bool | None = None  # 已确认的开合态；None=未确立
        self.suspended = False              # 手动覆盖挂起（flip 时自动清）
        self._pending: bool | None = None
        self._count = 0

    def feed(self, is_open: bool):
        if self._pending != is_open:
            self._pending, self._count = is_open, 1
        else:
            self._count += 1
        if self._count < self.confirm_n:
            return None
        self._pending, self._count = None, 0
        if self.confirmed is None:
            self.confirmed = is_open
            return ("adopt", is_open)
        if is_open == self.confirmed:
            return None  # 抖动后回到已确认态：无事件
        self.confirmed = is_open
        self.suspended = False  # 翻转即接管，清手动挂起
        return ("flip", is_open)

    def suspend(self) -> None:
        self.suspended = True

    def reset(self) -> None:
        """全清（新投影 / 跟随启停时用）。"""
        self.confirmed = None
        self.suspended = False
        self._pending = None
        self._count = 0

    def idle_reset(self) -> None:
        """只清待确认计数（overlay 不存在的空窗期，防陈旧 pending 跨越）。"""
        self._pending = None
        self._count = 0

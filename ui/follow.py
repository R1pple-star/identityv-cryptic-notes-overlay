# -*- coding: utf-8 -*-
"""
自动跟随·第一步：投影显隐跟随游戏内地图开合
==========================================
纯逻辑状态机（零 Qt 依赖，离线断言见 experiments/e3_follow_state_offline.py）。
app.py 用 QTimer 喂帧：tick 只截面板小图 → follow_features 取 雾色占比/结构占比
→ 双特征 OR 判开/关 → feed() → 连续 FOLLOW_CONFIRM_N 帧同判才确立/翻转
（吸收 G 开合动画的过渡帧）。

轮询边界（2026-09-14）：**只在投影可见期间轮询**（app._follow_tick 早退）——
投影隐藏即停止一切截屏检测，游戏中 NVIDIA 截图可正常用；重开地图想看投影
按热键重新匹配或点「显示地图」。轮询只在显示期间用于自动隐藏。

暂停语义（suspended）：
- 手动点 btn_hide 隐藏（跟随开启时）→ suspend()，此后 G 开合**不再自动唤起**投影；
- 挂起中状态翻转只更新 confirmed、不返回事件（跟随知情但不驱动投影）；
- 恢复途径：点「显示地图」（resume()）或热键/重对齐出新投影（reset() 全清）；
- 首态确立（adopt）不清挂起（防跟随刚启动就覆盖用户意图）。

判定阈值（2026-09-14 实测，63 开态 + 4 疑似关态）：
- struct = (cls2房间|cls3通路) 占比：开态 min 0.21 / med 0.44，关态 max 0.017
  → 阈值 0.10 取分离带中。旧 content(亮度≥42占比) 单特征已证不可用：关态游戏画面
  （Alt/F 键特效）实测冲到 0.31，跨过旧阈值 0.225 误开关投影。
- fog = FOG_BGR tol24 色距占比：关态 ≤0.002；重度探明开态可低至 0.001（雾开完了）
  → fog 只作第二特征与 struct OR，覆盖两端（雾多/结构多任一显著即判开）。
- 复标：补拍关态截图（含 Alt/F 误触发场景）进 captures/follow_closed/ 后跑
  experiments/e2_map_open_calib.py。
"""
from __future__ import annotations

FOLLOW_INTERVAL_MS = 250          # 轮询周期（只截面板小图，主线程占空比低；仅投影可见期间运行）
FOLLOW_CONFIRM_N = 2              # 连续同判帧数（防抖；最坏响应 ≈ 2 tick + 相位 ≈ 0.75s）
FOLLOW_FOG_THRESH = 0.10          # 雾色占比开闸阈值（e2 复标口径，见模块头）
FOLLOW_STRUCT_THRESH = 0.10       # 结构(cls2|cls3)占比开闸阈值（开态min0.21/关态max0.017 取中）
FOLLOW_MAX_CAPTURE_ERRORS = 10    # 连续截屏异常上限，达到即停跟随


class FollowState:
    """feed() 返回 None=无事件；("adopt", state)=首态确立；("flip", state)=状态翻转。"""

    def __init__(self, confirm_n: int = FOLLOW_CONFIRM_N):
        self.confirm_n = max(1, int(confirm_n))
        self.confirmed: bool | None = None  # 已确认的开合态；None=未确立
        self.suspended = False              # 手动覆盖挂起（恢复靠 resume()/reset()，flip 不再自动清）
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
        if self.suspended:
            return None  # 挂起中：跟随知情（confirmed 持续更新）但不驱动投影
        return ("flip", is_open)

    def suspend(self) -> None:
        self.suspended = True

    def resume(self) -> None:
        """解除挂起（点『显示地图』）。confirmed 已在挂起期间持续更新，无陈旧翻转。"""
        self.suspended = False

    def reset(self) -> None:
        """全清（新投影 / 跟随启停时用）。"""
        self.confirmed = None
        self.suspended = False
        self._pending = None
        self._count = 0

    def idle_reset(self) -> None:
        """只清待确认计数（overlay 不存在/隐藏的空窗期，防陈旧 pending 跨越）。"""
        self._pending = None
        self._count = 0

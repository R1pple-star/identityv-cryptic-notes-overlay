# -*- coding: utf-8 -*-
"""
自动跟随·第一步：投影显隐跟随游戏内地图开合
==========================================
纯逻辑状态机（零 Qt 依赖）。
app.py 用 QTimer 喂帧：tick 只截**地图 ROI**（面板 ∪ 右侧导航列，见 vision.map_roi）
→ `vision.map_open_from_roi` 判开/关 → feed() → 连续 FOLLOW_CONFIRM_N 帧同判才确立/
翻转（吸收 G 开合动画的过渡帧）。

轮询边界（**两种「隐藏」必须分开**）：
- **跟随判出地图关** → 投影隐藏，但**继续轮询**（降频到 FOLLOW_IDLE_INTERVAL_MS/格），
  于是按 G 把地图开回来时能自动恢复投影（用上次的变换，不重跑匹配）。
- **手动点 btn_hide 隐藏** → suspend()，**停轮询**（明确的用户意图；投影不显示时屏幕检测
  纯属浪费）。恢复途径：点「显示地图」（resume()）或热键/重对齐出新投影（reset() 全清）。
  首态确立（adopt）不清挂起（防跟随刚启动就覆盖用户意图）。

常驻 mss 单例（ui/capture.py）之后，低频轮询不会再干扰 NVIDIA 截图 —— 干扰源是
「每 tick 新建+销毁 DC 句柄」，不是「轮询」本身。

判定阈值（实测见 core/vision.py 的 NAV_BAND 长注）：
开 ⇔ 侧栏导航列 NCC ≥ NAV_NCC_MIN(0.45) **或** 雾占比 ≥ FOG_OPEN_MIN(0.30)。
**结构占比退出判定**：地图关着的游戏画面实测结构占比也能到 0.51 ⇒ 做开判据必误判，
投影留在地图关着的画面上。导航列是「高精度低召回」，所以雾兜底必须留。
"""
from __future__ import annotations

FOLLOW_INTERVAL_MS = 250          # 轮询周期（投影可见期间；只截地图 ROI 小图，主线程占空比低）
FOLLOW_IDLE_INTERVAL_MS = 1000    # 跟随自动隐藏后的降频周期（等 G 把地图开回来；见模块头）
FOLLOW_CONFIRM_N = 2              # 连续同判帧数（防抖；最坏响应 ≈ 2 tick + 相位 ≈ 0.75s）
FOLLOW_MAX_CAPTURE_ERRORS = 10    # 连续截屏异常上限，达到即停跟随

# 开合阈值不在本模块：判定走 vision.map_open_from_roi（NAV_NCC_MIN / FOG_OPEN_MIN）。
# 阈值只有一处可调，避免两套口径漂移。


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

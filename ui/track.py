# -*- coding: utf-8 -*-
"""自动跟随·第二步：投影跟着地图**平移/缩放**，不用再按热键
================================================================
纯逻辑状态机（零 Qt 依赖，离线断言见 experiments/e33_follow_track.py）。
`ui/follow.py` 管「地图开着没」（G 开关），本模块管「地图动到哪了」——两者都在
`app._follow_tick` 里驱动，共用同一张 ROI 截屏。

## 为什么不能「重跑一遍对齐」
全搜要 1.5~2.0 s（25 档粗搜 × 每档一次全平移 + 细搜 + 微搜），塞进 250ms 的 tick
里会冻死主线程。跟踪走的是**局部**：从上一帧那个**已经过闸、已经显示出来**的变换出发，
只问"它还对不对得上、该往哪挪一点"。

## 三档（越往后越贵，前档通过就不做后档）

| 档 | 做什么 | 成本 | 何时走 |
|---|---|---|---|
| `probe` | 在上一帧变换预测的位置上**单点评估**（1×1 响应） | ~0ms（省掉匹配） | 每 tick 先问一句 |
| `full` | 在**单尺度**上做**全平移**搜索 | ~43 ms | 探针分变差（地图动了） |
| `sweep` | 换尺度重做 full（尺度由滑条给出，见下） | 5 档 ~200 ms | 滑条说缩放变了 |

## 防「幽灵相位」：靠**证据边际**而不是窗搜
迷宫走廊网格自相似，格周期只有 **25~26 ref px**，而任何跟得上真实拖动的搜索窗都必须
远大于它 ⇒ **窗搜挡不住幽灵**（实测窗半径 32px 时分数 0.204「过闸」但位置偏 164px，
是个静默错位）。真正能分辨的是**证据强度**：
  真位移时「旧变换在新帧上的分」大幅劣化 —— 实测 0.121→0.430、0.052→0.631、0.086→0.362
  （3.6×/12×/4.2×，即新位置要好 72%~92%）；
  而格内幽灵相对旧位置只好 **8%~15%**。
⇒ 规则：**新位置的分必须 ≤ `TRACK_MARGIN`(0.60) × 旧位置的分**（要求好 40%），
卡在两者中间。探针分本来就够好（< `TRACK_OK`）时不做大跳，只接受原地/微动。

## 尺度从哪来：滑条的实测值 > 对齐分
对齐分**选不出尺度**（CLAUDE.md 技术备忘⑥：分对 s 单调递增、真尺度处不是极小 ⇒ 挑分
最小的尺度典型偏低 0.01~0.02、曲线平坦时偏低 0.09，且恒被下界截住）。而右侧导航列那道
**缩放滑条的圆点**直接给出当前缩放（11 点标定、图标误差 1~22px、全种子通用，见
`vision.zoom_scale_from_roi`）。滑条读不到（窗口化布局）才退回"在上次尺度上扫"。

## 跟丢怎么办（用户 2026-09-17 选定）
连续 `TRACK_LOST_MAX`(2) 次判丢 ⇒ **隐藏投影**并停止更新，日志提示按热键。
理由：项目既定原则是「宁可什么都不给，也不给一张像是对的的假图」（待办 1）——
粘在错位置的投影会被当成对的位置用。**不 suspend 跟随**（地图还开着，跟随继续看着屏幕）。
"""
from __future__ import annotations

import math

from core.alignment import ALIGN_SCALES

# ---- 判据阈值（改动前请先看 experiments/e38_scale_policy.py / e36_scale_band.py）----
TRACK_OK = 0.20           # 探针分低于此值 ⇒ 旧变换仍然对得上，不需要跳
TRACK_ACCEPT = 0.30       # 接受新变换的绝对闸（与 config [match].align_score_max 同源）
TRACK_OVERLAP_MIN = 0.40  # 与 config [match].overlap_min 同源
TRACK_MARGIN = 0.60       # 大跳的证据边际：新分须 ≤ 此值 × 探针分（实测真位移 0.08~0.28×，幽灵 0.85~0.92×）
TRACK_LOST_MAX = 2        # 连续判丢上限 ⇒ 隐藏投影 + 停跟踪
TRACK_DEADBAND_PX = 1.5   # 变换位移小于此值不重烘焙（防抖，省一次 warpAffine + QPixmap）
TRACK_DEADBAND_S = 0.004  # 尺度相对变化死区
TRACK_NEAR_R = 12         # 「上次位置附近」窗搜半径（ref px）。**必须 < 半个迷宫格周期
                          # (25px)**：这样窗内不可能出现幽灵相位，小的平移漂移就能被稳稳
                          # 纠回而不会被骗到大跳。超过它就该走全平移 + 证据边际。
TRACK_MOTION_MAD = 1.2    # 面板降采样帧差低于此值（灰度级）视为「画面没动」⇒ 跳过重对齐
TRACK_SCALE_BAND = 0.05   # 滑条给出的尺度只信到这个半宽；超出就在滑条值周围带内重扫
TRACK_IDLE_INTERVAL_MS = 1000   # 投影没动时的探针周期（动了立刻回到全速）
TRACK_LOG_MIN_SEC = 1.5   # 跟踪成功日志的最小间隔（拖动时别刷屏）
# 合法尺度域（防滑条读数离谱时把搜索带出域）
SCALE_LO, SCALE_HI = float(ALIGN_SCALES[0]), float(ALIGN_SCALES[-1])


def verdict(score_prev, score_new, ov_new):
    """纯判定：返回 `"keep"`（旧位置仍可信，别动）/ `"accept"`（用新变换）/ `"lost"`。

    score_prev: 旧变换在新帧上的分（探针；None = 预测位置落在参考图外，问不出来）
    score_new : 单尺度全平移搜到的最优分（None = 该尺度放不下/无解）
    ov_new    : 对应重叠
    """
    if score_prev is not None and score_prev < TRACK_OK:
        return "keep"          # 旧位置本来就够好 ⇒ 地图没怎么动（或只是抖动）
    if score_new is None or ov_new is None:
        return "lost"
    if score_new >= TRACK_ACCEPT or ov_new < TRACK_OVERLAP_MIN:
        return "lost"
    if score_prev is not None and score_new > TRACK_MARGIN * score_prev:
        return "lost"          # 改善不够大 ⇒ 宁可判丢，也不跳到可能是幽灵的位置
    return "accept"


def band_around(s, half=TRACK_SCALE_BAND, step=0.025):
    """以 s 为中心、半宽 ±half 的尺度带（步长步进，钳在合法域内，去重保序）。"""
    out, x = [], s - half
    while x <= s + half + 1e-9:
        v = round(min(max(x, SCALE_LO), SCALE_HI), 3)
        if not out or abs(v - out[-1]) > 1e-9:
            out.append(v)
        x += step
    return tuple(out)


class Tracker:
    """持「当前显示的投影是从哪张参考图、用哪个变换来的」，并给出下一步该做什么。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.panel = None
        self.seed = None
        self.floor = None
        self.ref_path = None
        self.M = None            # 当前投影的 ref→屏幕 变换（2x3）
        self.s = None            # 其尺度
        self.lost = 0            # 连续判丢计数
        self.misses = 0

    # ---- 生命周期 ----
    def seed_from(self, panel, seed, floor, ref_path, M, s) -> None:
        """建立/重置跟踪：新投影出来时调用（热键、手动重对齐都算）。"""
        self.reset()
        self.panel, self.seed, self.floor = panel, seed, floor
        self.ref_path = ref_path
        self.M, self.s = M, float(s)

    @property
    def active(self) -> bool:
        return self.M is not None and self.s is not None

    # ---- 几何 ----
    def pin(self):
        """把「上一帧变换」换成一个锚点四元组，喂给 `find_overlay_transform(hint_icon=…)`。

        四元组语义 = 「参考图上某点 ↔ 屏幕上某点」。取**面板中心**当屏幕点：
        纯平移下 pin 取哪个点都等价（P 在公式里自动抵消），而缩放时误差
        `(P−C)·s·(1−k)`（C=游戏缩放定点）⇒ 取面板中心最稳。
        返回 (ref_cx, ref_cy, screen_x, screen_y) 或 None。
        """
        if not self.active or self.panel is None:
            return None
        px, py, pw, ph = self.panel
        sx, sy = px + pw / 2.0, py + ph / 2.0
        tx, ty = float(self.M[0, 2]), float(self.M[1, 2])
        return ((sx - tx) * self.s, (sy - ty) * self.s, sx, sy)

    def advanced(self, M, s) -> bool:
        """新变换相对当前的是否超出了死区（要不要重烘焙）。"""
        if not self.active:
            return True
        d = math.hypot(float(M[0, 2]) - float(self.M[0, 2]),
                       float(M[1, 2]) - float(self.M[1, 2]))
        ds = abs(float(s) - self.s) / max(self.s, 1e-6)
        return d > TRACK_DEADBAND_PX or ds > TRACK_DEADBAND_S

    def adopt(self, M, s) -> None:
        """接受新变换。"""
        self.M, self.s = M, float(s)
        self.lost = 0

    def note_lost(self) -> int:
        """记一次判丢，返回累计次数。"""
        self.lost += 1
        return self.lost

    def note_ok(self) -> None:
        self.lost = 0

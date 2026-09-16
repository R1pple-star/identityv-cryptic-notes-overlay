# -*- coding: utf-8 -*-
"""
视觉原语（从 matcher.py 抽出，Phase 2.1）
==========================================
共享视觉底层：读图 / 迷雾色 / 面板坐标 / 颜色分类 / 入口图标检测 / 地图开合。
引索路径、对齐、校准都依赖本模块。

搬家自 matcher.py，纯移动、逻辑不变。唯一变化：FIXED_PANEL 与入口图标模板路径
改从 config.toml 读（值与旧硬编码一致，行为不变）。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 项目根（core/ 上一层）
ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = ROOT / "config.toml"


def _load_config():
    """读 config.toml 的面板坐标与图标模板路径；失败兜底硬编码默认（绝不崩，保证 import 安全）。"""
    defaults = {
        "panel_rect": (668, 166, 1064, 569),
        "icon_template": "_icon_entrance.png",
        "metric": "sqdiff",
    }
    try:
        with open(_CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        defaults["panel_rect"] = tuple(cfg["panel"]["rect"])
        defaults["panel_res"] = tuple(cfg["panel"].get("resolution", [1920, 1080]))
        defaults["panel_rects"] = {k: tuple(v)
                                   for k, v in cfg["panel"].get("rects", {}).items()}
        defaults["icon_template"] = cfg["paths"]["icon_template"]
        defaults["metric"] = cfg["match"].get("metric", "sqdiff")
    except Exception:
        pass
    return defaults


_CFG = _load_config()

# 基准分辨率下的固定面板（x, y, 宽, 高）。依据用户标注参考图的地图边界白框确定。
# 其它分辨率经 panel_for_screen 等比外推或查 config [panel.rects] 校准表。
FIXED_PANEL = _CFG["panel_rect"]
_PANEL_REF_RES = _CFG["panel_res"]
# 匹配口径（config [match] metric）："consistency" = 错配率（新）/ "sqdiff" = 类号平方差（旧）。
# 入口引索匹配与投影重合两侧共用此开关；原理、实测与代价见 simplex5 长注。
MATCH_METRIC = str(_CFG.get("metric", "sqdiff"))


def panel_for_screen(w: int, h: int):
    """按物理分辨率求地图面板 (x, y, w, h)；未适配分辨率返回 None。

    优先级：1920×1080（=基准）→ FIXED_PANEL 精确值；config [panel.rects] 每分辨率
    校准表；同为 16:9 → 按 W/基准宽 等比外推（右/下边缘取整再回推宽高，避免独立
    取整累计漂移）；非 16:9 且未校准 → None。

    注：动态检测（雾块/暗块连通块）已实测证伪——雾态下面板边缘两侧同为深色不可见，
    雾块 bbox 随探明状态漂移 dx −288~+480（experiments/diag_panel_dynamic.py），
    暗块被游戏场景连通吞整屏（diag_panel_window.py）。
    """
    if (w, h) == tuple(_PANEL_REF_RES):
        return FIXED_PANEL
    key = f"{w}x{h}"
    rects = _CFG.get("panel_rects", {})
    if key in rects:
        return tuple(rects[key])
    if abs(w * 9 - h * 16) <= 8:  # 16:9（容忍 1366×768 类取整误差）
        k = w / _PANEL_REF_RES[0]
        px, py, pw, ph = FIXED_PANEL
        x0, y0 = round(px * k), round(py * k)
        x1, y1 = round((px + pw) * k), round((py + ph) * k)
        if x0 <= 0 or y0 <= 0 or x1 > w or y1 > h:
            return None
        return (x0, y0, x1 - x0, y1 - y0)
    return None


def detect_fog_panel(bgr_screen: np.ndarray):
    """按截屏物理尺寸适配地图面板（阶段C 比例适配）。返回 (x, y, w, h) 或 None。

    旧 use_fixed=False 雾块连通块路径已删（实测不可作锚，见 panel_for_screen 注）。
    """
    return panel_for_screen(bgr_screen.shape[1], bgr_screen.shape[0])

# 入口图标模板路径（config.paths.icon_template，相对项目根解析为绝对路径）。
ICON_TEMPLATE = (ROOT / _CFG["icon_template"]).resolve()


def load_bgr(path: str) -> np.ndarray:
    """用 PIL 读图（支持中文路径），返回 BGR ndarray。"""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


# 游戏内地图迷雾的颜色（BGR 顺序）。RGB(37,47,58) -> BGR(58,47,37)
FOG_BGR = np.array([58, 47, 37], dtype=np.int16)

# 墙色标定（2026-09-15 用户提供纯色块样本 D:\yanshi 实测，双侧均验证出墙线网络）：
#   通路墙 BGR(138,119,111) 亮度119 冷-27（旧分类落 cls3=通路）
#   房间墙 BGR(114,122,139) 亮度126 暖+25（旧分类落 cls2=房间）
# 两墙中心相距55 → 并作一个墙类(拆两类会被交界带/抗锯齿跨侧翻转)；与最近地面色距≥99
# → tol48 覆盖两墙交界带(距两心44/45)且不吞地面。拦截必须在 room/passage 判定之前。
WALL_BGR = (np.array([138, 119, 111], dtype=np.int16),
            np.array([114, 122, 139], dtype=np.int16))
WALL_TOL = 48


def walls_as_floors(cls, region):
    """把 cls5(墙)按冷暖并回旧的 2/3/4/1 语义，供匹配两侧调用。

    墙类不进 SQDIFF 的死因是厚度惩罚(游戏内墙5-9px vs 参考墙1-3px, 直进重合分
    0.046→1.31, 参考侧膨胀也救不回)——两侧墙线画位在图标锚定下重合 60-67%(素材与
    游戏内同渲染器, 早先"画位不重合"系幽灵相位误判, 见 CLAUDE.md)。并回旧语义后
    匹配与基线逐位一致; cls5 保留给贴墙增益(wall_boost)/投影高亮等用途。"""
    out = cls.copy()
    w5 = out == 5
    if not w5.any():
        return out
    b = region[:, :, 0].astype(np.int16)
    r = region[:, :, 2].astype(np.int16)
    warm = r - b
    gray = 0.114 * b + 0.587 * region[:, :, 1].astype(np.int16) + 0.299 * r
    out[w5 & (warm > 12)] = 2
    out[w5 & (warm < -10)] = 3
    out[w5 & (out == 5) & (gray >= 85)] = 1   # 墙交界带(亮中性)——旧分类落 cls1 的那 0.2%
    out[w5 & (out == 5)] = 4                   # 中性且暗——旧 fog 桶
    return out


# ===== 匹配口径：一致性（新）vs 类号平方差（旧）=====
# 旧口径把 5 个类别当成数字 0..5 做 TM_SQDIFF，惩罚 = (类号差)²，纯属编号巧合：
#   路(cls3)↔黑(cls0) 罚 9、房(cls2)↔路(cls3) 罚 1 —— 但两者都是"错"，凭什么差 9 倍。
# 实测（2026-09-16，真种子+真尺度+锚定，实机截图）后果：真对齐下类别一致率 87%~92%，
# 可 4.4%~6.1% 的像素（全是 |差|=3 那类）贡献了 69%~95% 的总分 → 真对齐算成 0.473~2.560，
# 永远过不了 0.30 的显示闸；而"缩到最小尺度让模板躲进均匀区"能把分离群像素挤出去 → 分更低
# → 全搜恒钉尺度下界（把下界从 0.30 改 0.25，5/5 立刻跟着降到 0.250~0.258）。
#
# 新口径：把 5 类嵌成 R^4 里正单纯形的 5 个顶点 —— **任意两类之间平方距离全相等(=2)**。
# 于是 4 通道 TM_SQDIFF 的响应 = 2×(不一致像素数)，除以 2·权重和 即「错配率」∈[0,1]：
# 可读懂、与类号编排无关、且"缩模板"的收益从 9× 压到 1×。
# 只用 4 通道 ⇒ matchTemplate 调用次数不变（默认上限就是 4 通道，故 6 类不能直接 one-hot）。
# 实测同 M 下：旧口径 0.574 → 新口径 0.132（= 1-87%，与独立量出的一致率自洽）。
_SIMPLEX5 = None


def simplex5() -> np.ndarray:
    """5 类 → R^4 正单纯形顶点，shape (5,4)，任意两类平方距离 = 2。"""
    global _SIMPLEX5
    if _SIMPLEX5 is None:
        # 中心化 one-hot (I - J/5) 的右奇异向量前 4 行张成「正交于 (1,1,1,1,1) 的 4 维子空间」。
        # 中心化保距 ⇒ 投影到该子空间后仍保距，即 5 个顶点的两两平方距离全 = 2（one-hot 时 = 2）。
        _SIMPLEX5 = np.linalg.svd(np.eye(5) - 1.0 / 5)[2][:4].T.astype(np.float32)
    return _SIMPLEX5


def to_simplex(cls: np.ndarray) -> np.ndarray:
    """(H,W) 类别图 → (H,W,4) float32 单纯形嵌入。调用方须先经 walls_as_floors 归并 cls5。"""
    return simplex5()[np.clip(cls.astype(np.int32), 0, 4)]


def content_bbox(bgr_img: np.ndarray, thresh: int = 60) -> tuple[int, int, int, int]:
    """参考图里内容(迷宫)的包围盒，裁掉四周留白/边框。返回 (x0, y0, w, h)。"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(th > 0)
    if len(xs) == 0:
        return (0, 0, bgr_img.shape[1], bgr_img.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))


def classify_region(region):
    """按 墙色距 + R-B 冷暖度 + 亮度 分类。0=黑/无,1=亮剩余,2=房间(暖/棕),3=通路(冷/蓝灰),4=迷雾(中性灰),5=墙。

    颜色标定（来自用户标注参考图 + 2026-09-15 纯色块样本）:
      通路 RGB(76,82,100) R-B~-24  冷
      房间 RGB(109,96,87) R-B~+22  暖
      迷雾 RGB(70,70,76)  R-B~-6   中性
      黑   RGB(28,36,46)  亮度36   暗
      通路墙 BGR(138,119,111) 亮度119 R-B~-27（色距判定，先于冷暖）
      房间墙 BGR(114,122,139) 亮度126 R-B~+25（同上，与通路墙并作 cls5）
    """
    b = region[:, :, 0].astype(np.int16)
    r = region[:, :, 2].astype(np.int16)
    gray = 0.114 * b + 0.587 * region[:, :, 1].astype(np.int16) + 0.299 * r
    warmth = r - b

    cls = np.zeros(region.shape[:2], dtype=np.uint8)
    dark = gray < 42   # 真正的黑/空；雾(亮度45+)不算黑
    cls[dark] = 0
    # 墙色拦截（先于冷暖：通路墙冷-27 会落 cls3、房间墙暖+25 会落 cls2）
    i16 = region.astype(np.int16)
    wall = (~dark) & ((np.abs(i16 - WALL_BGR[0]).sum(axis=2) < WALL_TOL) |
                      (np.abs(i16 - WALL_BGR[1]).sum(axis=2) < WALL_TOL))
    cls[wall] = 5
    room = (~dark) & (cls == 0) & (warmth > 12)
    cls[room] = 2
    passage = (~dark) & (cls == 0) & (warmth < -10)
    cls[passage] = 3
    fog = (~dark) & (cls == 0) & (np.abs(warmth) <= 12) & (gray < 85)
    cls[fog] = 4
    cls[(~dark) & (cls == 0) & (gray >= 85)] = 1
    return cls


def classify_map(bgr):
    """兼容旧名，直接用 R-B 分类。"""
    return classify_region(bgr)


def _find_icon(bgr, px, py, pw, ph,
               scales=None):
    """在面板内找白色箭头入口图标（排除黄色玩家图标）。多尺度，能抓到随地图缩放缩小的图标。

    尺度网格细至 0.05 步（0.35-1.30）+ 上方粗档：真实图标尺度是「图标当尺子」的依据
    （core.entrance._crop_around_icon 按 k 缩放裁样），必须量准；旧粗网格下限 0.5 会把
    8 月基线图标钉在网格底（实测真 k=0.40-0.45，细化后 NCC 0.64→0.81+）。
    返回 (中心坐标, 分数, 获胜尺度k) 或 (None, 分, None)。
    """
    if scales is None:
        scales = tuple(round(0.35 + 0.05 * i, 2) for i in range(20)) + (1.5, 1.7, 2.0, 2.2)
    icon_path = ICON_TEMPLATE
    if not icon_path.exists():
        return None, 0, None  # 缺图标模板不检测（新结构由 assets/_icon_entrance.png 提供；旧版自动从硬编码截图创建的逻辑已删）
    icon = cv2.imread(str(icon_path), cv2.IMREAD_GRAYSCALE)
    if icon is None:
        return None, 0, None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    panel_gray = gray[py:py + ph, px:px + pw]
    region = bgr[py:py + ph, px:px + pw]
    rr = region[:, :, 2].astype(np.int16)
    bb = region[:, :, 0].astype(np.int16)
    yellow = ((rr - bb) > 50)
    yh, yw = panel_gray.shape[0], panel_gray.shape[1]

    iw, ih = icon.shape[1], icon.shape[0]
    best_score, best_pos, best_k = 0.0, None, None
    for scale in scales:
        siw, sih = max(8, int(iw * scale)), max(8, int(ih * scale))
        if siw > pw or sih > ph:
            continue
        icon_s = cv2.resize(icon, (siw, sih), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(panel_gray, icon_s, cv2.TM_CCOEFF_NORMED).copy()
        rh, rw = res.shape
        for _ in range(6):
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < 0.58:
                break
            mx, my = ml
            ccx, ccy = mx + siw // 2, my + sih // 2
            cy0, cy1 = max(0, ccy - 10), min(rh, ccy + 10)
            cx0, cx1 = max(0, ccx - 10), min(rw, ccx + 10)
            if yellow[cy0:cy1, cx0:cx1].size and yellow[cy0:cy1, cx0:cx1].mean() > 0.5:
                res[max(0, my - 3):my + 3, max(0, mx - 3):mx + 3] = -1
                continue
            if mv > best_score:
                best_score, best_pos, best_k = mv, (px + mx + siw // 2, py + my + sih // 2), scale
            break
    if best_pos is None or best_score < 0.58:
        return None, best_score, None
    return best_pos, best_score, best_k


def follow_features(region):
    """跟随开合判定的三特征：返回 (content, fog, struct)。

    content = 非暗像素占比(cls≠0)——仅日志展示，不参与判定（游戏画面亮度变化会误判，
    2026-09-14 实测关态 Alt/F 特效 content 冲到 0.31 跨过旧阈值）；
    fog    = FOG_BGR tol24 色距占比（关态 ≤0.002；重度探明开态可低至 0.001）；
    struct = (cls2房间|cls3通路) 占比（开态 min 0.21 / 关态 max 0.017，主判据）。
    判定规则在 ui/follow.py：fog ≥ FOLLOW_FOG_THRESH 或 struct ≥ FOLLOW_STRUCT_THRESH。
    """
    cls = classify_region(region)
    content = float((cls != 0).mean())
    fog = float((np.abs(region.astype(np.int16) - FOG_BGR).sum(axis=2) < 24).mean())
    struct = float(((cls == 2) | (cls == 3) | (cls == 5)).mean())
    return content, fog, struct


def map_is_open(shot_bgr, panel=FIXED_PANEL, icon_template=None):
    """地图是否打开：面板里须有「迷宫内容」（迷雾 或 通路/房间/墙 结构），而非一般游戏画面。

    大厅/游戏场景/结算界面 没有迷宫结构，应判为未打开。
    返回 (bool, 迷宫内容占比)。跟随的可靠判定请用 follow_features 双特征（见其注释）。
    """
    px, py, pw, ph = panel
    region = shot_bgr[py:py + ph, px:px + pw]
    content, _fog, _struct = follow_features(region)
    return content > 0.12, content


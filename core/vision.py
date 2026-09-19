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
        "panel_res": (1920, 1080),
        "icon_template": "_icon_entrance.png",
        "nav_template": "assets/ui_nav_column.png",
        "fog_bgr": (58, 47, 37),
        "fog_tol": 24,
    }
    try:
        with open(_CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        defaults["panel_rect"] = tuple(cfg["panel"]["rect"])
        defaults["panel_res"] = tuple(cfg["panel"].get("resolution", [1920, 1080]))
        defaults["panel_rects"] = {k: tuple(v)
                                   for k, v in cfg["panel"].get("rects", {}).items()}
        defaults["icon_template"] = cfg["paths"]["icon_template"]
        defaults["nav_template"] = cfg["paths"].get("nav_template", defaults["nav_template"])
        defaults["fog_bgr"] = tuple(cfg["vision"].get("fog_bgr", defaults["fog_bgr"]))
        defaults["fog_tol"] = int(cfg["vision"].get("fog_tol", defaults["fog_tol"]))
    except Exception:
        pass
    return defaults


_CFG = _load_config()

# 基准分辨率下的固定面板（x, y, 宽, 高）。依据用户标注参考图的地图边界白框确定。
# 其它分辨率经 panel_for_screen 等比外推或查 config [panel.rects] 校准表。
FIXED_PANEL = _CFG["panel_rect"]
_PANEL_REF_RES = _CFG["panel_res"]
# 匹配口径：**只有一种** —— 「错配率」（5 类嵌 R^4 正单纯形 / 数一致 CCORR，见 to_match3 长注）。
# 旧的「类号平方差」sqdiff 已于 2026-09-16 删除（它的病理见 CLAUDE.md 技术备忘①：
# 惩罚=(类号差)² 纯属编号巧合，真对齐算成 0.47~2.56 永远过不了闸）。要回滚请走 git。


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
# 侧栏导航列模板（config.paths.nav_template）—— 地图开合判据用，见下方 NAV_BAND 长注。
NAV_TEMPLATE = (ROOT / _CFG["nav_template"]).resolve()


def load_bgr(path: str) -> np.ndarray:
    """用 PIL 读图（支持中文路径），返回 BGR ndarray。"""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


# 游戏内地图迷雾的颜色（BGR 顺序）。RGB(37,47,58) -> BGR(58,47,37)
# 值与容差来自 config.toml [vision]（fog_bgr / fog_tol），缺配置时用同值默认兜底。
# 迷雾是冷色不是中性灰 —— 一切 revealed 判定必须按它剔除（CLAUDE.md 硬约束）。
FOG_BGR = np.array(_CFG["fog_bgr"], dtype=np.int16)
FOG_TOL = int(_CFG["fog_tol"])

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
    对齐层匹配与基线逐位一致; **cls5 原样保留**，现由入口层的「墙重合度量」
    (core/entrance.wall_overlap) 直接消费——那里正是把"墙"当几何用的地方。"""
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
# 永远过不了显示闸；而"缩到最小尺度让模板躲进均匀区"能把离群像素挤出去 → 分更低
# → 全搜恒钉尺度下界（把下界从 0.30 改 0.25，5/5 立刻跟着降到 0.250~0.258）。
#
# 新口径 = **错配率**（不一致像素占比 ∈[0,1]）：与类号编排无关、可直接读懂、且"缩模板"
# 的收益从 9× 压到 1×。实测同 M 下 0.574 → 0.132（= 1-87%，与独立量出的一致率自洽）。
#
# 实现（2026-09-16 第二版，为速度）：**数"一致"而不是数"不一致"**。
#   一致数 = Σ_{掩膜内 p} [t_p == r_p] = Σ_{c} ⟨掩膜内one-hot_c , 参考one-hot_c⟩，是一组
#   **互相关**。互相关用 `TM_CCORR` 且**不带 mask** —— 不带 mask 时 OpenCV 走 FFT 快路，
#   带 mask 则退化成逐点直算（最慢路径，且通道数再乘一遍）。掩膜可以**折进模板**：
#   CCORR 是乘积，模板在掩膜外填 0 就等于掩膜。于是
#       响应 = msum - 一致数   （msum = 掩膜权重和）
#   仍是"越小越匹配"的代价图，下游 minMaxLoc / 亚像素抛物线**一字不改**。
#   实测（引索 456×379 / 模板 200×200，单次 matchTemplate）：
#       带mask SQDIFF 1通道(旧) 5.48ms | 带mask SQDIFF 4通道(第一版) 21.85ms
#       无mask CCORR 4通道 9.25ms | **无mask CCORR 3通道(本版) ≈7ms**
#
# 为什么 3 通道够（不必 4/5）：
#   掩膜内**模板**类只可能是 1/2/3（cls0 黑 / cls4 雾 / cls5 墙 都在掩膜外 —— mask 定义即
#   (cls∈{1,2,3}) & ~fog），所以模板不需要 cls0/cls4 的通道；参考侧落在那两类的像素
#   三通道全 0，与模板任何通道都不相乘 ⇒ 自动计为"不一致"。逐位无损。
#   反之若数"不一致"（SQDIFF）就必须给 cls0/雾 留通道，5 类需 4 维单纯形 —— 又慢又笨。
def to_match3(cls: np.ndarray) -> np.ndarray:
    """(H,W) 类别图 → (H,W,3) float32 one-hot：通道 0/1/2 = [cls==1]/[cls==2]/[cls==3]。

    仅供「一致计数」用（见本段长注）。调用方须先经 walls_as_floors 归并 cls5。
    """
    idx = cls.astype(np.uint8)
    out = np.zeros(cls.shape + (3,), np.float32)
    out[..., 0] = idx == 1
    out[..., 1] = idx == 2
    out[..., 2] = idx == 3
    return out


def consistent_cost(ref_oh: np.ndarray, tpl_oh: np.ndarray, wsum: float) -> np.ndarray:
    """一致计数的代价图：`wsum - 加权一致像素数`（越小越匹配，= 不一致加权数）。

    ref_oh / tpl_oh 均为 to_match3 输出；tpl_oh 须已乘掩膜权重（掩膜外为 0）。
    **不带 mask**，故走 FFT 快路。返回 float32 响应面，正值 = 不一致数；FFT 舍入可能
    产生 ~1e-2 级负值（完美匹配附近），下游按"越小越好"处理即可，不必钳。
    """
    return (wsum - cv2.matchTemplate(ref_oh, tpl_oh, cv2.TM_CCORR)).astype(np.float32)


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
    fog    = FOG_BGR tol24 色距占比（开态 0.20~0.69；游戏世界最高 0.17）；
    struct = (cls2房间|cls3通路|cls5墙) 占比——**2026-09-17 起不再参与开合判定**，仅日志：
    游戏世界实测能到 0.21，旧阈 0.10 会被跨过而误判为「开」（见 NAV_BAND 长注）。
    开合判定现走 map_is_open / map_open_from_roi（侧栏导航列 NCC 为主判据）。
    """
    cls = classify_region(region)
    content = float((cls != 0).mean())
    fog = float((np.abs(region.astype(np.int16) - FOG_BGR).sum(axis=2) < FOG_TOL).mean())
    struct = float(((cls == 2) | (cls == 3) | (cls == 5)).mean())
    return content, fog, struct


# ---- 地图开合判据（2026-09-17 换口径：侧栏导航列）--------------------------------
# 侧栏导航列 = 地图画面**独有**的固定 UI（ESC / 导航至出口 / 门箭头圆钮 / + 圆钮 / 竖滑条），
# 位于**面板之外**（面板右缘 1732，列在 1736~1900）⇒ 位置与外观都固定，是个二值信号。
# 用户 2026-09-17 提议「用侧面 UI 判断地图是否开启」。
#
# ⚠️ 实测（experiments/e29_closed_scan.py / e30_map_open_gate.py / e31_nav_calib.py，176 张）：
#   **导航列是「高精度、低召回」**，且低召回有确切原因 —— **布局**，不是「UI 淡出」：
#     全屏实机（屏幕底行均值 ≈36，无任务栏）      → NCC 0.86~1.00  ✅可信
#     窗口化（底行 ≈220 是 Windows 任务栏）        → NCC 0.34~0.36  ← 低于闸，走雾兜底
#     照片查看器/桌面（底行亮、有窗口标题栏）      → NCC 0.03~0.16
#   窗口化那档**不能用邻域搜索救**：±40/±30 搜索把 17.11 从 0.36 抬到 0.664，但真负样本
#   同时被抬到 0.271，分离带反而更窄（e31 实测）⇒ **不做位置搜索**。
#   已人工逐张确认的负样本（游戏世界 3 / 结算 / 登录 / 桌面 / 黑屏）定点 NCC **最高 0.271**。
#   所以：NCC ≥ 此闸 ⇒ 判开**可信**；NCC 低**不能**判关 —— 必须靠雾兜底（窗口化时全靠它）。
#
# 旧判据（雾≥0.10 或 结构≥0.10）在同一批数据上的**实测误判**：
#   08.25 木楼梯(雾0.00 结构0.21)、09.14 走廊(雾0.17)、08.25 吊灯场景(结构**0.51**)、
#   09.14 登录画面(结构0.52) → 全被判成「开」⇒ 地图关着投影却留在画面上，
#   即用户报的「按 G 关地图不灵敏」。**故结构占比整个退出判定**（它是最脏的一路）。
# 雾兜底阈值同时从 0.10 上调到 0.30：负样本实测最高 0.17（旧阈 0.10 会被它跨过）。
NAV_BAND = (4, -106, 164, 740)   # 相对面板: (右缘起dx, 顶起dy, 宽, 高)，随面板宽等比缩放
NAV_NCC_MIN = 0.45               # 负样本最高 0.271 / 可信正样本最低 0.543 ⇒ 取中，两侧各留 ≥0.17
FOG_OPEN_MIN = 0.30              # 雾兜底：开态实测 0.20~0.69（兜底那一批全 ≥0.42）；负样本最高 0.17

_NAV_TPL_CACHE: dict = {}


def _nav_template():
    """导航列模板（懒加载 + 常驻缓存——跟随每秒都要用，不能每次读盘）。"""
    key = str(NAV_TEMPLATE)
    if key not in _NAV_TPL_CACHE:
        _NAV_TPL_CACHE[key] = load_bgr(key) if NAV_TEMPLATE.exists() else None
    return _NAV_TPL_CACHE[key]


def nav_band(panel=FIXED_PANEL):
    """侧栏导航列的屏幕矩形 (x0, y0, x1, y1)，随面板宽等比缩放（多分辨率）。"""
    px, py, pw, _ph = panel
    k = pw / float(FIXED_PANEL[2])
    dx, dy, w, h = NAV_BAND
    x0, y0 = px + pw + int(round(dx * k)), py + int(round(dy * k))
    return x0, y0, x0 + int(round(w * k)), y0 + int(round(h * k))


def map_roi(panel=FIXED_PANEL):
    """地图画面 ROI (x0, y0, x1, y1) = 面板 ∪ 导航列（∪ 圆点读数带）。

    跟随每 tick 只截这一块：既够算面板的雾特征，也够算导航列的 NCC，一次截屏两用。
    ⚠️ 纵向要按 `NAV_DOT_BAND_H` 加长：滑条圆点会跑到 NCC 带（740）之外，裁短了读不到
    最缩小态（2026-09-18，见 NAV_DOT_RANGE 长注）。只加长 ROI，NAV_BAND 本身不动。
    """
    px, py, pw, ph = panel
    nx0, ny0, nx1, ny1 = nav_band(panel)
    ny1 = max(ny1, ny0 + int(round(NAV_DOT_BAND_H * (nx1 - nx0) / float(NAV_BAND[2]))))
    return min(px, nx0), min(py, ny0), max(px + pw, nx1), max(py + ph, ny1)


def roi_of(shot_bgr, panel=FIXED_PANEL):
    """从全屏图裁出 map_roi。"""
    x0, y0, x1, y1 = map_roi(panel)
    return shot_bgr[y0:y1, x0:x1]


def nav_column_ncc(roi_bgr, panel=FIXED_PANEL):
    """ROI 内导航列与模板的灰度 NCC ∈[-1,1]；模板缺失/ROI 太小返回 None（调用方退回雾判据）。"""
    tpl = _nav_template()
    if tpl is None:
        return None
    nx0, ny0, nx1, ny1 = nav_band(panel)
    rx, ry, _x1, _y1 = map_roi(panel)
    band = roi_bgr[ny0 - ry:ny1 - ry, nx0 - rx:nx1 - rx]
    if band.size == 0:
        return None
    if band.shape != tpl.shape:
        tpl = cv2.resize(tpl, (band.shape[1], band.shape[0]))
    a = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def map_open_from_roi(roi_bgr, panel=FIXED_PANEL):
    """地图画面是否打开（ROI 版，返回 `(bool, 依据字符串, 主判据是否成立)`）。

    主判据 = 导航列 NCC ≥ NAV_NCC_MIN；导航列模板缺失时退回 雾 ≥ FOG_OPEN_MIN。

    第三项 `nav_ok` = **主判据自己**说「开」，而不是雾兜底说开。雾兜底是「低精度、高召回」：
    它能把开态救回来，代价是**关闭动画那几帧也会被判成开**（2026-09-18 实机 22:52:33：
    导航列 NCC −0.01（列根本不在）而雾 0.41 ⇒ 判开）。跟踪链路必须区分这两者 —— 只有主判据
    说开时，判丢才允许销毁 `_last_rgba`，否则 G 开回来无图可放回（bug① 换个入口复发，
    实机 22:52:35 判丢 → 22:52:42「上次没有成功投影 ⇒ 不自动恢复」）。
    """
    ncc = nav_column_ncc(roi_bgr, panel)
    px, py, pw, ph = panel
    rx, ry, _x1, _y1 = map_roi(panel)
    region = roi_bgr[py - ry:py - ry + ph, px - rx:px - rx + pw]
    fog = float((np.abs(region.astype(np.int16) - FOG_BGR).sum(axis=2) < FOG_TOL).mean())
    if ncc is None:
        return fog >= FOG_OPEN_MIN, f"导航列模板缺失→雾{fog:.2f}", False
    if ncc >= NAV_NCC_MIN:
        return True, f"导航列{ncc:.2f}", True
    if fog >= FOG_OPEN_MIN:
        return True, f"导航列{ncc:.2f}偏低→雾{fog:.2f}兜底", False
    return False, f"导航列{ncc:.2f} 雾{fog:.2f}", False


def map_is_open(shot_bgr, panel=FIXED_PANEL):
    """地图画面是否打开（全屏图版）。返回 (bool, 依据字符串)。

    热键路径在跑匹配前用它拒掉大厅/结算等非地图画面（CLAUDE.md 待办 3）。
    """
    is_open, why, _nav_ok = map_open_from_roi(roi_of(shot_bgr, panel), panel)
    return is_open, why


# ---- 缩放滑条读数（2026-09-17 用户提示：右侧 UI 的长条 + 圆点，点的位置=缩放程度）------
# 动机：**对齐分选不出尺度**（技术备忘⑥）—— 分对 s 单调递增、真尺度处不是极小，于是「挑分
# 最小的尺度」典型偏低 0.01~0.02、曲线平坦时偏低 0.09（几千像素错位），且恒被 ALIGN_SCALES
# 下界 0.30 截住；而游戏最放大时真尺度是 0.25。滑条直接给出当前缩放，绕开这一整条病根。
#
# 几何：滑条在导航列里，轨道是那条亮细竖条（模板内 x=NAV_TRACK_X），圆点=轨道上又亮又宽的
# 峰。实测模板（7 张全屏实机图的均值）轨道列 x=84，圆点活动范围 y∈[315,805]。
# ⚠️ 这个范围 2026-09-18 由 (380,730) 放宽：用户报「缩大小不准」，4 帧只改缩放的演示证明
# 圆点能走到 **y=752**（真尺度 0.83），而旧窗到 730 就截住 ⇒ 最缩小态**读数直接失败**
# （峰-基只剩 4，过不了闸），跟随于是拿旧尺度去做全平移搜索（实测图标偏 184px，见 e45）。
# 轨道实体一直延伸到 y≈790，再往下是底部圆钮（全览/我的位置，y≥860）——别把窗开过头。
NAV_TRACK_X = 84                 # 滑条轨道列（模板内坐标）
NAV_DOT_RANGE = (315, 805)       # 圆点可活动的 y 段（模板内坐标）
NAV_DOT_BAND_H = 820             # 圆点读数要裁的导航列高度：比 NAV_BAND 的 740 深 80px
                                 # （圆点会跑到 NCC 带之外）。只加深**读数**的裁切、不动
                                 # NAV_BAND —— 后者一高，nav_column_ncc 就会 resize 模板，
                                 # NCC 闸的标定（0.45）与 e30/e31/e32 全部作废。
ZOOM_DOT_MIN = 15.0              # 圆点峰须高出轨道基线这么多才算读到了（否则判读数失败）
# 圆点 y → 尺度 s 的曲线：`ln s = Σ c_i·t^i`，t=(y−560)/200（y 先钳到 [375,755]）。
# 2026-09-18 重标定（experiments/e46_slider_recalib.py + e39/e44/e45）：
#   **真值锚点**（图标锚点法，图标误差 1~5px —— 引索 json 的图标 cx/cy 经 M 映到 `_find_icon`
#   实测的屏幕位置）5 点：(382,0.230) (488,0.300) (589,0.395) (627,0.430) (752,0.830)
#   **图标尺子** `s≈0.374/k` 的中位（75 张全屏帧；k 量化 ±0.05 ⇒ 该源自身 ±0.05/k）7 点：
#     (545,0.325) (550,0.340) (584,0.374) (646,0.440) (685,0.528) (713,0.636) (728,0.701)
#   三阶拟合残差：锚点 −0.000/+0.001/−0.015/−0.002/−0.004，尺子 +0.016/+0.004/+0.001/
#   +0.019/+0.018/+0.000/−0.002。
#   旧表（09-17 视频 11 点）只覆盖 y∈[382,708]，尾段还偏低：表在 708 处给 0.60，真值是 0.83
#   —— 「缩大小不准」的另一半就在这里。
#   ⚠️ y>755 没有测量（滑条最下档停在 752），曲线在那里外推不可信 ⇒ 钳住不外推。
ZOOM_CURVE_C = (-1.042, 0.48454, 0.20267, 0.22509)
ZOOM_DOT_Y_CLAMP = (375.0, 755.0)
# 图标尺子：`s ≈ ICON_RULER_C / k`（k = `_find_icon` 的获胜尺度）。滑条的**备用尺度源**——
# 导航列读不到时（窗口化布局/UI 动画）仍可用，代价是 `_find_icon` 约 200ms。
# 依据：5 个图标锚点真值上 `0.374/k` 全部落在 ±0.011 内；75 张全屏帧的 (圆点y, 0.374/k)
# 与上表单调一致。**与分辨率无关**：它是「图标在参考图里的像素数 ÷ 模板像素数」，两者都不随
# 屏幕分辨率变。
ICON_RULER_C = 0.374


def zoom_scale_from_k(k):
    """图标尺子：由 `_find_icon` 的获胜尺度 k 反推 s。k 为 None/非正 ⇒ None。"""
    if not k or k <= 0:
        return None
    return ICON_RULER_C / float(k)


def _dot_y_to_s(y):
    """圆点 y（模板内坐标）→ 尺度 s（曲线来源见 ZOOM_CURVE_C 长注）。"""
    t = (min(max(float(y), ZOOM_DOT_Y_CLAMP[0]), ZOOM_DOT_Y_CLAMP[1]) - 560.0) / 200.0
    return float(np.exp(sum(c * t ** i for i, c in enumerate(ZOOM_CURVE_C))))


def nav_dot_y(roi_bgr, panel=FIXED_PANEL):
    """导航列滑条圆点的 y（**模板内坐标**，已按面板宽折算）+ 峰高出基线的量。

    读不到（ROI 太小/全黑）返回 (None, 0.0)。质量分给调用方自己判（< ZOOM_DOT_MIN 别用）。
    """
    nx0, ny0, nx1, ny1 = nav_band(panel)
    ny1 = ny0 + int(round(NAV_DOT_BAND_H * (nx1 - nx0) / float(NAV_BAND[2])))
    rx, ry, _x1, _y1 = map_roi(panel)
    band = roi_bgr[ny0 - ry:ny1 - ry, nx0 - rx:nx1 - rx]
    if band.size == 0:
        return None, 0.0
    k = band.shape[1] / float(NAV_BAND[2])          # 多分辨率：按宽度折算回模板坐标
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tx = int(round(NAV_TRACK_X * k))
    if tx < 3 or tx + 4 > gray.shape[1]:
        return None, 0.0
    seg = gray[:, tx - 3:tx + 4].mean(axis=1)
    r0 = max(0, int(round(NAV_DOT_RANGE[0] * k)))
    r1 = min(len(seg) - 1, int(round(NAV_DOT_RANGE[1] * k)))
    if r1 - r0 < 20:
        return None, 0.0
    base = float(np.median(seg[r0:r1]))
    i = int(np.argmax(seg[r0:r1])) + r0
    return i / k, float(seg[i]) - base


def zoom_scale_from_roi(roi_bgr, panel=FIXED_PANEL):
    """读当前地图缩放尺度 s（`s = 参考图px / 面板px`）。返回 (s, 依据字符串)。

    失败返回 (None, 原因)：导航列不在标定位置（窗口化布局）、圆点峰不显著、ROI 太小。
    **调用方必须能回退**（回退到尺度搜索，见 CLAUDE.md 技术备忘⑥）。
    """
    ncc = nav_column_ncc(roi_bgr, panel)
    if ncc is None or ncc < NAV_NCC_MIN:
        return None, f"导航列NCC{ncc if ncc is None else round(ncc, 2)}(<{NAV_NCC_MIN})，滑条读数不可用"
    y, q = nav_dot_y(roi_bgr, panel)
    if y is None:
        return None, "导航列 ROI 太小"
    if q < ZOOM_DOT_MIN:
        return None, f"圆点峰不显著(高出基线{q:.0f}<{ZOOM_DOT_MIN:.0f})"
    s = _dot_y_to_s(y)
    edge = ""
    if not (ZOOM_DOT_Y_CLAMP[0] <= y <= ZOOM_DOT_Y_CLAMP[1]):
        edge = f"（超出标定段[{ZOOM_DOT_Y_CLAMP[0]:.0f},{ZOOM_DOT_Y_CLAMP[1]:.0f}]，取端点）"
    return s, f"滑条圆点y={y:.0f}→s={s:.2f}{edge}"



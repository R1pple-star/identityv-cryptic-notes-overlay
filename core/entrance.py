# -*- coding: utf-8 -*-
"""
入口引索匹配（新匹配模式）
========================
用户先选入口(正门/侧门/二楼)；游戏内捕获后，按检测到的入口白箭头图标裁出入口区域，
与各种子的入口引索裁图(entrance_index/{seed}_{type}.png)严格对比：
分类(迷雾剔除) + 多尺度 SQDIFF。排名种子。

原理：正门房间形状固定(不判别)，侧门/二楼房间形状各异(主要判别依据)。
故侧门/二楼入口匹配能定种子；正门通常分不出(各种子相似)。

§3.1：find_seed_by_entrance 除返回 (results, icon_pos, icon_score) 外，每条 result
额外带 (匹配尺度 s, minMaxLoc 参考图匹配位置 mloc)，供两段式对齐第一段构造 M1。
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import cv2
import numpy as np

from core.vision import (FIXED_PANEL, FOG_BGR, FOG_TOL, _find_icon, classify_region,
                         load_bgr, walls_as_floors)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)

# 入口引索目录（config.paths.entrance_index，相对项目根解析）。
ENTRANCE_INDEX_DIR = (ROOT / _CFG["paths"]["entrance_index"]).resolve()
ENTRANCE_FLOOR = {"正门": "一楼", "侧门": "一楼", "二楼": "二楼"}
# 入口裁图尺度接近(都~200px)，窄范围多尺度对齐。下限 0.35：游戏默认（不碰缩放）入口
# 状态 s≈0.40（2026-09-14 实测 6 局图标 k=1.0，s=0.4/k），旧下限 0.6 会把默认态全部
# 撞底钉住 → 匹分失真/同分退化。上界不变（观测域）。
SCALES_ENT = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
# 最小模板世界覆盖闸：模板边长 < 此值直接跳过该尺度。旧下限 0.6 隐式兼任防小模板假
# 获胜（小模板塞进均匀区 SQDIFF 偏低——17.13 实测种子16 在 s=0.55 以 112px 模板拿
# 0.039 反超真种子 0.046）；扩域到 0.35 后必须显式补上。实测真匹配模板世界覆盖
# 163-204 参考px（k∈[0.40,1.0] 经 ruler/基线裁样），150 = 下沿留 ~8% 余量。大裁样
# （k 放大后 ~464px）在 0.35 档模板 162px 仍可搜，204px 基线裁样 0.735 以下全挡。
MIN_TEMPLATE_PX = 150
# 图标当尺子·棘轮基准：k 高于此值才放大裁样（k≤此值保持基线 half=102px）。8 月全部
# 基线数据 k∈[0.40,0.45]（细网格 NCC 峰）→ 棘轮保证其逐像素零扰动；默认档 k≈1.0 →
# half≈232，世界覆盖回到 8 月水平（~184 参考px）。s 与裁框大小无关，扩大取样梯子救
# 不了缩放——缩放只有这条 ruler 路径；只放大不缩小，缩小会让小模板假获胜（17.13 实测
# 纯 ruler 缩到 186px → 错种子 0.031 反超）。
ICON_K_REF = float(_CFG["match"].get("icon_k_ref", 0.44))
# 渐变迷雾剔除：迷雾是渐变色，tol24 精确色只剔得净雾核，渐变外圈会被判成通路
# 污染掩膜 —— 故把雾核膨胀 _FOG_DILATE px 一并剔（渐变段与雾核空间相邻）。
# 该掩膜现只用于**快速失败闸**（样本结构占比 < sample_mask_min 即拒答）与单类退化检测；
# 入口排序本身走墙重合度量，不吃这张掩膜。
_FOG_DILATE = int(_CFG["match"].get("fog_dilate", 5))
# 图标锚定（2026-09-15 上线）：样本图标 ↔ 引索图标重合后**钉死平移只搜尺度**，尺度由
# 「图标当尺子」定 s0 = r_fine/k（两侧对同一模板 assets/_icon_entrance.png 做 NCC；k 由
# _find_icon 细网格量，r_fine 由引索侧细网格量、存 json 的 scale_fine）。依据
# experiments/e5_anchor_ab.py 的 2×2 实测：引索窗口放大到 2× 后**全搜会炸**（主集 top-1
# 3/3→2/3、控制组低分确信 1→5——搜索域变大=自相似错位有机可乘），锚定把它抹平回 3/3 与 1，
# 且控制组真种子排名由 7~24 名提到 1~7 名。关掉（false）→ 该口径无锚点即全体弃权。
_ANCHOR_PIN = bool(_CFG["match"].get("anchor_pin", False))
# _ANCHOR_BAND/_ANCHOR_STEP/_ANCHOR_N（尺度窄带 s0±0.06）已废弃：s0=r_fine/k 的残差是 ~0.09
# 量级，窄带兜不住（2026-09-16 实测把 17.11 真种子18 挤成种子15）。锚定现在只钉平移、
# 尺度走全域 SCALES_ENT。config 的 [match].anchor_band 已删。
#
# 入口层口径：**墙重合度量**（唯一口径，见下方长注）。旧的「5 类错配率」class 档与
# 「类号平方差」sqdiff 档已于 2026-09-16 连同 `_scan_seed` 一起删除 —— 回滚走 git。


def _crop_box(panel, icon_pos, half_frac=0.18, icon_k=None):
    """算「以图标为中心」的方形裁样框，返回 (x0, y0, side)（panel 局部坐标）。

    单独抽出来是为了给图标锚定算**样本图标在裁样内的偏移**（= icon_pos 的 panel 局部
    坐标 - (x0,y0)）——`_crop_around_icon` 只回 ndarray，锚定取分需要原点。语义与
    `_crop_around_icon` 完全同源，改动见其文档串。
    """
    px, py, pw, ph = panel
    cx, cy = icon_pos
    cxp, cyp = cx - px, cy - py
    kf = max(1.0, (icon_k / ICON_K_REF)) if icon_k else 1.0
    # 上限 316=228/0.36/2：放大档(k≈1.0→真s≈0.36-0.40)下侧长 632×0.36=228 恰可搜，
    # 再大会把真尺度挤出可行域（710px 裁样实测全'-'）。同时保护梯子：0.25/0.32 档
    # 在默认缩放下不至于全顶到同一个帽（284 帽实测压扁梯子 → 144715 丢失梯子救援）。
    half_cap = 316
    half = min(int(min(pw, ph) * half_frac * kf), half_cap)
    edge = min(int(min(pw, ph) * 0.25 * kf), half_cap)
    if cxp < edge or cxp > pw - edge or cyp < edge or cyp > ph - edge:
        half = edge  # 贴边 → 增大样本（17.13 中央不触发，基线不变）
    side = 2 * half
    x0 = min(max(cxp - half, 0), max(0, pw - side))
    y0 = min(max(cyp - half, 0), max(0, ph - side))
    return x0, y0, side


def _crop_around_icon(shot, panel, icon_pos, half_frac=0.18, icon_k=None):
    """以入口图标为中心裁方区域，裁样大小按图标检测尺度 k 放大（「图标当尺子·棘轮」）。

    游戏内放大（默认档 k≈1.0 / 玩家缩放）会让固定 half_frac 的裁样只覆盖一小块世界 →
    房形信息缺失、单类退化；且 s 与裁框大小无关，扩大取样救不了缩放。故 k>K_REF 时
    half ∝ k/ICON_K_REF 放大裁样，把世界覆盖拉回基线水平。
    **只放大不缩小（棘轮）**：k≤K_REF 保持基线 half_frac 行为——缩小裁样会丢信息让
    小模板假获胜（2026-09-14 实测 17.13 k=0.40 纯 ruler 缩到 186px → 错种子 0.031 反超），
    且 8 月全部基线数据 k≤0.45，棘轮保证其逐像素零扰动。
    half 另设上限 min(pw,ph)//2：k 量偏大/贴边档放大时防止裁样超过 228/SCALES_ENT[0]
    → 全尺度放不进引索 → 匹配恒空（同日实测 710px 裁样全'-'）。
    icon 贴面板边缘时自动增大到 0.25 档（同乘 k 放大）。方框 clamp 入面板不截断。
    icon_k=None（手框样本等无图标尺度的场景）→ 保持旧行为不缩放。"""
    px, py, pw, ph = panel
    x0, y0, side = _crop_box(panel, icon_pos, half_frac, icon_k)
    region = shot[py:py + ph, px:px + pw]
    return region[y0:y0 + side, x0:x0 + side].copy()


def sample_structure(crop):
    """样本的「探明结构」掩膜：迷雾(含渐变外圈)剔除后的墙/房/通路。

    迷雾是渐变色：tol24 精确色只剔得掉雾核，渐变外圈会被 classify_region 判成
    通路(cls3) 污染 mask——故把雾核膨胀 _FOG_DILATE px 一并剔除（渐变段与雾核
    空间相邻）。参数见 config [match]。

    返回 (cls, mask_u8)。**不再回传权重**：权重原本只喂入口层的 class 口径
    （贴墙增益/房间加权），该口径 2026-09-16 已随 `_scan_seed` 一起删除；入口排序
    现在走墙重合度量，只吃两侧的墙掩膜（classify_region(...)==5）。"""
    cls_raw = classify_region(crop)
    cls = walls_as_floors(cls_raw, crop)
    fog = (np.abs(crop.astype(np.int16) - FOG_BGR).sum(axis=2) < FOG_TOL)
    if _FOG_DILATE > 0:
        fog = cv2.dilate(fog.astype(np.uint8),
                         np.ones((_FOG_DILATE, _FOG_DILATE), np.uint8)) > 0
    mask = (((cls == 1) | (cls == 2) | (cls == 3)) & (~fog)).astype(np.uint8)
    return cls, mask


# ===== 墙重合度量（2026-09-16 用户提议，入口层**唯一**口径）=====
# 动机：5 类标签的"错配率"在大片均匀区（长走廊/大房间）没有判别力 —— 那里挪 40px 还是
# "全一致"，正是入口样本最常见的形态；而"墙"是 1-9px 的细线，一挪就错开。
# 做法（只比几何，不比色块）：
#   ① 两侧各取墙掩膜 `classify_region(...)==5`（墙色已标定、**雾免疫**：雾最亮~75 < 墙带下沿~95）
#   ② 样本墙按入口尺度 s 缩放，锚定落点对齐（样本图标 ↔ 引索图标）
#   ③ 双向覆盖率：自己有多少比例的墙落在「对方墙膨胀 r 像素」带上，两个方向取平均
#   ④ 容差 r 吸收厚度差（游戏内墙 5-9px vs 参考墙 1-3px），故不要求逐像素相等
#   ⑤ 分 = 1 - 双向覆盖率（沿用"越小越匹配"，下游排序/闸门语义不变）
# 实测（2026-09-16，6 张有真值的样本）：真种子**5/5 排第一**，领先次名 17%~58%；
# 位置敏感对照：引索墙平移 40px 后覆盖率 0.83→0.30 / 0.78→0.36 / 0.56→0.18。
# 天然拒答：样本几乎没墙（未探明）时全部尺度跳过 ⇒ 该种子弃权 ⇒ 全体弃权则无结果，
# 正好落在"刚进门信息缺失应拒答"上。
_WALL_RADII = (2, 3, 4, 6)   # 膨胀容差(引索像素)
_MIN_WALL_PX = 30            # 单侧墙像素下限：低于此值覆盖率没统计意义，该尺度弃用


def _cover(a, b, r):
    """a 中有多少比例落在 dilate(b, r) 上。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return float((cv2.dilate(b.astype(np.uint8), k) > 0)[a].mean()) if a.any() else 0.0


def wall_overlap(g, r_wall):
    """双向墙覆盖率均值 ∈[0,1]，越大越像。任一侧墙太少 → None（不可判）。"""
    if g.sum() < _MIN_WALL_PX or r_wall.sum() < _MIN_WALL_PX:
        return None
    return float(np.mean([np.mean([_cover(g, r_wall, r), _cover(r_wall, g, r)])
                          for r in _WALL_RADII]))


def _scan_seed_wall(game_wall, ref_wall, scales, anchor, icon_off):
    """墙重合档：扫 `scales`，返回 (1-双向墙覆盖率, 获胜尺度, 落点)（越小越好）。

    只在锚定落点处评一次（不做平移搜索）—— 墙是尖的，但迷宫走廊网格自相似，全搜仍可能
    锁进幽灵相位，锚点仍是唯一无歧义约束。锚点缺失时调用方应弃权（不许回退全搜：统计量
    不同就不可比，见 find_seed_by_entrance 注释）。
    """
    best = (1e9, None, None)
    gh, gw = game_wall.shape[:2]
    for s in scales:
        tw, th = int(gw * s), int(gh * s)
        if tw < 10 or th < 10 or th > ref_wall.shape[0] or tw > ref_wall.shape[1]:
            continue
        if min(tw, th) < MIN_TEMPLATE_PX:
            continue
        g = cv2.resize(game_wall.astype(np.uint8), (tw, th),
                       interpolation=cv2.INTER_NEAREST) > 0
        if g.sum() < _MIN_WALL_PX:
            continue
        ax = int(round(anchor[0] - icon_off[0] * (tw / gw)))
        ay = int(round(anchor[1] - icon_off[1] * (th / gh)))
        if not (0 <= ax <= ref_wall.shape[1] - tw and 0 <= ay <= ref_wall.shape[0] - th):
            continue
        ov = wall_overlap(g, ref_wall[ay:ay + th, ax:ax + tw])
        if ov is not None and 1.0 - ov < best[0]:
            best = (1.0 - ov, s, (ax, ay))
    return best



def find_seed_by_entrance(shot, lib, entrance_type: str,
                          index_dir=ENTRANCE_INDEX_DIR, panel=FIXED_PANEL,
                          top_n=6, sample_crop=None, sample_origin=None):
    """入口引索匹配。返回 ([(score, seed, 方向-门, 楼层, s, mloc), ...], icon_pos, icon_score, icon_k)。

    score 越小越匹配。entrance_type ∈ {正门, 侧门, 二楼}。
    s 为获胜尺度(SCALES_ENT)，mloc=(mlx,mly) 为该尺度下 matchTemplate 在参考裁图里的
    argmin 位置；二者供 build_entrance_transform 构造两段式对齐第一段 M1。缺时为 None。
    icon_k 为图标检测获胜尺度（「图标当尺子」，裁样按它缩放），缺时 None。
    sample_crop: 手动框选的入口样本(HxWx3 BGR)，阶段3 手框纠错用；给了则用它做
    in_cls/in_mask(跳过 _crop_around_icon)，None 则自动以图标为中心裁。默认 None=基线。
    sample_origin: 手框样本左上角的**屏幕坐标**(x0,y0)。墙重合口径**必须**有它：
    锚点=「样本图标 ↔ 引索图标」，而图标在样本内的偏移要靠样本原点算得。
    2026-09-16 实机前该参数不存在 ⇒ 手框路径 icon_off 恒 None ⇒ 墙口径下每个种子都
    弃权 ⇒ 手框**恒返回空**（日志里 9/9 次「手框样本仍无匹配」即此因，与框大小无关）。
    """
    icon_pos, icon_score, icon_k = _find_icon(shot, *panel)
    if icon_pos is None and sample_crop is None:
        return [], None, 0.0, None
    in_crop = sample_crop if sample_crop is not None else _crop_around_icon(
        shot, panel, icon_pos, icon_k=icon_k)
    _in_cls, in_mask = sample_structure(in_crop)
    if in_mask.sum() < 100:
        return [], icon_pos, icon_score, icon_k
    # 样本侧墙掩膜（雾免疫：雾最亮~75 < 墙带下沿~95，故"有墙 ⇒ 已探明"严格成立）
    game_wall = classify_region(in_crop) == 5

    fl = ENTRANCE_FLOOR[entrance_type]
    # 样本图标在样本内的偏移（屏幕px）——锚定档的平移基准。手框路径同样要算（见下面注释）。
    icon_off = None
    if icon_pos is not None:
        if sample_crop is None:
            bx0, by0, _side = _crop_box(panel, icon_pos, 0.18, icon_k=icon_k)
            icon_off = ((icon_pos[0] - panel[0]) - bx0, (icon_pos[1] - panel[1]) - by0)
        elif sample_origin is not None:
            # 手框：框原点即样本左上角 ⇒ 图标在样本内偏移 = 图标屏幕坐标 − 框原点。
            # 与自动路径同一个量纲（样本内像素），故锚定公式一字不改。缺 sample_origin
            # （老调用方）仍是 None ⇒ 墙口径下全体弃权，不静默退化成别的东西。
            icon_off = (icon_pos[0] - sample_origin[0], icon_pos[1] - sample_origin[1])

    results = []
    for seed in lib.seeds():
        idx_path = index_dir / f"{seed}_{entrance_type}.png"
        if not idx_path.exists():
            continue
        idx_bgr = load_bgr(str(idx_path))
        ref_wall = classify_region(idx_bgr) == 5

        # 锚点 = 引索图标在裁图内的偏移 (rx,ry)。尺度**不再**由 r_fine/k 提示：窄带口径
        # 废弃后尺度走全域 SCALES_ENT，锚定只负责钉死平移。
        anchor = None
        if _ANCHOR_PIN and icon_off is not None:
            js = load_index(seed, index_dir)
            ent = js.get(entrance_type) if js else None
            if ent and "box" in ent:
                anchor = (ent["cx"] - ent["box"][0], ent["cy"] - ent["box"][1])

        # **必须锚定**：墙重合只在锚点处评一次。无锚点 = 该种子弃权，绝不回退全搜 ——
        # 全搜取的是每个尺度的全局最小、恒 ≤ 锚点那一点的值，于是"锚点用不了的种子"白拿
        # 更宽的自由度，靠不公平优势压过真种子（17.11 实测：种子15 锚点 0/13 档可用，
        # 回退全搜得 0.109，以 0.005 之差翻掉真种子18 的锚定值 0.114）。统计量不同就不可比；
        # 锚点不可用就不参与排名。这同时是治自相似「幽灵相位」的唯一无歧义约束。
        best_sc, best_s, best_mloc = (
            _scan_seed_wall(game_wall, ref_wall, SCALES_ENT, anchor, icon_off)
            if anchor is not None else (1e9, None, None))
        info = lib.get(seed, "一楼")
        if best_s is None:
            # 无任何尺度放得下（样本大于该种子引索裁图，如手框过大）：不计入。
            # 旧版计入 1e9 哨兵分，多种子全 1e9 时排序无意义 → 假种子居中显示（2026-09-13 实测）。
            continue
        results.append((best_sc, seed, info.key if info else str(seed), fl,
                        best_s, best_mloc))
    results.sort(key=lambda x: x[0])
    return results[:top_n], icon_pos, icon_score, icon_k


def score_desc(score) -> str:
    """把入口分翻译成人话，供 UI 日志/状态栏。分 = 1 − 墙双向覆盖率，越小越好。"""
    if score is None:
        return "-"
    return f"墙重合{max(0.0, 1.0 - score):.0%}"


def load_index(seed: int, index_dir=ENTRANCE_INDEX_DIR) -> dict | None:
    """加载某种子的 {seed}_index.json（含各入口的 cx/cy/box）。不存在返回 None。"""
    p = index_dir / f"{seed}_index.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_entrance_transform(best, entrance_type: str, index_json: dict,
                             icon_pos, panel=FIXED_PANEL):
    """两段式对齐第一段：由入口引索匹配结果构造相似变换 M1(ref→屏幕，北朝上无旋转)。

    best: find_seed_by_entrance 的单条结果 (score, seed, key, fl, s, mloc)。
    对应关系：屏幕入口图标 icon_pos ↔ 参考图入口图标 (cx,cy)（来自 index_json）；
    尺度：入口匹配尺度 s（各裁图均原生分辨率，s 与 find_overlay_transform 的 s 同义，
    见 CLAUDE.md 两段式备忘）。M1 = [[1/s,0, ix-cx/s],[0,1/s, iy-cy/s]]。
    返回 (M1(2x3), hint_s=s) 或 None（缺尺度/缺入口坐标）。
    """
    _sc, _seed, _key, _fl, s, _mloc = best
    if s is None:
        return None
    entry = index_json.get(entrance_type)
    if entry is None:
        return None
    cx, cy = entry["cx"], entry["cy"]
    a = 1.0 / s                 # M[0,0] = inv_s = 1/s（ref→屏幕）
    ix, iy = float(icon_pos[0]), float(icon_pos[1])
    tx = ix - a * cx
    ty = iy - a * cy
    M1 = np.array([[a, 0.0, tx], [0.0, a, ty]], dtype=np.float64)
    return M1, s


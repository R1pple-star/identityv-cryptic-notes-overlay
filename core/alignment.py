# -*- coding: utf-8 -*-
"""
对齐引擎（从 find_seed_submap.py + matcher.py 抽出，Phase 2.2）
=====================================================
投影重合核心：用迷雾剔除后的探明模板，对参考图(内容裁剪)做掩膜匹配，
取最佳 (尺度,位置) 构造相似变换 M(北朝上、仅缩放+平移)，warp 整张参考图到屏幕。
以及手动标定(affine_from_points) / 点变换(transform_point) / RGBA 悬浮层生成。

搬家自 find_seed_submap.py（find_overlay_transform 及附属）与 matcher.py
（affine_from_points/transform_point/map_to_overlay_rgba/auto_align_overlay），
纯移动、逻辑不变。PAD 由函数局部提为模块常量。
"""
from __future__ import annotations

import cv2
import numpy as np

from core.vision import (FIXED_PANEL, FOG_BGR, classify_region, content_bbox,
                         walls_as_floors)


# 尺度搜索：游戏内地图缩放随玩家平移/缩放而变，真实尺度常落在 0.7~0.85(面板适配)。
# 旧版只试 0.4/0.6/0.8/1.0 四档，常擦肩真实尺度→大模板错位、边缘像素全错、分数偏高。
# 细化到 0.05 步长后 17.13 真种子 0.131→0.106，与第二名拉开(0.264)。
SCALES = (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)

# 对齐(投影重合)专用尺度域：上界放宽到 1.5，下界放宽到 0.30。s = ref像素/面板像素，
# 实测视图 0.75(整图适配)~0.99(放大)；游戏内还能继续放大，17.13 放大 8% 就要 s=1.065，
# 超原上界 1.0 会让对齐滑到边界判不可靠。下界：游戏默认（不碰缩放）入口状态 s≈0.40
# （2026-09-14 实测），正贴旧下界 0.40 且 hint_s 细搜会被 max(s_lo,…)) 截断，故放到
# 0.30 留余量。种子匹配仍用 SCALES(观测域即可，多搜只增假阳风险)。
ALIGN_SCALES = tuple(np.round(np.arange(0.30, 1.501, 0.05), 2))

# 内容四周补黑边放宽可行域（刀锋问题）：玩家缩放到「整图刚好适配」时，模板尺寸≈内容尺寸，
# int 取整可能让 tw>rw → 尺度被整体跳过、细搜可行域只剩单侧。坐标推导按 (cx0-PAD+ml) 补偿。
PAD = 12

# 图标锚点窗搜半径：hint_icon 给定时平移**钉死**在锚定位置(0=不搜平移，只搜尺度)。
# 迷宫走廊网格自相似，填充 SQDIFF 景观极平——全图搜索锁进错位 146px 的「幽灵相位」
# （分 0.046/overlap 0.80 双闸全过，2026-09-15 实测 17.13），窗搜给 4px 余量也会漂到
# 次级幽灵(墙重合 60%→4%)。图标锚定+钉平移: score 0.046/ov 1.00/s 0.986/墙重合 60%。
ANCHOR_RADIUS = 0


def _revealed_mask(region, cls_r):
    """已探明真结构掩膜：分类为 墙/房/通路 且 **非迷雾**。

    迷雾色 BGR(58,47,37) 冷暖度 -21，会被 classify_region 误判成「通路(cls=3)」。
    用颜色距离(FOG_BGR, tol24)显式剔除迷雾，避免迷雾污染模板。
    """
    fog = (np.abs(region.astype(np.int16) - FOG_BGR).sum(axis=2) < 24)
    return (((cls_r == 1) | (cls_r == 2) | (cls_r == 3)) & (~fog)).astype(np.uint8)


def _match_at_scale(ref_f, temp_cls, temp_mask, bw, bh, s, ref_shape,
                    center=None, radius=0):
    """单尺度掩膜 SQDIFF 匹配。

    返回 (score, ml_x, ml_y, res, msum, (off_x, off_y)) 或 None(尺度越界/掩膜太小)。
    score = min(SQDIFF)/掩膜数，越小越匹配；ml_x/ml_y 为匹配左上角在 ref 内容里的
    **全图**整数位置；res 为响应面（center 给定时是窗口局部坐标系，配 off 用）。
    center: 图标锚定的模板左上角(ref坐标)——给定时平移只在该点 ±radius 邻域内搜
    （防自相似迷宫的幽灵相位，见 ANCHOR_RADIUS 注）。"""
    tw, th = int(bw * s), int(bh * s)
    if tw < 10 or th < 10 or th > ref_shape[0] or tw > ref_shape[1]:
        return None
    tcl = cv2.resize(temp_cls, (tw, th), interpolation=cv2.INTER_NEAREST)
    tmk = cv2.resize(temp_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    msum = float(tmk.sum())
    if msum < 50:
        return None
    off_x = off_y = 0
    if center is None:
        res = cv2.matchTemplate(ref_f, tcl.astype(np.float32), cv2.TM_SQDIFF,
                               mask=tmk.astype(np.float32))
    else:
        cx, cy = center
        x0 = max(0, int(cx - radius))
        y0 = max(0, int(cy - radius))
        x1 = min(ref_shape[1], int(cx + tw + radius))
        y1 = min(ref_shape[0], int(cy + th + radius))
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        if radius == 0 and (x0 != int(cx) or y0 != int(cy)):
            return None   # 钉平移模式: 锚点出界该尺度直接跳过, 不许钳位漂移
        res = cv2.matchTemplate(ref_f[y0:y1, x0:x1], tcl.astype(np.float32),
                                cv2.TM_SQDIFF, mask=tmk.astype(np.float32))
        off_x, off_y = x0, y0
    mn, _, _, ml = cv2.minMaxLoc(res)
    return mn / msum, ml[0] + off_x, ml[1] + off_y, res, msum, (off_x, off_y)


def _subpixel_min(res, mx, my):
    """对 SQDIFF 响应面在整数极小点做二次曲线拟合，得亚像素 (x, y)。

    分类 SQDIFF 响应面在极小点附近近似抛物面（平移半个像元只改边界像元），
    用 min 及其左右/上下邻值拟合顶点。偏移限幅 ±0.5 像元，退化时不动。
    """
    def _parabola(v_minus, v0, v_plus):
        denom = v_minus - 2.0 * v0 + v_plus
        if abs(denom) < 1e-9:
            return 0.0
        off = 0.5 * (v_minus - v_plus) / denom
        return max(-0.5, min(0.5, off))
    rh, rw = res.shape
    v0 = float(res[my, mx])
    xm = float(res[my, mx - 1]) if mx - 1 >= 0 else v0
    xp = float(res[my, mx + 1]) if mx + 1 < rw else v0
    ym = float(res[my - 1, mx]) if my - 1 >= 0 else v0
    yp = float(res[my + 1, mx]) if my + 1 < rh else v0
    return mx + _parabola(xm, v0, xp), my + _parabola(ym, v0, yp)


def find_overlay_transform(shot_bgr, ref_bgr, panel=FIXED_PANEL, scales=ALIGN_SCALES,
                           region=None, hint_s=None, hint_icon=None, fast=False):
    """求参考图→屏幕的最佳对齐变换 M(2x3)，使参考图与游戏内已探明结构重合。

    用迷雾剔除后的探明模板，对参考图(内容裁剪)做掩膜匹配，取最佳
    (尺度,位置)构造相似变换。地图恒北朝上，故无旋转，仅缩放+平移。
    返回 (M, score, overlap) 或 None(探明不足)。score 越小越重合(<0.10 较可信)。

    region: 已裁好的面板 BGR(跟随循环只截面板区域时用)；None 则从 shot_bgr 按 panel 裁。
    hint_s: 上次对齐的尺度提示。给了就跳过 12 档粗搜，直接在 hint±0.06 细搜(省 ~0.4s)。
    hint_icon: (ref_cx, ref_cy, screen_ix, screen_iy) 入口图标锚点四元组(M1 的核心)。
            给定时平移搜索限制在锚定位置 ±ANCHOR_RADIUS 邻域，且粗搜全尺度窗扫——
            迷宫走廊网格自相似，全图搜索可锁进错位 146px 的「幽灵相位」且双闸拦不住
            (2026-09-15 实测 17.13：分 0.046/overlap 0.80 全过但墙重合仅 4%，
            图标锚定后 67%)。图标是唯一无歧义锚点。
    fast:   只在 hint_s 单尺度匹配(平移跟踪用，~30ms；须与 hint_s 同给)。
            玩家平移地图不改尺度，单档即可拿到精确平移；缩放变了分会上来，由调用方降级。

    尺度搜索三段式(粗 0.05→细 0.01→微 0.002)+亚像素位置(二次曲线拟合)：
    旧版只搜 0.05 粗档，真实尺度常落在档间(如 17.13 真尺度 0.986 落在 0.95/1.00 间)，
    粗档最优 0.95(score 0.106)实为擦肩，远端可偏 20+ 像元。细化后 0.986(score 0.045)，
    远端对齐显著改善。微搜(0.002)用来逼近真极小(亚像素尺度二次曲线在 0.01 档上不对称、
    不可靠，故用细网格而非抛物插值)。

    坐标推导：模板(面板坐标 bx0,by0,宽bw 高bh) 缩放 s 后在 ref 内容里匹配到
    (ml_x,ml_y)，故 ref内容坐标(rx,ry) ↔ 面板(bx0+(rx-ml_x)/s, by0+(ry-ml_y)/s)，
    再加面板偏移(px,py)得屏幕坐标。ref 全图(X,Y) 再减内容裁剪原点(cx0,cy0)。
    """
    px, py, pw, ph = panel
    if region is None:
        if shot_bgr is None:
            return None
        region = shot_bgr[py:py + ph, px:px + pw]
    if region is None or region.shape[0] < ph * 0.5 or region.shape[1] < pw * 0.5:
        return None  # 面板裁空(分辨率不符等)，不做无效匹配
    cls_r = walls_as_floors(classify_region(region), region)
    revealed = _revealed_mask(region, cls_r)
    if revealed.sum() < 300:
        return None
    ys, xs = np.where(revealed > 0)
    by0, by1 = int(ys.min()), int(ys.max())
    bx0, bx1 = int(xs.min()), int(xs.max())
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    temp_cls = cls_r[by0:by1 + 1, bx0:bx1 + 1]
    temp_mask = revealed[by0:by1 + 1, bx0:bx1 + 1]

    cx0, cy0, cw, ch = content_bbox(ref_bgr)
    # 内容四周补黑边再匹配：玩家缩放到「整图刚好适配」时，模板尺寸≈内容尺寸，
    # int 取整可能让 tw>rw → 尺度被整体跳过、细搜可行域只剩单侧(真尺度刀锋)。
    # 补边放宽可行域；黑边 cls=0，掩膜像素不受影响。坐标推导按 (cx0-PAD+ml_x) 补偿。
    ref_c = cv2.copyMakeBorder(ref_bgr[cy0:cy0 + ch, cx0:cx0 + cw],
                               PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ref_cls = walls_as_floors(classify_region(ref_c), ref_c)
    ref_f = ref_cls.astype(np.float32)
    ref_shape = ref_cls.shape

    def _anchored_center(s):
        """该尺度下图标锚定的模板左上角(ref 补边坐标系)。ref 补边坐标 = 原图-(cx0-PAD)，
        图标坐标(cx,cy)是原图系 ⇒ 先折算。ref=(panel-t)·s 且图标钉死 ⇒
        模板TL(bx0,by0)panel → (icx-(cx0-PAD))+(bx0-icon_panel)·s。"""
        icx, icy, isx, isy = hint_icon
        return (icx - (cx0 - PAD) + (bx0 - (isx - px)) * s,
                icy - (cy0 - PAD) + (by0 - (isy - py)) * s)

    def _search(scale_list):
        """在给定尺度列表上做掩膜匹配，返回最佳 (score,s,ml_x,ml_y,res,msum,off) 或 None。"""
        best = None
        for s in scale_list:
            kw = ({"center": _anchored_center(float(s)), "radius": ANCHOR_RADIUS}
                  if hint_icon is not None else {})
            r = _match_at_scale(ref_f, temp_cls, temp_mask, bw, bh, s, ref_shape, **kw)
            if r is None:
                continue
            if best is None or r[0] < best[0]:
                best = (r[0], s, r[1], r[2], r[3], r[4], r[5])
        return best

    s_lo, s_hi = float(scales[0]), float(scales[-1])

    if fast:
        # 快速档：只在 hint_s 单尺度匹配(平移跟踪)。缩放变了分会上来，由调用方降级重搜。
        if hint_s is None:
            return None
        kw = ({"center": _anchored_center(float(hint_s)), "radius": ANCHOR_RADIUS}
              if hint_icon is not None else {})
        r = _match_at_scale(ref_f, temp_cls, temp_mask, bw, bh, float(hint_s), ref_shape, **kw)
        if r is None:
            return None
        sc, s, ml_x, ml_y, res, _msum, _off = r[0], float(hint_s), r[1], r[2], r[3], r[4], r[5]
    else:
        # 有图标锚点 → 粗搜也走窗搜且全尺度扫（入口匹配 s 与真实 s 可差 ~0.09，17.13
        # 实测 0.9 vs 0.986——锚住平移后尺度交给全档扫描，别信 hint_s 的窄窗细搜）。
        if hint_s is None or hint_icon is not None:
            coarse = _search(scales)
            if coarse is None:
                return None
            s0 = coarse[1]
        else:
            # 仅尺度提示(跟随循环已知道上次尺度)：跳过粗搜
            s0 = float(hint_s)
        # 2) 细搜(0.01 步，粗/hint 最优 ±0.06，覆盖 ±1.2 个粗档以避免落入局部极小)
        fine = _search(np.arange(max(s_lo, s0 - 0.06), min(s_hi, s0 + 0.06) + 1e-9, 0.01))
        if hint_s is None or hint_icon is not None:
            base = fine if (fine and fine[0] <= coarse[0]) else coarse
        else:
            base = fine
            if base is None:
                return None
        # 3) 微搜(0.002 步，细最优 ±0.012)：逼近真极小
        s1 = base[1]
        micro = _search(np.arange(max(s_lo, s1 - 0.012), min(s_hi, s1 + 0.012) + 1e-9, 0.002))
        best = micro if (micro and micro[0] <= base[0]) else base
        sc, s, ml_x, ml_y, res, _msum, _off = best
    # 4) 亚像素位置：在微搜响应面上做二次曲线拟合(限幅 ±0.5 像元)。
    #    窗搜时 res 是窗口局部坐标，先减 off 拟合再加回。
    spx, spy = _subpixel_min(res, int(ml_x - _off[0]), int(ml_y - _off[1]))
    ml_x, ml_y = spx + _off[0], spy + _off[1]

    inv_s = 1.0 / s
    # ml 在补边坐标系里，内容原点 = (cx0-PAD, cy0-PAD)
    tx = px + bx0 - (cx0 - PAD + ml_x) * inv_s
    ty = py + by0 - (cy0 - PAD + ml_y) * inv_s
    # 可靠性闸：已探明像素经逆变换(X=(screen-tx)*s)后，查参考图「内容掩膜(非背景)」
    # 看是否落在真实内容上。只查内容外接框太松(框很大，几乎都落框内，不判别)。
    # 小模板侥幸低分但变换错位时(如 7% 的 23.43)，探明会落到参考图背景上 → overlap 低。
    # 注：细搜会压低小模板分数，故 overlap 闸是必备二次验证(挡侥幸低分错配)。
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_content = ref_gray > 40
    rev_ys, rev_xs = np.where(revealed > 0)
    ref_xc = (px + rev_xs - tx) * s
    ref_yc = (py + rev_ys - ty) * s
    rxi = ref_xc.astype(np.int32)
    ryi = ref_yc.astype(np.int32)
    h0, w0 = ref_content.shape
    ok = (rxi >= 0) & (rxi < w0) & (ryi >= 0) & (ryi < h0)
    overlap = float(ref_content[ryi[ok], rxi[ok]].mean()) if ok.any() else 0.0
    M = np.array([[inv_s, 0.0, tx], [0.0, inv_s, ty]], dtype=np.float64)
    return M, sc, overlap


# ====== 以下从 matcher.py 搬入（标定 / 点变换 / RGBA 悬浮层）======

def affine_from_points(src_pts, dst_pts) -> np.ndarray:
    """由对应点估算仿射变换矩阵 M (2x3)，把 src(参考图) 映射到 dst(屏幕)。

    - 2 点：相似变换(缩放+旋转+平移)
    - >=3 点：仿射变换
    """
    src = np.float32(src_pts).reshape(-1, 1, 2)
    dst = np.float32(dst_pts).reshape(-1, 1, 2)
    if len(src_pts) >= 3:
        return cv2.getAffineTransform(src[:3], dst[:3])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return M


def transform_point(M: np.ndarray, pt) -> tuple[float, float]:
    """用 M 变换单个点。"""
    x, y = pt
    p = np.array([x, y, 1.0])
    q = M @ p
    return float(q[0]), float(q[1])


def map_to_overlay_rgba(map_path: str, M: np.ndarray, screen_w: int, screen_h: int,
                        bg_thresh: int = 55, wall_alpha: int = 160,
                        bg_alpha: int = 0) -> np.ndarray:
    """把参考地图 warp 到屏幕，生成 RGBA 悬浮图层。

    M 是「参考图->屏幕」正向相似变换(screen = X·inv_s + t，来自 find_overlay_transform)。
    cv2.warpAffine 默认(不加 WARP_INVERSE_MAP)传的就是 src->dst 正向矩阵，直传 M 即可；
    若传 M 的逆反而会把地图投错位置(踩过坑，勿改)。

    深色背景(迷宫的空地/底色)设为透明，墙体/房间等亮色内容设为半透明，
    这样悬浮层不会盖黑游戏画面，只把「路线/房间结构」显示出来。

    返回 shape=(screen_h, screen_w, 4) 的 uint8 RGBA 数组。
    """
    from PIL import Image
    a = np.asarray(Image.open(map_path).convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    warped = cv2.warpAffine(a, M, (screen_w, screen_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_gray = cv2.warpAffine(gray, M, (screen_w, screen_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # alpha 通道：亮的内容可见，暗背景透明
    alpha = np.where(warped_gray > bg_thresh, wall_alpha, bg_alpha).astype(np.uint8)

    rgba = np.dstack([warped[:, :, ::-1], alpha])  # BGR->RGB，再加 alpha
    return rgba


def auto_align_overlay(ref_bgr: np.ndarray, panel, rotate: int = 0,
                       wall_alpha: int = 160) -> np.ndarray:
    """自动对齐：把参考图按 rotate 度旋转后，缩放到迷雾面板位置，生成悬浮层。

    ref_bgr: 参考完整地图(BGR)
    panel:   (x0, y0, w, h) 迷雾面板在屏幕上的位置
    rotate:  0/90/180/270 度（参考图按「面向方向」命名，通常 0 即可）
    返回 (rgba, (放置偏移x, 放置偏移y))
    """
    # 1. 裁掉参考图留白，只留迷宫内容
    x0, y0, cw, ch = content_bbox(ref_bgr)
    content = ref_bgr[y0:y0 + ch, x0:x0 + cw]

    # 2. 旋转
    if rotate == 90:
        content = cv2.rotate(content, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        content = cv2.rotate(content, cv2.ROTATE_180)
    elif rotate == 270:
        content = cv2.rotate(content, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 3. 等比缩放到迷雾面板内（不拉伸，不裁剪）
    px0, py0, pw, ph = panel
    ch, cw = content.shape[:2]
    scale = min(pw / cw, ph / ch) if (cw > 0 and ch > 0) else 1.0
    nw, nh = int(cw * scale), int(ch * scale)
    content = cv2.resize(content, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 4. 生成 RGBA 悬浮层（背景透明，内容半透明），居中放入面板
    gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray > 55, wall_alpha, 0).astype(np.uint8)
    rgba = np.dstack([content[:, :, ::-1], alpha])  # BGR->RGB + alpha
    return rgba, (px0 + (pw - nw) // 2, py0 + (ph - nh) // 2)

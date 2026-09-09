# -*- coding: utf-8 -*-
"""验证自动跟随的核心对齐原语(find_overlay_transform 的 hint/fast 路径)+ 渲染约定。

全部用真实截图验证。曾尝试「合成面板」(把参考图按已知变换投影成游戏面板)：无论
最近邻/线性/4x超采样+AREA，都无法复刻「游戏原生渲染 vs 手绘参考」的噪声关系——
合成面板在真位置的匹配分反而不极小(走样/混色伪差异)，据此断言会误报，已废弃。

  A) 真截图全搜过闸：17.13/17.11(种子18/一楼)；
  B) 同一面板 fast 重对齐：漂移 <0.5px(平移不改尺度，单档即精确)；
  C) 真实跨截图平移恢复：17.13→17.11 是同局两次截图，其间玩家平移了地图
     (Δt≈(55,222))；用 17.13 的尺度提示对 17.11 做 fast，应精确恢复 17.11
     独立全搜的结果(<2px)；
  D) 渲染约定：warpAffine 默认传 src->dst 正向 M(直传 find_overlay_transform
     的 M 即可，**勿传其逆**——实测传逆会把地图投到屏幕左上角外)。判据：
     overlay 在面板内有可见内容；按 M 把探明像素映回参考图落内容的比例
     (overlap)应高于故意偏移 40px 的对照；
  E) 低探明鲁棒性：把 17.13 的探明随机抹掉 70%(留真实像素)，全搜仍应恢复
     同一变换(<3px)——跟随循环里探明增长/掩膜变化不影响稳定。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.alignment import find_overlay_transform, _revealed_mask
from core.map_library import MapLibrary
from core.vision import FIXED_PANEL, classify_region, load_bgr
from core.alignment import map_to_overlay_rgba

MAP_DIR = r"D:\Pictures\20260818摸金地图"
SHOT_DIR = r"D:\Videos\NVIDIA\IdentityV"
RNG = np.random.default_rng(42)


def gates(sc, ov):
    return sc < 0.30 and ov >= 0.4


def shot_region(dt, ts):
    shot = load_bgr(f"{SHOT_DIR}\\IdentityV Screenshot 2026.08.{dt} - {ts}.png")
    px, py, pw, ph = FIXED_PANEL
    return shot[py:py + ph, px:px + pw]


def overlap_of(ref, region, M):
    """按 M(正向: X=(screen-t)*s) 把探明像素映回参考图，落内容(非暗)的比例。"""
    px, py, pw, ph = FIXED_PANEL
    cls_r = classify_region(region)
    revealed = _revealed_mask(region, cls_r)
    ref_content = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) > 40
    h0, w0 = ref_content.shape
    ys, xs = np.where(revealed > 0)
    X = ((px + xs - M[0, 2]) / M[0, 0]).astype(np.int32)
    Y = ((py + ys - M[1, 2]) / M[1, 1]).astype(np.int32)
    ok = (X >= 0) & (X < w0) & (Y >= 0) & (Y < h0)
    return float(ref_content[Y[ok], X[ok]].mean()) if ok.any() else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib = MapLibrary.load(MAP_DIR)
    px, py, pw, ph = FIXED_PANEL
    ref18 = load_bgr(str(lib.get(18, "一楼").path))
    reg13 = shot_region("26", "17.13.06.98")
    reg11 = shot_region("26", "17.11.35.17")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  {'✓' if ok else '✗'} {name} {detail}")

    # ---- A) 真截图全搜过闸 ----
    M13, sc13, ov13 = find_overlay_transform(None, ref18, FIXED_PANEL, region=reg13)
    M11, sc11, ov11 = find_overlay_transform(None, ref18, FIXED_PANEL, region=reg11)
    check("17.13 全搜过闸", gates(sc13, ov13), f"score={sc13:.3f} overlap={ov13:.2f}")
    check("17.11 全搜过闸", gates(sc11, ov11), f"score={sc11:.3f} overlap={ov11:.2f}")
    s13 = 1.0 / M13[0, 0]

    # ---- B) 同一面板 fast 不漂移 ----
    Mf, scf, ovf = find_overlay_transform(None, ref18, FIXED_PANEL, region=reg13,
                                          hint_s=s13, fast=True)
    drift = abs(Mf[0, 2] - M13[0, 2]) + abs(Mf[1, 2] - M13[1, 2])
    check("17.13 同面板fast不漂移", drift < 0.5 and gates(scf, ovf), f"漂移={drift:.2f}px")

    # ---- C) 真实跨截图平移恢复：17.13 的 hint 用到 17.11 ----
    Ma, sca, ova = find_overlay_transform(None, ref18, FIXED_PANEL, region=reg11,
                                          hint_s=s13, fast=True)
    dt_ = abs(Ma[0, 2] - M11[0, 2]) + abs(Ma[1, 2] - M11[1, 2])
    ds = abs(1.0 / Ma[0, 0] - 1.0 / M11[0, 0])
    pan = (M11[0, 2] - M13[0, 2], M11[1, 2] - M13[1, 2])
    check("17.13→17.11 真实平移恢复", dt_ < 2.0 and ds < 0.005 and gates(sca, ova),
          f"期间平移Δt=({pan[0]:.0f},{pan[1]:.0f}) 恢复误差={dt_:.2f}px ds={ds:.4f}")

    # ---- D) 渲染约定：warpAffine 默认传 src->dst 正向 M(直传即可，勿传逆)。
    # 判据：overlay 在面板内有可见内容；正确 M 的 overlap 高于偏移 40px 对照 ----
    ov_ok = overlap_of(ref18, reg13, M13)
    Ms = M13.copy(); Ms[0, 2] += 40; Ms[1, 2] += 40
    ov_shift = overlap_of(ref18, reg13, Ms)
    rgba = map_to_overlay_rgba(str(lib.get(18, "一楼").path), M13, 1920, 1080)
    visible = (rgba[:, :, 3] > 0)
    panel_vis = visible[py:py + ph, px:px + pw].mean()
    check("渲染约定(直传M)+overlay落面板", ov_ok > ov_shift and panel_vis > 0.3,
          f"overlap={ov_ok:.2f} 偏移40px对照={ov_shift:.2f} 面板内overlay可见占比={panel_vis:.2f}")

    # ---- E) 低探明(随机抹掉70%探明像素，留真实结构) ----
    cls_r = classify_region(reg13)
    revealed = _revealed_mask(reg13, cls_r) > 0
    keep = RNG.random(reg13.shape[:2]) > 0.7      # 30% 位置保留探明
    reg_sparse = reg13.copy()
    reg_sparse[revealed & ~keep] = (20, 20, 20)   # 抹掉处当黑(未探明)
    Me, sce, ove = find_overlay_transform(None, ref18, FIXED_PANEL, region=reg_sparse)
    dte = abs(Me[0, 2] - M13[0, 2]) + abs(Me[1, 2] - M13[1, 2])
    dse = abs(1.0 / Me[0, 0] - 1.0 / M13[0, 0])
    check("30%探明子集恢复", dte < 3.0 and dse < 0.01, f"误差={dte:.2f}px ds={dse:.4f} "
          f"score={sce:.3f} overlap={ove:.2f}")

    n_ok = sum(results)
    print(f"\n跟随原语: {n_ok}/{len(results)} 通过")


if __name__ == "__main__":
    main()

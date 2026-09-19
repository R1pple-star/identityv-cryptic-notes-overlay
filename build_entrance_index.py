# -*- coding: utf-8 -*-
"""
入口引索构建工具
================
按用户思路：参考图里每个出入口都有白箭头图标(_icon_entrance.png)。
  一楼有2个入口(正门+侧门)，二楼有1个(二楼入口)。
  正门固定在最南边(y最大)，房间形状固定——主要判别依据。
  侧门/二楼位置随机、形状不同。

本工具自动检测参考图里的入口图标，按规则标注：
  一楼: 2个图标 → y最大=正门, 另一个=侧门
  二楼: 1个图标 → 二楼入口
裁出每个入口的区域(引索) + 画框标注图供人工核对。

用法: python build_entrance_index.py [种子号 ...]   (默认试几个种子)
输出: entrance_index/ 下 {seed}_正门.png / _侧门.png / _二楼.png / _一楼标注.png / _二楼标注.png
"""
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.map_library import MapLibrary
from core.vision import content_bbox, load_bgr

_ROOT = Path(__file__).resolve().parent
with open(_ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)
MAP_DIR = _CFG["paths"]["map_library"]
ICON_PATH = _ROOT / _CFG["paths"]["icon_template"]
OUT_DIR = _ROOT / _CFG["paths"]["entrance_index"]

# 框颜色 (BGR -> 转 PIL 用 RGB)
C_MAIN = (220, 60, 60)    # 正门 红
C_SIDE = (60, 200, 80)    # 侧门 绿
C_2F = (60, 120, 240)     # 二楼 蓝

# 图标检测：真实入口分 0.73~0.91，杂峰≤0.61，有清晰鸿沟。门槛 0.60 只留真实入口。
SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0)
MIN_SCORE = 0.60
PEAK_SUPPRESS = 30  # 峰值抑制半径(像素)
# 引索裁剪半边 = 图标检测尺寸 * CROP_K。6.0（含入口房间形状）已不够：图标锚定
# （core.entrance._ANCHOR_PIN）要求模板**完整落在锚点周围**，半幅需 ≥ 样本裁样的世界覆盖
# （≈0.36×面板半幅×kf ≈ 83~116 参考px），6.0 只有 ~114px 且常被参考图边缘裁短
# （实测 iwx 低到 166）→ 锚定档整张算不出种子（e5 A/B：标准窗 25 张里丢 6~8 张）。
# 12.0 = 2×。放大窗口本身会让全搜变差（e5：主集 top-1 3/3→2/3、控制组低分确信 1→5），
# 由锚定抵消（大窗+锚定回到 3/3 与 1）。
CROP_K = 12.0
# 细网格：0.20-1.00 步 0.02（r≈0.36 邻域要密），1.05-3.00 步 0.05（兜底到旧粗网格上界）。
# 旧粗网格 SCALES 下界 0.4 会把真实 r 钉在边界（实测 82 个入口全部 0.4 = 饱和），而锚定要
# s0 = r_fine/k，必须量准。
SCALES_FINE = tuple(round(0.20 + 0.02 * i, 2) for i in range(41)) + \
              tuple(round(1.05 + 0.05 * i, 2) for i in range(40))
FINE_HALF = 90     # 重测邻域半边（参考px）


def find_entrance_icons(ref_bgr, icon_gray,
                        scales=SCALES, min_score=MIN_SCORE,
                        suppress=PEAK_SUPPRESS, max_peaks=6):
    """多尺度多峰模板匹配，返回 [(score, cx, cy, scale), ...] 去重后。"""
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    iw, ih = icon_gray.shape[1], icon_gray.shape[0]
    cands = []
    for s in scales:
        siw, sih = max(8, int(iw * s)), max(8, int(ih * s))
        if siw > gray.shape[1] or sih > gray.shape[0]:
            continue
        ic = cv2.resize(icon_gray, (siw, sih), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, ic, cv2.TM_CCOEFF_NORMED).copy()
        for _ in range(max_peaks):
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < min_score:
                break
            cx, cy = ml[0] + siw // 2, ml[1] + sih // 2
            cands.append((float(mv), cx, cy, s, siw, sih))
            # 抑制邻域
            x0, x1 = max(0, ml[0] - suppress), ml[0] + siw + suppress
            y0, y1 = max(0, ml[1] - suppress), ml[1] + sih + suppress
            res[max(0, ml[1] - suppress):y1, max(0, ml[0] - suppress):x1] = -1
    # 全局去重：按分数降序，保留距离够远者
    cands.sort(reverse=True)
    kept = []
    for sc, cx, cy, s, siw, sih in cands:
        if all((cx - k[1]) ** 2 + (cy - k[2]) ** 2 > suppress ** 2 for k in kept):
            kept.append((sc, cx, cy, s, siw, sih))
    return kept


def label_floor1(icons):
    """一楼: icons 已按分数降序，取前2(真实入口)；其中 y最大(最南)=正门，另一个=侧门。

    注意不能取所有检测里最南两个——底部常有杂峰低分假阳性，会被误选。
    真实入口分0.73~0.91远高于杂峰≤0.61，故取分数前2。
    """
    top2 = icons[:2]
    if len(top2) < 2:
        return None
    by_y = sorted(top2, key=lambda c: c[2])  # cy 升序
    return {"正门": by_y[-1], "侧门": by_y[0]}


def measure_fine_scale(ref_bgr, icon_gray, cx, cy, half=FINE_HALF):
    """在已定位的 (cx,cy) 邻域用细网格重测图标尺度，返回 (r_fine, ncc)。

    位置已由 find_entrance_icons 定准（实测峰位偏移 ≤1px、NCC 0.78~0.92），故只重测尺度：
    细网格 + 下探到 0.20，避开粗网格下界饱和。r_fine 供图标锚定的 s0 = r_fine/k。
    """
    H, W = ref_bgr.shape[:2]
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    win = cv2.cvtColor(ref_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    iw, ih = icon_gray.shape[1], icon_gray.shape[0]
    best = (None, 0.0)
    for r in SCALES_FINE:
        siw, sih = max(8, int(round(iw * r))), max(8, int(round(ih * r)))
        if siw > win.shape[1] or sih > win.shape[0]:
            continue
        ic = cv2.resize(icon_gray, (siw, sih), interpolation=cv2.INTER_AREA)
        _, mv, _, _ = cv2.minMaxLoc(cv2.matchTemplate(win, ic, cv2.TM_CCOEFF_NORMED))
        if mv > best[1]:
            best = (r, float(mv))
    return best


def crop_around(ref_bgr, icon, half_k=CROP_K):
    """以图标为中心裁 K 倍图标尺寸的方框区域，返回 (crop_bgr, x0,y0,x1,y1)。

    **保持「居中 + 硬裁」**：贴参考图边缘的入口会裁成残片，但这正是锚定需要的——图标
    必须尽量落在裁图中心，两侧各留出 ≥ 模板世界覆盖(≈84~116px) 的余量。试过「整体平移
    塞进参考图」，结果图标被推到裁图边缘、锚定放不下 → 掉回大窗口全搜（=A/B 里会炸的
    E 档），e2e 2/4→0/4、主集 3/3→2/3，已撤销。贴边入口是已知死角，见计划文档。
    """
    sc, cx, cy, s, siw, sih = icon
    half = int(max(siw, sih) * half_k)
    H, W = ref_bgr.shape[:2]
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    return ref_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def draw_boxes(ref_bgr, boxes):
    """boxes: [(label, x0,y0,x1,y1, color_rgb), ...]。返回 PIL 图(BGR转RGB)。"""
    rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    for label, x0, y0, x1, y1, color in boxes:
        dr.rectangle([x0, y0, x1, y1], outline=color, width=4)
        # 标签底色块 + 文字(英文缩写避免中文字体问题)
        tag = {"正门": "MAIN", "侧门": "SIDE", "二楼": "2F"}[label]
        dr.rectangle([x0, max(0, y0 - 22), x0 + 60, y0], fill=color)
        dr.text((x0 + 4, max(0, y0 - 20)), tag, fill=(255, 255, 255))
    return img


def build_for_seed(lib, seed, out_dir, min_score=MIN_SCORE):
    floors = lib.floors_for(seed)
    info1 = lib.get(seed, "一楼")
    info2 = lib.get(seed, "二楼")
    icon = cv2.imread(str(ICON_PATH), cv2.IMREAD_GRAYSCALE)
    if icon is None:
        print(f"[缺图标模板] {ICON_PATH}"); return

    print(f"\n=== 种子 {seed} ===")
    index = {"seed": seed}
    for fl, info in [("一楼", info1), ("二楼", info2)]:
        if info is None:
            print(f"  {fl}: 无参考图"); continue
        ref = load_bgr(str(info.path))
        icons = find_entrance_icons(ref, icon, min_score=min_score)
        print(f"  {fl}({info.key}): 检测到 {len(icons)} 个图标: " +
              ", ".join(f"({c[1]},{c[2]})分{c[0]:.2f}s{c[3]}" for c in icons))
        if fl == "一楼":
            lab = label_floor1(icons)
            if lab is None:
                print(f"    ⚠ 一楼图标数<2，无法标注正门/侧门")
                continue
            boxes = []
            for name, ic in lab.items():
                crop, bb = crop_around(ref, ic)
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(
                    out_dir / f"{seed}_{name}.png")
                r_fine, ncc = measure_fine_scale(ref, icon, ic[1], ic[2])
                index[name] = {"floor": "一楼", "cx": int(ic[1]), "cy": int(ic[2]),
                               "scale": float(ic[3]),
                               "scale_fine": float(r_fine if r_fine else ic[3]),
                               "box": list(map(int, bb))}
                print(f"    {name} r_fine={index[name]['scale_fine']:.2f}(NCC{ncc:.2f}) "
                      f"box边长{bb[2]-bb[0]}x{bb[3]-bb[1]}")
                color = C_MAIN if name == "正门" else C_SIDE
                boxes.append((name, *bb, color))
            draw_boxes(ref, boxes).save(out_dir / f"{seed}_一楼标注.png")
            print(f"    正门@{lab['正门'][1]},{lab['正门'][2]} 侧门@{lab['侧门'][1]},{lab['侧门'][2]}")
        else:  # 二楼
            if not icons:
                print(f"    ⚠ 二楼未检测到图标")
                continue
            ic = icons[0]  # 唯一
            crop, bb = crop_around(ref, ic)
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(
                out_dir / f"{seed}_二楼.png")
            index["二楼"] = {"floor": "二楼", "cx": int(ic[1]), "cy": int(ic[2]),
                             "scale": float(ic[3]),
                             "scale_fine": float(measure_fine_scale(ref, icon, ic[1], ic[2])[0]
                                                 or ic[3]),
                             "box": list(map(int, bb))}
            draw_boxes(ref, [("二楼", *bb, C_2F)]).save(out_dir / f"{seed}_二楼标注.png")
    import json
    (out_dir / f"{seed}_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="构建入口引索（检测参考图入口图标→裁图+标注+JSON）")
    ap.add_argument("seeds", nargs="*", type=int, default=[18, 27, 9, 20], help="种子号(默认几个)")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE,
                    help=f"图标检测阈值(默认{MIN_SCORE}；一楼图标<2 时降低试之)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib = MapLibrary.load(MAP_DIR)
    for seed in args.seeds:
        if lib.get(seed, "一楼") is None:
            print(f"种子 {seed} 不在库，跳过"); continue
        build_for_seed(lib, seed, OUT_DIR, min_score=args.min_score)
    print(f"\n输出目录: {OUT_DIR}")
    print("核对：打开 *_一楼标注.png 看红框(正门,最南)绿框(侧门)；*_二楼标注.png 看蓝框(二楼)。")
    print("确认无误后，引索裁图 *_正门.png/*_侧门.png/*_二楼.png 即可用于新匹配模式。")


if __name__ == "__main__":
    main()

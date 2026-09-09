# -*- coding: utf-8 -*-
"""
单一评估 harness
================
在「刚进入口」标注集(eval/labels.csv)上跑入口引索匹配 + 两段式对齐，报：
  主集(is_entrance_view=1) top-1/top-3 准确率 + overlap 闸通过率
  控制组(is_entrance_view=0) 误报率(不该确信却确信了)
路径从 config 读。每改动对着它跑(见 CLAUDE.md)。

labels.csv 列：file,seed,floor,entrance_type,is_entrance_view,notes
  file=截图文件名(相对 config.paths.shot_library)；seed=真种子；entrance_type=侧门/二楼/正门；
  is_entrance_view=1=刚进入口(主集)，0=非刚进入口/判不出(控制组，防假阳)。
"""
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from core.alignment import find_overlay_transform  # noqa: E402
from core.entrance import build_entrance_transform, find_seed_by_entrance, load_index  # noqa: E402
from core.map_library import MapLibrary  # noqa: E402
from core.vision import FIXED_PANEL, load_bgr  # noqa: E402

with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)
SHOT_DIR = Path(_CFG["paths"]["shot_library"])
MAP_DIR = _CFG["paths"]["map_library"]
SCORE_CONFIDENT = float(_CFG["match"]["score_confident"])
ALIGN_SCORE_MAX = float(_CFG["match"]["align_score_max"])
OVERLAP_MIN = float(_CFG["match"]["overlap_min"])
LABELS = ROOT / "eval" / "labels.csv"


def _align_overlap(shot, best, entrance_type, icon_pos, lib):
    """两段式对齐 overlap（与 app._two_stage_align 一致：hint 精修→回退全搜）。"""
    info = lib.get(best[1], best[3])
    if info is None:
        return None
    ref = load_bgr(str(info.path))
    hint_s = None
    idx = load_index(best[1])
    if idx is not None:
        m1 = build_entrance_transform(best, entrance_type, idx, icon_pos, FIXED_PANEL)
        if m1 is not None:
            hint_s = m1[1]
    if hint_s is not None:
        a = find_overlay_transform(shot, ref, FIXED_PANEL, hint_s=hint_s)
        if a is not None and a[1] < ALIGN_SCORE_MAX and a[2] >= OVERLAP_MIN:
            return a
    return find_overlay_transform(shot, ref, FIXED_PANEL)


def load_labels():
    rows = []
    with open(LABELS, encoding="utf-8-sig") as f:
        for r in __import__("csv").DictReader(f):
            if r.get("file"):
                rows.append(r)
    return rows


def evaluate_sample(shot, lib, entrance_type, gt_seed):
    """返回 dict: top3, best_seed, best_score, overlap, gate_pass, top1_correct, top3_correct。"""
    res, ip, isc = find_seed_by_entrance(shot, lib, entrance_type, top_n=3)
    if not res:
        return dict(top3=[], best_seed=None, best_score=None, overlap=None,
                    gate_pass=False, top1_correct=False, top3_correct=False, icon=isc)
    best = res[0]
    top3 = [(r[1], r[0]) for r in res]
    align = _align_overlap(shot, best, entrance_type, ip, lib)
    ov = align[2] if align else None
    gate = (best[0] < SCORE_CONFIDENT and ov is not None and ov >= OVERLAP_MIN)
    return dict(top3=top3, best_seed=best[1], best_score=best[0], overlap=ov,
                gate_pass=gate, icon=isc,
                top1_correct=(best[1] == gt_seed),
                top3_correct=(gt_seed in [r[1] for r in res]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib = MapLibrary.load(MAP_DIR)
    rows = load_labels()
    print(f"标注集: {len(rows)} 张 (eval/labels.csv)\n")
    hdr = f"{'file':<40}{'view':<5}{'et':<5}{'GT':<4}{'best':<5}{'分':<7}{'overlap':<8}{'闸':<3}{'top1':<5}{'top3'}"
    print(hdr); print("-" * len(hdr))

    main_n = main_t1 = main_t3 = main_gate = 0
    ctrl_n = ctrl_fp = 0
    for r in rows:
        p = SHOT_DIR / r["file"]
        if not p.exists():
            print(f"{r['file']:<40}[跳过: 截图不存在]"); continue
        shot = load_bgr(str(p))
        gt = int(r["seed"])
        et = r["entrance_type"]
        is_view = r.get("is_entrance_view", "0").strip() == "1"
        e = evaluate_sample(shot, lib, et, gt)
        view = "主" if is_view else "控"
        gate = "✓" if e["gate_pass"] else " "
        t1 = "✓" if e["top1_correct"] else "✗"
        t3 = "✓" if e["top3_correct"] else "✗"
        ov_txt = f"{e['overlap']:.2f}" if e["overlap"] is not None else "-"
        sc_txt = f"{e['best_score']:.3f}" if e["best_score"] is not None else "-"
        print(f"{r['file'][:40]:<40}{view:<5}{et:<5}{gt:<4}{(e['best_seed'] or '-'):<5}"
              f"{sc_txt:<7}{ov_txt:<8}{gate:<3}{t1:<5}{t3}")
        if is_view:
            main_n += 1
            main_t1 += int(e["top1_correct"])
            main_t3 += int(e["top3_correct"])
            main_gate += int(e["gate_pass"])
        else:
            ctrl_n += 1
            ctrl_fp += int(e["gate_pass"])  # 控制组确信 = 误报

    print("-" * len(hdr))
    print(f"\n主集(刚进入口): top-1 {main_t1}/{main_n}  top-3 {main_t3}/{main_n}  "
          f"overlap闸通过 {main_gate}/{main_n}")
    fp_rate = (ctrl_fp / ctrl_n) if ctrl_n else 0
    print(f"控制组(非刚进入口): 误报率 {ctrl_fp}/{ctrl_n} = {fp_rate:.0%} "
          f"(确信=误报，应为0)")
    print(f"闸: 入口分<{SCORE_CONFIDENT} 且 overlap≥{OVERLAP_MIN}；对齐显示分<{ALIGN_SCORE_MAX}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
单一评估 harness
================
在「刚进入口」标注集(eval/labels.csv)上跑入口引索匹配 + 两段式对齐，报：
  [!] 分母只算「应答的那几张」—— 拒答被排除在外，头条数字偏乐观。
  用户真实体验看 experiments/e26_e2e_truth.py（"按下热键后屏幕上出现的东西"）。
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
from core.entrance import (  # noqa: E402
    _crop_around_icon, build_entrance_transform, sample_structure,
    find_seed_by_entrance, load_index,
)
from core.map_library import MapLibrary  # noqa: E402
from core.vision import load_bgr, panel_for_screen  # noqa: E402

with open(ROOT / "config.toml", "rb") as _f:
    _CFG = tomllib.load(_f)
SHOT_DIR = Path(_CFG["paths"]["shot_library"])
MAP_DIR = _CFG["paths"]["map_library"]
SCORE_CONFIDENT = float(_CFG["match"]["score_confident"])
ALIGN_SCORE_MAX = float(_CFG["match"]["align_score_max"])
OVERLAP_MIN = float(_CFG["match"]["overlap_min"])
SAMPLE_MASK_MIN = float(_CFG["match"]["sample_mask_min"])
LEAD_MIN = float(_CFG["match"].get("lead_min", 0.25))


def _lead(res):
    """入口 top1 相对 top2 的领先幅度（app._lead_frac 同款）；只有一个结果 ⇒ 1.0。"""
    if len(res) < 2 or res[1][0] <= 1e-9:
        return 1.0
    return (res[1][0] - res[0][0]) / res[1][0]
DOMINANT_CLS_MAX = float(_CFG["match"].get("dominant_cls_max", 0.90))
LABELS = ROOT / "eval" / "labels.csv"


def _degen(res):
    """与 app._degenerate 同款退化判定（多种子同分）。"""
    return len(res) >= 2 and res[0][0] < 0.001 and (res[1][0] - res[0][0]) < 0.001


def _align_overlap(shot, best, entrance_type, icon_pos, lib):
    """两段式对齐 overlap（与 app._two_stage_align 一致：hint 精修→回退全搜）。"""
    info = lib.get(best[1], best[3])
    if info is None:
        return None
    ref = load_bgr(str(info.path))
    panel = panel_for_screen(shot.shape[1], shot.shape[0])
    if panel is None:
        return None
    hint_s = None
    hint_icon = None
    idx = load_index(best[1])
    if idx is not None:
        entry = idx.get(entrance_type)
        if entry is not None:
            hint_icon = (entry["cx"], entry["cy"], icon_pos[0], icon_pos[1])
        m1 = build_entrance_transform(best, entrance_type, idx, icon_pos, panel)
        if m1 is not None:
            hint_s = m1[1]
    if hint_s is not None or hint_icon is not None:
        a = find_overlay_transform(shot, ref, panel, hint_s=hint_s, hint_icon=hint_icon)
        if a is not None and a[1] < ALIGN_SCORE_MAX and a[2] >= OVERLAP_MIN:
            return a
    return find_overlay_transform(shot, ref, panel)


def load_labels():
    rows = []
    with open(LABELS, encoding="utf-8-sig") as f:
        for r in __import__("csv").DictReader(f):
            if r.get("file"):
                rows.append(r)
    return rows


def resolve_shot(fname: str):
    """标注 file 依次在 shot_library / captures / eval/inbox 下找（captures 是运行时
    截图、inbox 是失败回收，标注集三处都能指）。找不到返回 None。"""
    for cand in (SHOT_DIR / fname, ROOT / "captures" / fname, ROOT / "eval" / "inbox" / fname):
        if cand.exists():
            return cand
    return None


def evaluate_sample(shot, lib, entrance_type, gt_seed):
    """返回 dict: top3, best_seed, best_score, overlap, gate_pass, top1_correct, top3_correct, degenerate, rejected。

    rejected=True = 触发快速失败闸（mask<SAMPLE_MASK_MIN 未探明）——app 同款行为是拒答
    并提示「周围已探明再按」，不产生任何匹配主张，故不计入 top-1/top-3 分母、单列统计。"""
    panel = panel_for_screen(shot.shape[1], shot.shape[0])
    if panel is None:
        return dict(top3=[], best_seed=None, best_score=None, overlap=None,
                    gate_pass=False, top1_correct=False, top3_correct=False, icon=0.0,
                    degenerate=False, rejected=True)
    res, ip, isc, ik = find_seed_by_entrance(shot, lib, entrance_type, panel=panel, top_n=3)
    if not ip or not res:
        return dict(top3=[], best_seed=None, best_score=None, overlap=None,
                    gate_pass=False, top1_correct=False, top3_correct=False, icon=isc,
                    degenerate=False, rejected=False)
    # 快速失败闸镜像（app._entrance_pipeline_impl 同款）：未探明样本 app 默认拒答，
    # **但 2026-09-18 起有逃生门** —— 入口 top1 领先次名 ≥ lead_min 时放行（结构占比只是
    # "能不能判"的代理，入口层判的是墙；依据 experiments/e34_evidence_gate.py）。
    cls, mask = sample_structure(_crop_around_icon(shot, panel, ip, 0.18, icon_k=ik))
    if mask.mean() < SAMPLE_MASK_MIN and _lead(res) < LEAD_MIN:
        return dict(top3=[], best_seed=None, best_score=None, overlap=None,
                    gate_pass=False, top1_correct=False, top3_correct=False, icon=isc,
                    degenerate=False, rejected=True)
    # 扩大取样梯子（与 app._entrance_pipeline_impl 保持同步）：单类占比>阈值且
    # 当前档退化 → 0.25/0.32 重裁重匹配，首个非退化即停；全退化维持首档结果
    # （下游闸自会拦，排名统计口径与不救时一致）。
    if ip is not None:
        tot = int(mask.sum())
        dom = (max(float(((cls == c) & (mask > 0)).sum()) / max(1, tot) for c in (1, 2, 3))
               if tot > 0 else 1.0)
        if dom > DOMINANT_CLS_MAX:
            for hf in (0.25, 0.32):
                if res and not _degen(res):
                    break
                crop = _crop_around_icon(shot, panel, ip, hf, icon_k=ik)
                _c2, m2 = sample_structure(crop)
                if m2.mean() < SAMPLE_MASK_MIN:
                    continue
                r2, _ip2, _isc2, _ik2 = find_seed_by_entrance(
                    shot, lib, entrance_type, panel=panel, top_n=3, sample_crop=crop)
                if r2:
                    res = r2
    best = res[0]
    top3 = [(r[1], r[0]) for r in res]
    # 退化 = 多种子同分（app._after_match 同款闸：sc<0.001 且与次名分差<0.001）
    degenerate = _degen(res)
    align = _align_overlap(shot, best, entrance_type, ip, lib)
    ov = align[2] if align else None
    gate = (best[0] < SCORE_CONFIDENT and ov is not None and ov >= OVERLAP_MIN)
    return dict(top3=top3, best_seed=best[1], best_score=best[0], overlap=ov,
                gate_pass=gate, icon=isc, degenerate=degenerate, rejected=False,
                top1_correct=(best[1] == gt_seed),
                top3_correct=(gt_seed in [r[1] for r in res]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib = MapLibrary.load(MAP_DIR)
    rows = load_labels()
    print(f"标注集: {len(rows)} 张 (eval/labels.csv)\n")
    hdr = (f"{'file':<40}{'view':<5}{'et':<5}{'GT':<4}{'best':<5}{'分':<7}"
           f"{'overlap':<8}{'闸':<3}{'退':<3}{'拒':<3}{'top1':<5}{'top3'}")
    print(hdr); print("-" * len(hdr))

    main_n = main_t1 = main_t3 = main_gate = main_deg = main_rej = 0
    ctrl_n = ctrl_fp = ctrl_rej = 0
    for r in rows:
        p = resolve_shot(r["file"])
        if p is None:
            print(f"{r['file'][:40]:<40}[跳过: 截图不存在]"); continue
        shot = load_bgr(str(p))
        gt = int(r["seed"])
        et = r["entrance_type"]
        is_view = r.get("is_entrance_view", "0").strip() == "1"
        e = evaluate_sample(shot, lib, et, gt)
        view = "主" if is_view else "控"
        gate = "✓" if e["gate_pass"] else " "
        t1 = "✓" if e["top1_correct"] else "✗"
        t3 = "✓" if e["top3_correct"] else "✗"
        deg = "!" if e["degenerate"] else " "
        rej = "拒" if e["rejected"] else " "
        ov_txt = f"{e['overlap']:.2f}" if e["overlap"] is not None else "-"
        sc_txt = f"{e['best_score']:.3f}" if e["best_score"] is not None else "-"
        print(f"{r['file'][:40]:<40}{view:<5}{et:<5}{gt:<4}{(e['best_seed'] or '-'):<5}"
              f"{sc_txt:<7}{ov_txt:<8}{gate:<3}{deg:<3}{rej:<3}{t1:<5}{t3}")
        if is_view:
            main_n += 1
            if e["rejected"]:
                main_rej += 1  # 拒答不计入 top-1/top-3 分母（app 同款：不产生匹配主张）
                continue
            main_t1 += int(e["top1_correct"])
            main_t3 += int(e["top3_correct"])
            main_gate += int(e["gate_pass"])
            main_deg += int(e["degenerate"])
        else:
            ctrl_n += 1
            if e["rejected"]:
                ctrl_rej += 1
                continue
            ctrl_fp += int(e["gate_pass"])  # 控制组确信 = 误报

    print("-" * len(hdr))
    answered = main_n - main_rej
    print(f"\n主集(刚进入口): 应答 top-1 {main_t1}/{answered}  top-3 {main_t3}/{answered}  "
          f"overlap闸通过 {main_gate}/{answered}  匹配退化 {main_deg}/{answered}")
    print(f"  拒答(结构闸挡下，真实通过率看 experiments/e26_e2e_truth.py) {main_rej}/{main_n}"
          " —— 不计入应答分母")
    ctrl_ans = ctrl_n - ctrl_rej
    fp_rate = (ctrl_fp / ctrl_ans) if ctrl_ans else 0
    print(f"控制组(非刚进入口): 误报率 {ctrl_fp}/{ctrl_ans} = {fp_rate:.0%} "
          f"(确信=误报，应为0；另拒答{ctrl_rej})")
    print(f"闸: 入口分<{SCORE_CONFIDENT} 且 overlap≥{OVERLAP_MIN}；对齐显示分<{ALIGN_SCORE_MAX}")


if __name__ == "__main__":
    main()

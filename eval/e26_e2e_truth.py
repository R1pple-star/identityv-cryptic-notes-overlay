# -*- coding: utf-8 -*-
"""按「按下热键后屏幕上最终出现的东西是否正确」统计真实通过率。

逐张复刻 app 的完整决策链（样本闸 → 入口匹配 → 两段式对齐 → 显示闸），并同时算
**反事实**：如果把样本结构闸放开，这张会判对还是判错（衡量结构闸的代价/收益）。

用法: python eval/e26_e2e_truth.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.alignment import find_overlay_transform
from core.entrance import (_crop_around_icon, find_seed_by_entrance, load_index,
                           sample_structure)
from core.map_library import MapLibrary
from core.vision import _find_icon, detect_fog_panel, load_bgr

import tomllib
_CFG = tomllib.load(open(ROOT / "config.toml", "rb"))
_cfg = _CFG["match"]
ALIGN_MAX = float(_cfg["align_score_max"])
OV_MIN = float(_cfg["overlap_min"])
SCORE_CONF = float(_cfg["score_confident"])
MASK_MIN = float(_cfg["sample_mask_min"])
LEAD_MIN = float(_cfg.get("lead_min", 0.25))
SHOT_DIRS = [ROOT / "captures", Path(_CFG["paths"]["shot_library"])]
lib = MapLibrary.load(_CFG["paths"]["map_library"])


def shot_path(f):
    for d in SHOT_DIRS:
        if (d / f).exists():
            return d / f
    return None


def align_shown(shot, panel, res, ip, et):
    """复刻 app 的两段式对齐 + 显示闸。返回 (shown, 分, ov, seed)。"""
    sc, seed, key, fl, _s, _ml = res[0]
    info, idx = lib.get(seed, fl), load_index(seed)
    e = idx.get(et) if idx else None
    if info is None or e is None:
        return False, None, None, seed
    ref = load_bgr(str(info.path))
    a = find_overlay_transform(shot, ref, panel, hint_s=_s,
                               hint_icon=(e["cx"], e["cy"], ip[0], ip[1]))
    if a is None or not (a[1] < ALIGN_MAX and a[2] >= OV_MIN):
        a = find_overlay_transform(shot, ref, panel)
    if a is None:
        return False, None, None, seed
    return (a[1] < ALIGN_MAX and a[2] >= OV_MIN), a[1], a[2], seed


cur = {}          # 现网：出图 = 过结构闸 且 过显示闸
free = {}         # 反事实：拆掉结构闸，只留显示闸
for row in csv.DictReader(open(ROOT / "eval" / "labels.csv", encoding="utf-8")):
    if row.get("is_entrance_view", "").strip() != "1":
        continue
    f = row["file"]
    p = shot_path(f)
    if p is None:
        continue
    gt, et = int(row["seed"]), row["entrance_type"]
    shot = load_bgr(str(p))
    panel = detect_fog_panel(shot)
    ip, isc, ik = _find_icon(shot, *panel)
    if ip is None:
        cur[f] = free[f] = ("图标未检出", 0, 0); continue
    struct = sample_structure(_crop_around_icon(shot, panel, ip, icon_k=ik))[1].mean()
    res, _p, _s, _k = find_seed_by_entrance(shot, lib, et, panel=panel, top_n=3)
    # 结构闸的逃生门（config [match].lead_min）：top1 领先次名够多就放行
    lead = (1.0 if len(res) < 2 or res[1][0] <= 1e-9
            else (res[1][0] - res[0][0]) / res[1][0])
    gated = (struct < MASK_MIN and lead < LEAD_MIN) or not res
    if not res:
        cur[f] = ("无结果", 0, 0); free[f] = ("无结果", 0, 0); continue
    shown, asc, ov, seed = align_shown(shot, panel, res, ip, et)
    right = seed == gt
    if gated:
        cur[f] = ("拒答", right, struct)
    else:
        cur[f] = ("出图·对" if (shown and right)
                  else "出图·错" if shown else "藏图", right, struct)
    free[f] = ("出图·对" if (shown and right)
               else "出图·错" if shown else "藏图", right, struct)

n = len(cur)
print(f"主集 {n} 张\n")
print(f"{'现网（结构闸 %.2f + 显示闸 %.2f）' % (MASK_MIN, ALIGN_MAX):<34}")
for label, d in (("现网", cur), ("放开结构闸", free)):
    good = sum(1 for v in d.values() if v[0] == "出图·对")
    bad = sum(1 for v in d.values() if v[0] == "出图·错")
    hid = sum(1 for v in d.values() if v[0] == "藏图")
    ref = sum(1 for v in d.values() if v[0] == "拒答")
    print(f"  {label:<12} 出图且对 {good:>2}/{n} ({good / n:.0%})  "
          f"出图但错 {bad:>2}  藏图 {hid:>2}  拒答 {ref:>2}")

print("\n拒答里「本来能判对」的（拆掉结构闸就会多出这几张对的）：")
for f, v in cur.items():
    if v[0] == "拒答" and v[1]:
        print(f"   {f[8:22]} 结构{v[2]:.0%}  → 入口 top1 就是真种子")
print("\n现网出图但错的：")
for f, v in cur.items():
    if v[0] == "出图·错":
        print(f"   {f[8:22]} 结构{v[2]:.0%}")
print("\n现网藏图的（种子对但显示闸没过）：")
for f, v in cur.items():
    if v[0] == "藏图":
        print(f"   {f[8:22]} 种子{'对' if v[1] else '错'} 结构{v[2]:.0%}")

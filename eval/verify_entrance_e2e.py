# -*- coding: utf-8 -*-
"""端到端验证：入口引索匹配 → 找种子 → 投影重合(find_overlay_transform)。
对确认样本跑全流程，看：① 入口匹配定对种子没 ② 自动重合 overlap 合不合理。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.map_library import MapLibrary
from core.vision import load_bgr, panel_for_screen, _find_icon
from core.entrance import find_seed_by_entrance, ENTRANCE_FLOOR, build_entrance_transform, load_index
from core.alignment import find_overlay_transform

MAP_DIR = r"D:\Pictures\20260818摸金地图"
SHOT_DIR = r"D:\Videos\NVIDIA\IdentityV"
GT = [("26", "17.13.06.98", 18), ("26", "17.11.35.17", 18),
      ("26", "01.11.13.38", 27), ("26", "16.42.33.89", 9)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib = MapLibrary.load(MAP_DIR)
    for dt, ts, seed in GT:
        p = f"{SHOT_DIR}\\IdentityV Screenshot 2026.08.{dt} - {ts}.png"
        shot = load_bgr(p)
        panel = panel_for_screen(shot.shape[1], shot.shape[0])
        print(f"\n=== {ts} 真种子{seed} ===")
        chosen = None
        for et in ("侧门", "二楼", "正门"):  # 侧门/二楼优先(判别力强)
            res, ip, isc, _ik = find_seed_by_entrance(shot, lib, et, panel=panel, top_n=3)
            if not res:
                continue
            best, second = res[0], (res[1] if len(res) > 1 else (1e9,))
            conf = best[0] < 0.10 and (second[0] - best[0]) >= 0.02
            hit = "✓" if best[1] == seed else "✗"
            print(f"  [{et}] {hit} {best[2]}[{best[3]}]={best[0]:.3f} 确信={conf}")
            if chosen is None and conf:
                chosen = (et, best)
        if chosen is None:
            # 退而取侧门最佳(即便不确信)
            res, _, _, _ = find_seed_by_entrance(shot, lib, "侧门", panel=panel, top_n=1)
            chosen = ("侧门", res[0]) if res else None
        if chosen is None:
            print("  无匹配，跳过重合"); continue
        et, best = chosen
        ms, mseed, mkey, mfl, _ms, _mloc = best
        info = lib.get(mseed, mfl)
        if info is None:
            print(f"  种子{mseed}/{mfl} 无参考图"); continue
        ref = load_bgr(str(info.path))
        # 生产同款两段式: 图标锚定优先(防幽灵相位), 不过闸回退全搜
        align = None
        icon_pos, _isc, _ik = _find_icon(shot, *panel)
        idx = load_index(mseed)
        if idx is not None and idx.get(et) is not None and icon_pos is not None:
            hint_icon = (idx[et]["cx"], idx[et]["cy"], icon_pos[0], icon_pos[1])
            m1 = build_entrance_transform(best, et, idx, icon_pos, panel)
            a = find_overlay_transform(shot, ref, panel, hint_s=m1[1] if m1 else None,
                                       hint_icon=hint_icon)
            if a is not None and a[1] < 0.30 and a[2] >= 0.40:
                align = a
        if align is None:
            align = find_overlay_transform(shot, ref, panel)
        if align is None:
            print(f"  重合: 探明不足，居中兜底"); continue
        M, sc, overlap = align
        ok = "✓" if mseed == seed else "✗"
        print(f"  => 选[{et}] 种子{mseed}({mkey}) {ok}真{seed} | 重合 score={sc:.3f} overlap={overlap:.2f}")


if __name__ == "__main__":
    main()

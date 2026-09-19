# -*- coding: utf-8 -*-
"""
地图库加载器
============
负责解析「摸金地图」目录下的完整地图图片，建立种子索引。

文件名格式（困难模式）:
    {种子号} {方向}-{门特征}{楼层}.png
    例: "1 右-左上右下门一楼.png" -> 种子=1, 方向=右, 门特征=左上右下门, 楼层=一楼

方向: 北 / 南 / 左 / 右
楼层: 一楼 / 二楼 / 地下室 (噩梦模式会有地下室)
门特征: 自由文本（红门、T门、锤子门、青蛙房...），用于从入口肉眼识别种子

方向 + 门特征 的组合可唯一确定一个种子。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 方向（北/南/左/右）与楼层（一楼/二楼/地下室）都是单字符/固定词
DIRECTIONS = ("北", "南", "左", "右")
FLOORS = ("地下室", "一楼", "二楼")  # 长词在前，避免"一楼"匹配到"地下室"的一部分

# 匹配: "{种子号} {方向}[-一]?{门特征}{楼层}.png"
_FILENAME_RE = re.compile(r"^(\d+)\s+([北南左右])[-一]?(.+?)(地下室|一楼|二楼)\.png$")


@dataclass
class MapInfo:
    """单张地图的元信息。"""
    seed: int
    direction: str          # 北/南/左/右
    door: str               # 门特征文本
    floor: str              # 一楼/二楼/地下室
    path: Path

    @property
    def key(self) -> str:
        """方向+门特征 的唯一标识（肉眼识别种子用的线索）。"""
        return f"{self.direction}-{self.door}"


# 项目根（core/ 上一层）：相对路径的 map_library 按它解析，与运行时 cwd 无关
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MapLibrary:
    """按种子和楼层索引的地图库。"""
    base_dir: Path
    maps: dict[int, dict[str, MapInfo]] = field(default_factory=dict)  # seed -> {floor: MapInfo}
    # 方向+门特征 -> 种子号（线索反查表）
    clue_to_seed: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, base_dir: str | os.PathLike) -> "MapLibrary":
        base = Path(base_dir)
        if not base.is_absolute():
            base = (ROOT / base).resolve()
        lib = cls(base_dir=base)
        lib.scan()
        return lib

    def scan(self) -> None:
        """扫描目录下所有 png，解析文件名并建索引。"""
        self.maps.clear()
        self.clue_to_seed.clear()
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"地图目录不存在: {self.base_dir}")

        skipped: list[str] = []
        for p in sorted(self.base_dir.iterdir()):
            if p.suffix.lower() != ".png":
                continue
            info = parse_filename(p.name)
            if info is None:
                skipped.append(p.name)
                continue
            info.path = p
            self.maps.setdefault(info.seed, {})[info.floor] = info
            # 每个种子的"方向+门特征"线索应唯一；若重复则记录并覆盖
            if info.key in self.clue_to_seed:
                # 一个线索理论上只对应一个种子，重复说明文件名有误，留最后一张为准
                pass
            self.clue_to_seed[info.key] = info.seed

        self.skipped = skipped

    # ---- 查询接口 ----
    def seeds(self) -> list[int]:
        return sorted(self.maps.keys())

    def floors_for(self, seed: int) -> list[str]:
        return sorted(self.maps.get(seed, {}).keys())

    def get(self, seed: int, floor: str) -> Optional[MapInfo]:
        return self.maps.get(seed, {}).get(floor)

    def find_by_clue(self, direction: str, door: str) -> Optional[int]:
        """根据 方向+门特征 反查种子号。"""
        return self.clue_to_seed.get(f"{direction}-{door}")

    def directions(self) -> list[str]:
        """所有方向（北/南/左/右）。"""
        return sorted({i.direction for floors in self.maps.values() for i in floors.values()})

    def doors_for_direction(self, direction: str) -> list[str]:
        """某方向下所有门特征名（用于过滤门下拉）。"""
        return sorted({i.door for floors in self.maps.values()
                       for i in floors.values() if i.direction == direction})


def parse_filename(name: str) -> Optional[MapInfo]:
    """解析单个文件名。无法解析返回 None。"""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    seed = int(m.group(1))
    direction = m.group(2)
    door = m.group(3).strip()
    floor = m.group(4)
    return MapInfo(seed=seed, direction=direction, door=door, floor=floor,
                   path=Path(name))


"""
Universal random baseline generator for platformer-style VGLC levels.

- Reads a JSON config (Spelunky / Rainbow Islands / your own) specifying:
  * grid width/height, seed
  * tiles: names -> characters (e.g., {"empty": ".", "platform": "B", "ground": "G"})
  * generation params (platform_density, segments per row, etc.)
  * physics.solids: list[str] of solid tile characters
  * physics.jumps: list[list[list[int]]] jump arcs (dx, dy per frame)
- Can ensure solvability using your existing A* solver test_level.findPaths
- Saves out VGLC-style .txt

CLI examples:
  python3 universal_level_generator.py --config spelunky_cfg.json --out SpelunkyRand01.txt --ensure-solvable
  python3 universal_level_generator.py --config rainbow_cfg.json  --out RI_Rand01.txt

Config JSON (example for Spelunky-like tiles/physics):
{
  "width": 40,
  "height": 32,
  "seed": 123,
  "tiles": {"empty": ".", "platform": "X", "ground": "#"},
  "generation": {
    "platform_density": 0.12,
    "max_segments_per_row": 3,
    "segment_len_min": 2,
    "segment_len_max": 6,
    "ground_rows": 2,
    "left_right_border": true
  },
  "physics": {
    "solids": ["X", "#"],
    "jumps": [[ [0,-1],[0,-2],[1,-2],[1,-1] ], [ [0,-1],[0,-2],[1,-3],[2,-3],[2,-2] ]]
  }
}
"""

from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Your A* pathfinder (already in your env/project)
from test_level import findPaths  # type: ignore

# ---------------------------
# Data models
# ---------------------------
@dataclass
class GenerationParams:
    platform_density: float = 0.10
    max_segments_per_row: int = 2
    segment_len_min: int = 2
    segment_len_max: int = 6
    ground_rows: int = 1
    left_right_border: bool = False  # if True, keep first/last column solid ground

@dataclass
class PhysicsParams:
    solids: List[str]
    jumps: List[List[List[int]]]

@dataclass
class Tiles:
    empty: str = "."
    platform: str = "B"
    ground: str = "G"
    start: str = "D"   # optional marker for game engines (solver doesn't require)
    goal: str = "E"    # optional marker for game engines (solver doesn't require)

@dataclass
class GenConfig:
    width: int
    height: int
    seed: Optional[int]
    tiles: Tiles
    generation: GenerationParams
    physics: PhysicsParams

# ---------------------------
# Config loading
# ---------------------------

def _dict_get(d: Dict, key: str, default):
    v = d.get(key, default)
    return v if v is not None else default


def load_config(path: Path) -> GenConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    width = int(_dict_get(cfg, "width", 33))
    height = int(_dict_get(cfg, "height", 165))
    seed = cfg.get("seed")

    tiles_d = _dict_get(cfg, "tiles", {})
    tiles = Tiles(
        empty=str(_dict_get(tiles_d, "empty", ".")),
        platform=str(_dict_get(tiles_d, "platform", "B")),
        ground=str(_dict_get(tiles_d, "ground", "G")),
        start=str(_dict_get(tiles_d, "start", "D")),
        goal=str(_dict_get(tiles_d, "goal", "E")),
    )

    gen_d = _dict_get(cfg, "generation", {})
    generation = GenerationParams(
        platform_density=float(_dict_get(gen_d, "platform_density", 0.10)),
        max_segments_per_row=int(_dict_get(gen_d, "max_segments_per_row", 2)),
        segment_len_min=int(_dict_get(gen_d, "segment_len_min", 2)),
        segment_len_max=int(_dict_get(gen_d, "segment_len_max", 6)),
        ground_rows=int(_dict_get(gen_d, "ground_rows", 1)),
        left_right_border=bool(_dict_get(gen_d, "left_right_border", False)),
    )

    phys_d = _dict_get(cfg, "physics", {})
    solids = list(_dict_get(phys_d, "solids", [tiles.platform, tiles.ground]))
    jumps = list(_dict_get(phys_d, "jumps", []))  # expect [[[dx,dy],...], ...]
    if not jumps:
        # Reasonable default arcs (Rainbow Islands style)
        jumps = [
            [[0, -1], [0, -2], [1, -2], [1, -1]],
            [[0, -1], [0, -2], [1, -3], [1, -3], [2, -2], [2, -1]],
            [[1, -1], [1, -2], [2, -3], [3, -3], [4, -2]],
            [[1, -1], [1, -2], [2, -2], [3, -3], [4, -3], [5, -2]],
        ]
    physics = PhysicsParams(solids=solids, jumps=jumps)

    return GenConfig(
        width=width,
        height=height,
        seed=seed,
        tiles=tiles,
        generation=generation,
        physics=physics,
    )

# ---------------------------
# Generation core
# ---------------------------

def generate_random_grid(cfg: GenConfig) -> np.ndarray:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    W, H = cfg.width, cfg.height
    t = cfg.tiles
    g = cfg.generation

    grid = np.full((H, W), t.empty, dtype='<U1')

    # Optional solid borders at x=0 and x=W-1
    if g.left_right_border:
        grid[:, 0] = t.ground
        grid[:, W - 1] = t.ground

    # Ground rows
    for y in range(H - g.ground_rows, H):
        grid[y, :] = t.ground

    # Random platform segments
    usable_y_max = H - g.ground_rows - 1
    for y in range(usable_y_max):  # exclude ground rows
        if random.random() < g.platform_density:
            nseg = random.randint(1, g.max_segments_per_row)
            for _ in range(nseg):
                seg_len = random.randint(g.segment_len_min, g.segment_len_max)
                # keep 1-tile padding from borders if borders are solid
                x_min = 1 if g.left_right_border else 0
                x_max = W - seg_len - (1 if g.left_right_border else 0)
                if x_max <= x_min:
                    continue
                start_x = random.randint(x_min, x_max)
                grid[y, start_x:start_x + seg_len] = t.platform

    # Place start/goal markers for engine parity (solver ignores, but helpful for exports)
    _place_start_and_goal(grid, cfg)
    return grid


# ---------------------------
# Start/Goal placement
# ---------------------------

def _place_start_and_goal(grid: np.ndarray, cfg: GenConfig) -> None:
    """Place a start (e.g., 'D') near the upper-left and a goal (e.g., 'E') near
    the lower-right. We try to put them on walkable positions above a solid tile.
    This is only for engine/export semantics; the solver itself starts near (2,2)
    and succeeds on reaching the rightmost column.
    """
    t = cfg.tiles
    H, W = grid.shape

    # Helper: find a cell with empty on (y,x) and solid just below (y+1,x)
    def find_anchor(y_range, x_range):
        for y in y_range:
            for x in x_range:
                y_below = min(y+1, H-1)
                if grid[y, x] == t.empty and grid[y_below, x] in cfg.physics.solids:
                    return y, x
        return None

    # Start: search top 20% rows, left 25% columns
    y_top = range(1, max(2, H//5))
    x_left = range(1, max(2, W//4))
    anchor_s = find_anchor(y_top, x_left) or (1, 1)
    sy, sx = anchor_s
    grid[sy, sx] = t.start

    # Goal: search bottom 30% rows, first 50% columns, bias near middle-left
    y_bot = range(max(1, H - max(3, H//3)), H - cfg.generation.ground_rows - 1)
    x_left_half = list(range(1, max(2, W//2)))
    # put a platform under the goal if needed
    anchor_g = find_anchor(y_bot, reversed(x_left_half)) or (max(1, H - cfg.generation.ground_rows - 2), max(1, W//4))
    gy, gx = anchor_g
    # ensure footing if not solid beneath
    if grid[min(gy+1, H-1), gx] not in cfg.physics.solids:
        grid[min(gy+1, H-1), gx] = t.platform
    grid[gy, gx] = t.goal

# ---------------------------
# I/O helpers
# ---------------------------

def save_txt(grid: np.ndarray, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write("".join(row.tolist()) + "\n")


def grid_to_lines(grid: np.ndarray) -> List[str]:
    return ["".join(row.tolist()) for row in grid]


# ---------------------------
# Solvability wrapper
# ---------------------------

def is_solvable(grid: np.ndarray, physics: PhysicsParams, sub_optimal: int = 0) -> Tuple[bool, Dict]:
    level_str = grid_to_lines(grid)
    paths = findPaths(sub_optimal, set(physics.solids), physics.jumps, level_str)
    if not paths:
        return False, {"num_paths": 0}
    return True, {"num_paths": len(paths), "best_length": len(paths[0]), "paths": paths}


def generate_until_solvable(cfg: GenConfig, max_attempts: int = 200, sub_optimal: int = 0) -> Tuple[np.ndarray, Dict]:
    tries = 0
    seed = cfg.seed
    while tries < max_attempts:
        grid = generate_random_grid(cfg)
        ok, info = is_solvable(grid, cfg.physics, sub_optimal=sub_optimal)
        if ok:
            return grid, info
        # advance seed to change RNG and try again
        tries += 1
        if seed is not None:
            seed = seed + 1
            cfg.seed = seed
    raise RuntimeError("Failed to generate a solvable level within max_attempts")


# ---------------------------
# CLI
# ---------------------------

def _main():
    ap = argparse.ArgumentParser(description="Universal random platformer generator (config-driven)")
    ap.add_argument("--config", type=Path, required=True, help="Path to JSON config containing tiles, physics, and generation params")
    ap.add_argument("--out", type=Path, required=True, help="Path to save generated .txt level")
    ap.add_argument("--ensure-solvable", action="store_true", help="Search until a solvable layout is produced")
    ap.add_argument("--max-attempts", type=int, default=200, help="Attempts when ensuring solvability")
    ap.add_argument("--sub", type=int, default=0, help="Sub-optimality band for path search")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.ensure_solvable:
        grid, info = generate_until_solvable(cfg, max_attempts=args.max_attempts, sub_optimal=args.sub)
        print(f"Generated SOLVABLE: best_length={info['best_length']} paths={info['num_paths']}")
    else:
        grid = generate_random_grid(cfg)

    save_txt(grid, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    _main()

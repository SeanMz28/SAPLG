"""
Random baseline generator for Rainbow Islands style levels.
- Produces a numpy char grid with '.', 'B', 'G'
- Supports optional solvability check using test_level.findPaths
- Can save out to VGLC-style .txt for your parser/preview scripts
"""

import numpy as np
import random
from typing import List, Tuple
from test_level import findPaths   # A* solver

class GenConfig:
    def __init__(
        self,
        width=33,
        height=165,
        seed=None,
        platform_density=0.10,
        max_segments_per_row=2,
        segment_len_min=2,
        segment_len_max=6,
        ground_rows=1
    ):
        self.width = width
        self.height = height
        self.seed = seed
        self.platform_density = platform_density
        self.max_segments_per_row = max_segments_per_row
        self.segment_len_min = segment_len_min
        self.segment_len_max = segment_len_max
        self.ground_rows = ground_rows

# --- GENERATOR ---
def generate_random_level(cfg: GenConfig) -> np.ndarray:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    grid = np.full((cfg.height, cfg.width), '.', dtype='<U1')

    # bottom rows as ground
    for y in range(cfg.height - cfg.ground_rows, cfg.height):
        grid[y, :] = 'G'

    # randomly scatter platform segments
    for y in range(cfg.height - cfg.ground_rows - 1):  # exclude ground
        if random.random() < cfg.platform_density:
            nseg = random.randint(1, cfg.max_segments_per_row)
            for _ in range(nseg):
                seg_len = random.randint(cfg.segment_len_min, cfg.segment_len_max)
                start_x = random.randint(1, cfg.width - seg_len - 1)
                grid[y, start_x:start_x+seg_len] = 'B'
    return grid

def save_txt(grid: np.ndarray, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for row in grid:
            f.write("".join(row.tolist()) + "\n")

def grid_to_lines(grid: np.ndarray) -> List[str]:
    return ["".join(row.tolist()) for row in grid]

# --- SOLVABILITY CHECK ---
DEFAULT_SOLIDS = ["B", "G"]
DEFAULT_JUMPS = [
    [[0, -1],[0, -2],[1, -2],[1, -1]],              # short hop
    [[0, -1],[0, -2],[1, -3],[2, -3],[2, -2]],      # medium
    [[1, -1],[1, -2],[2, -3],[3, -3],[4, -2]],      # running arc
    [[1, -1],[1, -2],[2, -2],[3, -3],[4, -3],[5, -2]]  # long arc
]

def is_solvable(grid: np.ndarray, sub_optimal=0) -> Tuple[bool, dict]:
    level_str = grid_to_lines(grid)
    paths = findPaths(sub_optimal, set(DEFAULT_SOLIDS), DEFAULT_JUMPS, level_str)
    if not paths:
        return False, {"num_paths": 0}
    return True, {"num_paths": len(paths), "best_length": len(paths[0])}

def generate_until_solvable(cfg: GenConfig, max_attempts=100, sub_optimal=0):
    for attempt in range(max_attempts):
        grid = generate_random_level(cfg)
        ok, info = is_solvable(grid, sub_optimal=sub_optimal)
        if ok:
            return grid, info
        if cfg.seed is not None:
            cfg.seed += 1
    raise RuntimeError("Could not generate solvable level after max_attempts")

# --- CLI ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Path to save generated .txt level")
    ap.add_argument("--width", type=int, default=33)
    ap.add_argument("--height", type=int, default=165)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--platform-density", type=float, default=0.10)
    ap.add_argument("--ensure-solvable", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=100)
    args = ap.parse_args()

    cfg = GenConfig(width=args.width, height=args.height, seed=args.seed,
                    platform_density=args.platform_density)

    if args.ensure_solvable:
        grid, info = generate_until_solvable(cfg, max_attempts=args.max_attempts)
        print(f"Generated solvable level: best_length={info['best_length']}, paths={info['num_paths']}")
    else:
        grid = generate_random_level(cfg)

    save_txt(grid, args.out)
    print(f"Saved -> {args.out}")


#use this to run the script:
# python3 random_baseline.py \
#   --out "RandomSolvable01.txt" \
#   --seed 42 \
#   --ensure-solvable \
#   --max-attempts 200

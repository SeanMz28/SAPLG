# constructive_baseline.py
"""
Constructive baseline generator for Rainbow Islands-style levels.
- Deterministic scaffold (no RNG by default): side walls, ground, ascending staircase
- Tiles: '.' (air), 'B' (platform), 'G' (ground), plus 'A' (start), 'T' (goal) [both passable]
- TODO: verify with pathfinder (test_level.findPaths)

CLI examples:
  python3 constructive_baseline.py --out "Constructive01.txt"
  python3 constructive_baseline.py --out "ConstructiveWide.txt" --width 33 --height 165
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

# TODO: wire to pathfinder; safe if it's missing
try:
    from test_level import findPaths
except Exception:
    findPaths = None

@dataclass
class Cfg:
    width: int = 33
    height: int = 165
    ground_rows: int = 1
    wall_thickness: int = 1
    step_vertical: int = 4     # rows between staircase steps
    step_length: int = 5       # horizontal length of each platform step
    step_overlap: int = 2      # horizontal overlap between successive steps
    margin_left: int = 2       # clear area near left for start
    margin_right: int = 2      # clear area near right for goal
    place_start_goal: bool = True
    check_with_astar: bool = False   # quick sanity check if pathfinder is available
    sub_optimal: int = 0

SOLIDS = {"B", "G"}          # for the pathfinder
JUMPS: List[List[List[int]]] = [
    [[0,-1],[0,-2],[1,-2],[1,-1]],
    [[0,-1],[0,-2],[1,-3],[2,-3],[2,-2]],
    [[1,-1],[1,-2],[2,-3],[3,-3],[4,-2]],
    [[1,-1],[1,-2],[2,-2],[3,-3],[4,-3],[5,-2]],
]

def make_empty(w: int, h: int) -> np.ndarray:
    return np.full((h, w), ".", dtype="<U1")

def add_enclosure(grid: np.ndarray, cfg: Cfg):
    h, w = grid.shape
    # side walls
    grid[:, :cfg.wall_thickness] = "B"
    grid[:, w - cfg.wall_thickness:w] = "B"
    # ground
    grid[h - cfg.ground_rows:h, :] = "G"
    # keep a doorway at bottom of walls to avoid full block:
    grid[h - cfg.ground_rows:h, 0:cfg.wall_thickness] = "G"
    grid[h - cfg.ground_rows:h, w - cfg.wall_thickness:w] = "G"

def add_staircase(grid: np.ndarray, cfg: Cfg):
    """Build an ascending rightward staircase above ground that always connects."""
    h, w = grid.shape
    y = h - cfg.ground_rows - 2                  # start just above ground
    x = cfg.wall_thickness + cfg.margin_left     # start after left wall/margin
    step = 0
    while y > 1 and x + cfg.step_length < w - cfg.wall_thickness - cfg.margin_right:
        # lay a horizontal platform step
        grid[y, x:x + cfg.step_length] = "B"
        # next step rises up and goes right with overlap so jumps are small
        x = x + cfg.step_length - cfg.step_overlap
        y = y - cfg.step_vertical
        step += 1

def place_start_goal(grid: np.ndarray, cfg: Cfg):
    """Mark start 'A' on first accessible platform/ground near left; goal 'T' near right."""
    h, w = grid.shape
    # find first non-air cell near left that has air above -> place 'A' on that air above
    for x in range(cfg.wall_thickness + cfg.margin_left, w // 2):
        for y in range(h - cfg.ground_rows - 2, 1, -1):
            if grid[y, x] in ("B", "G") and grid[y-1, x] == ".":
                grid[y-1, x] = "A"   # start sits in air tile above solid
                break
        else:
            continue
        break

    # find goal column near right
    for x in range(w - cfg.wall_thickness - cfg.margin_right - 1, w // 2, -1):
        for y in range(2, h - cfg.ground_rows - 1):
            if grid[y, x] == "." and (grid[y+1, x] in ("B", "G")):
                grid[y, x] = "T"
                return

def to_lines(grid: np.ndarray) -> List[str]:
    return ["".join(row.tolist()) for row in grid]

def astar_ok(grid: np.ndarray, cfg: Cfg) -> Tuple[bool, dict]:
    if findPaths is None:
        return True, {"note": "pathfinder not available, skipped"}
    level_str = to_lines(grid)
    paths = findPaths(cfg.sub_optimal, SOLIDS, JUMPS, level_str)
    if not paths:
        return False, {"num_paths": 0}
    return True, {"num_paths": len(paths), "best_length": len(paths[0])}

def generate_constructive(cfg: Cfg) -> Tuple[np.ndarray, dict]:
    g = make_empty(cfg.width, cfg.height)
    add_enclosure(g, cfg)
    add_staircase(g, cfg)
    if cfg.place_start_goal:
        place_start_goal(g, cfg)

    info = {"enclosure": True, "staircase": True}
    if cfg.check_with_astar:
        ok, extra = astar_ok(g, cfg)
        info.update(extra)
        info["solvable_check"] = ok
    return g, info

def save_txt(grid: np.ndarray, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write("".join(row.tolist()) + "\n")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Constructive baseline (staircase) generator")
    ap.add_argument("--out", required=True, help="Output .txt path")
    ap.add_argument("--width", type=int, default=33)
    ap.add_argument("--height", type=int, default=165)
    ap.add_argument("--ground-rows", type=int, default=1)
    ap.add_argument("--wall-thickness", type=int, default=1)
    ap.add_argument("--step-vertical", type=int, default=4)
    ap.add_argument("--step-length", type=int, default=5)
    ap.add_argument("--step-overlap", type=int, default=2)
    ap.add_argument("--no-start-goal", action="store_true", help="Do not place A/T markers")
    ap.add_argument("--check", action="store_true", help="Check solvability with A* if available")
    ap.add_argument("--sub", type=int, default=0, help="suboptimality band for A*")
    args = ap.parse_args()

    cfg = Cfg(
        width=args.width,
        height=args.height,
        ground_rows=args.ground_rows,
        wall_thickness=args.wall_thickness,
        step_vertical=args.step_vertical,
        step_length=args.step_length,
        step_overlap=args.step_overlap,
        place_start_goal=not args.no_start_goal,
        check_with_astar=args.check,
        sub_optimal=args.sub
    )

    grid, info = generate_constructive(cfg)
    save_txt(grid, args.out)
    print(f"Saved -> {args.out}")
    if cfg.check_with_astar:
        print("Solvable check:", info.get("solvable_check", None), "|", {k:v for k,v in info.items() if k!='paths'})

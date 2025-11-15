from __future__ import annotations
import argparse, json, os
from typing import List, Dict, Tuple, Optional
import numpy as np

# Optional: A* for solvability check (only used when --check is passed)
try:
    from src.core.solvability import is_level_solvable
except ImportError:
    try:
        # When run as a script from within src/generators/
        from ..core.solvability import is_level_solvable
    except (ImportError, ValueError):
        is_level_solvable = None
except Exception:
    is_level_solvable = None

# --------------------------- Config / IO ---------------------------

def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_lines(grid: np.ndarray) -> List[str]:
    return ["".join(row.tolist()) for row in grid]

# --------------------------- Grid building ---------------------------

def make_empty_grid(w: int, h: int, ch_empty: str) -> np.ndarray:
    return np.full((h, w), ch_empty, dtype="<U1")

def add_enclosure(grid: np.ndarray, tiles: Dict[str, str],
                  wall_thickness: int = 1, ground_rows: int = 1):
    """Side walls = platform tile; bottom = ground tile."""
    h, w = grid.shape
    plat = tiles["platform"]
    grd  = tiles["ground"]
    if wall_thickness > 0:
        grid[:, :wall_thickness] = plat
        grid[:, w - wall_thickness:] = plat
    if ground_rows > 0:
        grid[h - ground_rows:, :] = grd

def add_staircase(grid: np.ndarray, tiles: Dict[str, str],
                  start_x: int, start_y: int,
                  step_length: int = 4, step_vertical: int = 3,
                  step_overlap: int = 1, right_margin: int = 2, wall_thickness: int = 1):
    """Simple ascending rightward staircase of platforms."""
    h, w = grid.shape
    x = start_x
    y = start_y
    plat = tiles["platform"]
    while y > 1 and x + step_length < w - wall_thickness - right_margin:
        grid[y, x:x+step_length] = plat
        x = x + step_length - step_overlap
        y = y - step_vertical

def place_start_goal(grid: np.ndarray, tiles: Dict[str, str],
                     left_margin: int = 2, right_margin: int = 2,
                     wall_thickness: int = 1, ground_rows: int = 1) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Place Start (tiles['start']) above a solid ledge near the left,
    and Goal (tiles['goal']) above a solid ledge near the right.
    """
    h, w = grid.shape
    empty = tiles["empty"]
    solids_for_ledge = {tiles["ground"], tiles["platform"]}

    # Start: scan left → mid
    start_xy = None
    for x in range(wall_thickness + left_margin, w // 2):
        for y in range(h - ground_rows - 2, 1, -1):
            if grid[y, x] in solids_for_ledge and grid[y - 1, x] == empty:
                grid[y - 1, x] = tiles["start"]
                start_xy = (x, y - 1)
                break
        if start_xy:
            break
    if start_xy is None:
        start_xy = (wall_thickness + left_margin, h - ground_rows - 3)
        grid[start_xy[1], start_xy[0]] = tiles["start"]

    # Goal: scan right → mid
    goal_xy = None
    for x in range(w - wall_thickness - right_margin - 1, w // 2, -1):
        for y in range(2, h - ground_rows - 1):
            if grid[y + 1, x] in solids_for_ledge and grid[y, x] == empty:
                grid[y, x] = tiles["goal"]
                goal_xy = (x, y)
                break
        if goal_xy:
            break
    if goal_xy is None:
        goal_xy = (w - wall_thickness - right_margin - 2, 2)
        grid[goal_xy[1], goal_xy[0]] = tiles["goal"]

    return start_xy, goal_xy

# --------------------------- Solvability check ---------------------------

def astar_ok(grid: np.ndarray, cfg: Dict, start_xy: Tuple[int,int], goal_xy: Tuple[int,int], sub_optimal: int = 0):
    """
    Run A* solvability check using the Spelunky-specific solvability checker from solvability.py.
    Uses the config's tiles, physics, and jump definitions with ladder support.
    
    Uses a timeout to prevent hanging on difficult levels.
    """
    if is_level_solvable is None:
        return True, {"note": "solvability checker not available, skipped"}

    # Convert grid to level_rows format expected by solvability checker
    level_rows = to_lines(grid)
    
    # Use the is_level_solvable function from solvability.py with timeout
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Solvability check timed out")
    
    # Set a timeout of 10 seconds for the solvability check
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        solvable, info = is_level_solvable(level_rows, cfg, sub_optimal=sub_optimal, return_paths=False)
        signal.alarm(0)  # Cancel the alarm
        return solvable, info
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        return False, {"note": "solvability check timed out after 10 seconds", "timeout": True}
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        return False, {"error": str(e)}

# --------------------------- Generator orchestration ---------------------------

def generate(out_path: str, cfg_path: str,
             width: Optional[int] = None, height: Optional[int] = None,
             ground_rows: int = 1, wall_thickness: int = 1,
             step_length: int = 4, step_vertical: int = 3, step_overlap: int = 1,
             left_margin: int = 2, right_margin: int = 2):
    cfg = load_cfg(cfg_path)
    tiles = cfg["tiles"]  # expects: empty, platform, ground, start, goal

    W = width  if width  is not None else cfg.get("width", 33)
    H = height if height is not None else cfg.get("height", 165)

    grid = make_empty_grid(W, H, tiles["empty"])
    add_enclosure(grid, tiles, wall_thickness=wall_thickness, ground_rows=ground_rows)

    stair_y = H - ground_rows - 2
    stair_x = wall_thickness + left_margin
    add_staircase(grid, tiles, stair_x, stair_y,
                  step_length=step_length, step_vertical=step_vertical,
                  step_overlap=step_overlap, right_margin=right_margin,
                  wall_thickness=wall_thickness)

    start_xy, goal_xy = place_start_goal(grid, tiles,
                                         left_margin=left_margin, right_margin=right_margin,
                                         wall_thickness=wall_thickness, ground_rows=ground_rows)

    # write file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in to_lines(grid):
            f.write(row + "\n")

    info = {"width": W, "height": H, "start": start_xy, "goal": goal_xy}
    return info, grid, cfg

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Constructive generator using spelunky.json tiles & solids (with --check)")
    ap.add_argument("--config", required=True, help="Path to spelunky.json")
    ap.add_argument("--out", required=True, help="Output .txt")
    # optional overrides
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--ground-rows", type=int, default=1)
    ap.add_argument("--wall-thickness", type=int, default=1)
    ap.add_argument("--step-length", type=int, default=6)
    ap.add_argument("--step-vertical", type=int, default=2)
    ap.add_argument("--step-overlap", type=int, default=1)
    ap.add_argument("--margin-left", type=int, default=2)
    ap.add_argument("--margin-right", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="Run A* solvability check after generating")
    ap.add_argument("--sub", type=int, default=0, help="A* suboptimality band (0 = strict shortest)")
    args = ap.parse_args()

    info, grid, cfg = generate(
        out_path=args.out,
        cfg_path=args.config,
        width=args.width, height=args.height,
        ground_rows=args.ground_rows, wall_thickness=args.wall_thickness,
        step_length=args.step_length, step_vertical=args.step_vertical, step_overlap=args.step_overlap,
        left_margin=args.margin_left, right_margin=args.margin_right,
    )
    print(f"Saved -> {args.out}")
    print({"width": info["width"], "height": info["height"], "start": info["start"], "goal": info["goal"]})

    if args.check:
        ok, extra = astar_ok(grid, cfg, info["start"], info["goal"], sub_optimal=args.sub)
        print("Solvable check:", ok, extra)

if __name__ == "__main__":
    main()

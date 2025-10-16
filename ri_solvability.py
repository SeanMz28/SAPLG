# Usage (CLI):
#   python ri_solvability.py --level "Level 25"        # check one level
#   python ri_solvability.py --all                      # check every .txt under Rainbow Islands/Processed
#   python ri_solvability.py --level "Level 25" --sub 10 --show-path
#
# Programmatic:
#   from ri_solvability import is_level_solvable, check_all_levels
#   ok, info = is_level_solvable("Level 25")

from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple, Optional

from rainbow_islands_parser import load_levels  # loader used in your preview script  :contentReference[oaicite:2]{index=2}
from test_level import findPaths                # A* + platformer neighbor model       :contentReference[oaicite:3]{index=3}

# --- Default Rainbow Islands physics--- (Tune game physics)
# Tiles treated as solid by the solver:
DEFAULT_SOLIDS = ["B", "G"]

# Jump arcs: each entry is a per-frame sequence of (dx, dy) offsets.
DEFAULT_JUMPS: List[List[List[int]]] = [
    # short hop
    [[0, -1], [0, -2], [1, -2], [1, -1]],
    # medium arc
    [[0, -1], [0, -2], [1, -3], [1, -3], [2, -2], [2, -1]],
    # running arc
    [[1, -1], [1, -2], [2, -3], [3, -3], [4, -2]],
    # longer running arc
    [[1, -1], [1, -2], [2, -2], [3, -3], [4, -3], [5, -2]],
]

def _grid_to_lines(grid) -> List[str]:
    """Convert numpy char grid to the list[str] that findPaths expects."""
    return ["".join(row) for row in grid]

def is_level_solvable(
    level_name: str,
    base_dir: str = ".",
    solids: Optional[List[str]] = None,
    jumps: Optional[List[List[List[int]]]] = None,
    sub_optimal: int = 0,
) -> Tuple[bool, Dict]:
    """
    Return (solvable, info) for a single Rainbow Islands level.

    info contains:
      - 'num_paths': number of paths found within best_cost + sub_optimal
      - 'best_length': length (cost) of the shortest path (if any)
      - 'paths': optional list of paths (tuples of (x,y)) if you want to inspect
    """
    levels = load_levels(base_dir=base_dir)  # uses your parser  :contentReference[oaicite:5]{index=5}
    if level_name not in levels:
        raise KeyError(f"Level '{level_name}' not found. Available: {sorted(levels.keys())[:10]} ...")

    lvl = levels[level_name]
    level_str = _grid_to_lines(lvl["grid"])

    solids = solids or DEFAULT_SOLIDS
    jumps = jumps or DEFAULT_JUMPS

    # Use your A* wrapper. By default it starts near (2,2) and succeeds when x == rightmost column.  :contentReference[oaicite:6]{index=6}
    paths = findPaths(sub_optimal, set(solids), jumps, level_str)

    solvable = len(paths) > 0
    info: Dict = {"num_paths": len(paths)}
    if solvable:
        # paths are already reconstructed in order; first one is shortest  :contentReference[oaicite:7]{index=7}
        info["best_length"] = len(paths[0])
        info["paths"] = paths
    return solvable, info

def check_all_levels(
    base_dir: str = ".",
    solids: Optional[List[str]] = None,
    jumps: Optional[List[List[List[int]]]] = None,
    sub_optimal: int = 0,
) -> Dict[str, Dict]:
    """Batch-check every level under Rainbow Islands/Processed; returns {level_name: info}."""
    results: Dict[str, Dict] = {}
    levels = load_levels(base_dir=base_dir)  #  :contentReference[oaicite:8]{index=8}
    for name in sorted(levels.keys()):
        ok, info = is_level_solvable(name, base_dir, solids, jumps, sub_optimal)
        results[name] = {"solvable": ok, **info}
    return results

# ---------------- CLI ----------------
def _main():
    ap = argparse.ArgumentParser(description="Rainbow Islands solvability checker")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--level", type=str, help="Level name (e.g., 'Level 25')")
    g.add_argument("--all", action="store_true", help="Check all levels")
    ap.add_argument("--sub", type=int, default=0, help="Sub-optimality band (allow near-shortest alternatives)")
    ap.add_argument("--solids", type=str, help="JSON list of solid tile chars (override)")
    ap.add_argument("--jumps", type=str, help="Path to JSON file describing jumps (override)")
    ap.add_argument("--show-path", action="store_true", help="Print shortest path coordinates")
    args = ap.parse_args()

    solids = json.loads(args.solids) if args.solids else DEFAULT_SOLIDS
    jumps = DEFAULT_JUMPS
    if args.jumps:
        with open(args.jumps, "r") as f:
            jumps = json.load(f)

    if args.level:
        ok, info = is_level_solvable(args.level, solids=solids, jumps=jumps, sub_optimal=args.sub)
        print(f"{args.level}: {'SOLVABLE' if ok else 'UNSOLVABLE'}")
        if ok:
            print(f"  paths: {info['num_paths']}  best_length: {info['best_length']}")
            if args.show_path:
                print("  shortest path (x,y):")
                print(info["paths"][0])
    else:
        results = check_all_levels(solids=solids, jumps=jumps, sub_optimal=args.sub)
        solved = sum(1 for r in results.values() if r["solvable"])
        print(f"Solved {solved}/{len(results)} levels")
        for name, info in results.items():
            if info["solvable"]:
                print(f"{name}: SOLVABLE (best_length={info['best_length']}, paths={info['num_paths']})")
            else:
                print(f"{name}: UNSOLVABLE")

if __name__ == "__main__":
    _main()

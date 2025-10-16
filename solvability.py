# ri_solvability.py
# ------------------------------------------------------------
# Usage (CLI):
#   python ri_solvability.py --level-file path/to/level.txt --config ./configs/spelunky.json --show-path
#   python ri_solvability.py --level "Level 25" --config ./configs/rainbow_islands.json
#   python ri_solvability.py --all --levels-dir "Rainbow Islands/Processed" --config ./configs/rainbow_islands.json
#
# Programmatic:
#   from ri_solvability import is_level_solvable_universal
#   ok, info = is_level_solvable_universal(level_rows, config)
#
# Requires:
#   - test_level.py providing: findPaths(sub_optimal, solids, jumps, level_rows, start=None, goal_xy=None, is_goal=None)
#   - Optional: rainbow_islands_parser.load_levels(name, base_dir) if you use --level (named levels)
# ------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

# Optional: only needed if you use --level with named Rainbow Islands levels
try:
    from rainbow_islands_parser import load_levels  # your existing loader
except Exception:
    load_levels = None  # still fine if you only use --level-file

# Must exist in your repo
from test_level import findPaths


# ---------------------------
# Helpers
# ---------------------------
def find_char(level_rows: List[str], ch: str) -> Optional[Tuple[int, int]]:
    """Find first (x,y) of character ch in the grid."""
    for y, row in enumerate(level_rows):
        x = row.find(ch)
        if x != -1:
            return (x, y)
    return None


def scrub_chars(rows: List[str], repl: Dict[str, str]) -> List[str]:
    """Replace characters in the grid (e.g., turn start/goal into empty for search)."""
    table = str.maketrans(repl)
    return [row.translate(table) for row in rows]


def validate_rect_grid(rows: List[str]) -> None:
    if not rows:
        raise ValueError("Level has no rows.")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("Level rows must all be the same width (rectangular grid).")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Core: universal solvability
# ---------------------------
def is_level_solvable_universal(
    level_rows: List[str],
    cfg: Dict,
    sub_optimal: int = 0,
    return_paths: bool = False,
) -> Tuple[bool, Dict]:
    """
    Check solvability using explicit Start/Goal defined in `cfg["tiles"]`.
    Returns (solvable, info).

    info:
      - 'num_paths': number of acceptable paths
      - 'best_length': cost/length of the shortest path
      - 'paths' (optional): list of coordinate paths [(x,y), ...]
      - 'start': (x,y)
      - 'goal': (x,y)
    """
    validate_rect_grid(level_rows)

    # --- tiles & physics ---
    tiles = cfg["tiles"]               # {empty, platform, ground, start, goal, ...}
    physics = cfg.get("physics", {})
    solids_list = physics.get("solids", ["X", "#"])
    jumps = physics.get("jumps")  # list of jump arcs (frames of (dx,dy))

    # Never treat start/goal as solid
    solids = set(solids_list)
    solids.discard(tiles["start"])
    solids.discard(tiles["goal"])

    # --- locate S/G ---
    start_xy = find_char(level_rows, tiles["start"])
    goal_xy = find_char(level_rows, tiles["goal"])
    if start_xy is None:
        raise ValueError(f"Start tile '{tiles['start']}' not found in level.")
    if goal_xy is None:
        raise ValueError(f"Goal tile '{tiles['goal']}' not found in level.")

    # --- search grid: scrub S/G to walkable (usually 'empty') ---
    empty_symbol = tiles.get("empty", ".")
    level_rows_search = scrub_chars(level_rows, {
        tiles["start"]: empty_symbol,
        tiles["goal"]: empty_symbol,
    })

    # --- run A* ---
    paths = findPaths(
    sub_optimal,
    solids,
    jumps,
    level_rows_search,
    start_xy,
    goal_xy,
)

    solvable = bool(paths)
    info = {
        "solvable": solvable,
        "num_paths": len(paths) if paths else 0,
        "best_length": len(paths[0]) if paths else None,
        "start": start_xy,
        "goal": goal_xy,
    }
    if return_paths:
        info["paths"] = paths
    return solvable, info


# ---------------------------
# Rainbow Islands convenience
# ---------------------------
def is_named_level_solvable(
    level_name: str,
    cfg: Dict,
    base_dir: str = ".",
    sub_optimal: int = 0,
    return_paths: bool = False,
) -> Tuple[bool, Dict]:
    if load_levels is None:
        raise RuntimeError("rainbow_islands_parser.load_levels is not available. Use --level-file instead.")
    levels = load_levels(level_name, base_dir=base_dir)
    # Expecting levels[level_name] -> List[str] of rows
    level_rows = levels[level_name]
    return is_level_solvable_universal(level_rows, cfg, sub_optimal, return_paths)


def is_file_level_solvable(
    level_file: str,
    cfg: Dict,
    sub_optimal: int = 0,
    return_paths: bool = False,
) -> Tuple[bool, Dict]:
    with open(level_file, "r", encoding="utf-8") as f:
        # allow plain text grids with newline-separated rows
        rows = [line.rstrip("\n") for line in f]
    return is_level_solvable_universal(rows, cfg, sub_optimal, return_paths)


def check_all_levels_in_dir(
    levels_dir: str,
    cfg: Dict,
    sub_optimal: int = 0,
) -> Dict[str, Dict]:
    """
    Scan a directory for .txt grids and check each.
    Returns mapping: filename -> info
    """
    out = {}
    for name in sorted(os.listdir(levels_dir)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(levels_dir, name)
        try:
            ok, info = is_file_level_solvable(path, cfg, sub_optimal, return_paths=False)
            info["solvable"] = ok
            out[name] = info
        except Exception as e:
            out[name] = {"solvable": False, "error": str(e)}
    return out


# ---------------------------
# CLI
# ---------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check platformer level solvability with explicit Start/Goal.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--level-file", type=str, help="Path to a .txt grid level file.")
    g.add_argument("--level", type=str, help="Named level to load via rainbow_islands_parser.load_levels")
    g.add_argument("--all", action="store_true", help="Check every .txt in --levels-dir")
    p.add_argument("--levels-dir", type=str, default="Rainbow Islands/Processed",
                   help="Directory to scan when using --all")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config (tiles, physics, jumps).")
    p.add_argument("--sub", type=int, default=0, help="Accept paths within this extra cost above the best.")
    p.add_argument("--show-path", action="store_true", help="Print the shortest path coordinates.")
    p.add_argument("--base-dir", type=str, default=".", help="Base directory for named levels.")
    return p


def _main():
    args = _build_argparser().parse_args()
    cfg = load_config(args.config)

    if args.level_file:
        ok, info = is_file_level_solvable(args.level_file, cfg, sub_optimal=args.sub, return_paths=args.show_path)
        print(f"{os.path.basename(args.level_file)}: {'SOLVABLE' if ok else 'UNSOLVABLE'}")
        if ok:
            print(f"  best_length={info['best_length']}  paths={info['num_paths']}")
            print(f"  start={info['start']}  goal={info['goal']}")
            if args.show_path and "paths" in info and info["paths"]:
                print("  shortest path (x,y):")
                print(info["paths"][0])

    elif args.level:
        ok, info = is_named_level_solvable(args.level, cfg, base_dir=args.base_dir,
                                           sub_optimal=args.sub, return_paths=args.show_path)
        print(f"{args.level}: {'SOLVABLE' if ok else 'UNSOLVABLE'}")
        if ok:
            print(f"  best_length={info['best_length']}  paths={info['num_paths']}")
            print(f"  start={info['start']}  goal={info['goal']}")
            if args.show_path and "paths" in info and info["paths"]:
                print("  shortest path (x,y):")
                print(info["paths"][0])

    else:  # --all
        results = check_all_levels_in_dir(args.levels_dir, cfg, sub_optimal=args.sub)
        solved = sum(1 for r in results.values() if r.get("solvable"))
        print(f"Solved {solved}/{len(results)} levels")
        for name, info in results.items():
            if info.get("error"):
                print(f"{name}: ERROR: {info['error']}")
            elif info.get("solvable"):
                print(f"{name}: SOLVABLE (best_length={info.get('best_length')}, paths={info.get('num_paths')})")
            else:
                print(f"{name}: UNSOLVABLE")


if __name__ == "__main__":
    _main()

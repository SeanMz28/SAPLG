# solvability.py
# ------------------------------------------------------------
# Spelunky-specific solvability checker with ladder support
# 
# Usage (CLI):
#   python solvability.py --level-file path/to/level.txt --config ./configs/spelunky.json --show-path
#   python solvability.py --all --levels-dir captured_levels --config ./configs/spelunky.json
#
# Programmatic:
#   from solvability import is_level_solvable
#   ok, info = is_level_solvable(level_rows, config)
# ------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Set, Callable
import pathfinding


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
# Spelunky pathfinding with ladders
# ---------------------------
def make_is_solid(solids: Set[str]) -> Callable[[str], bool]:
    def is_solid(tile: str) -> bool:
        return tile in solids
    return is_solid


def make_get_neighbors_spelunky(jumps, level_rows, visited, is_solid, ladder_char='L', max_ropes=4):
    """
    Modified neighbor function that includes ladder climbing and rope mechanics.
    
    State: (x, y, jump_idx, frame_idx, direction, on_ladder, ropes_used, rope_positions)
    - jump_idx: -1 if not jumping, otherwise index into jumps list
    - frame_idx: which frame of the jump animation
    - direction: 1 for right, -1 for left
    - on_ladder: True if currently on a ladder or rope
    - ropes_used: number of ropes used so far
    - rope_positions: frozenset of (x, y_start, y_end) tuples representing placed ropes
    """
    max_x = len(level_rows[0]) - 1
    max_y = len(level_rows) - 1
    
    # Convert absolute jump frames to per-frame deltas
    jump_diffs = []
    for jump in jumps:
        jump_diff = [jump[0]]
        for ii in range(1, len(jump)):
            jump_diff.append((jump[ii][0] - jump[ii-1][0], jump[ii][1] - jump[ii-1][1]))
        jump_diffs.append(jump_diff)
    
    def get_neighbors(pos):
        dist = pos[0] - pos[2]
        state = pos[1]  # (x, y, jump_idx, frame_idx?, dir?, on_ladder?, ropes_used?, rope_positions?)
        x, y = state[0], state[1]
        visited.add((x, y))
        neighbors = []
        
        # Extract rope state if present
        ropes_used = state[6] if len(state) > 6 else 0
        rope_positions = state[7] if len(state) > 7 else frozenset()
        
        # Bounds check
        if y + 1 > max_y or y < 0 or x < 0 or x > max_x:
            return []
        
        current_tile = level_rows[y][x]
        below = (x, y + 1)
        
        # Check if on ladder or rope
        on_ladder = current_tile == ladder_char
        on_rope = any(rope[0] == x and rope[1] <= y <= rope[2] for rope in rope_positions)
        
        # Continue jump if in mid-jump
        if len(state) > 2 and state[2] != -1:
            jump_idx = state[2]
            frame_idx = state[3]
            direction = state[4]
            next_frame = frame_idx + 1
            
            if next_frame < len(jump_diffs[jump_idx]):
                dx = direction * jump_diffs[jump_idx][next_frame][0]
                dy = jump_diffs[jump_idx][next_frame][1]
                new_x, new_y = x + dx, y + dy
                
                if 0 <= new_x <= max_x and 0 <= new_y <= max_y:
                    if not is_solid(level_rows[new_y][new_x]):
                        neighbors.append([dist + 1, (new_x, new_y, jump_idx, next_frame, direction, False, ropes_used, rope_positions)])
        
        # On ladder or rope - can move up/down/dismount
        if on_ladder or on_rope:
            # Climb up
            if y > 0 and not is_solid(level_rows[y - 1][x]):
                neighbors.append([dist + 1, (x, y - 1, -1, 0, 0, False, ropes_used, rope_positions)])
            
            # Climb down (if there's a ladder/rope below or we can stand)
            if y + 1 <= max_y:
                below_tile = level_rows[y + 1][x]
                on_rope_below = any(rope[0] == x and rope[1] <= y + 1 <= rope[2] for rope in rope_positions)
                if below_tile == ladder_char or on_rope_below or not is_solid(below_tile):
                    neighbors.append([dist + 1, (x, y + 1, -1, 0, 0, False, ropes_used, rope_positions)])
            
            # Dismount left
            if x > 0 and not is_solid(level_rows[y][x - 1]):
                neighbors.append([dist + 1, (x - 1, y, -1, 0, 0, False, ropes_used, rope_positions)])
            
            # Dismount right
            if x < max_x and not is_solid(level_rows[y][x + 1]):
                neighbors.append([dist + 1, (x + 1, y, -1, 0, 0, False, ropes_used, rope_positions)])
        
        # Not in a jump and not on ladder - regular ground movement
        elif len(state) <= 2 or state[2] == -1:
            # Can we grab a ladder at current position?
            if current_tile == ladder_char:
                # Already handled above
                pass
            
            # Check if on solid ground
            if below[1] <= max_y and is_solid(level_rows[below[1]][below[0]]):
                # On ground - can walk or jump
                
                # Walk right
                if x + 1 <= max_x:
                    right_tile = level_rows[y][x + 1]
                    if not is_solid(right_tile):
                        neighbors.append([dist + 1, (x + 1, y, -1, 0, 0, False, ropes_used, rope_positions)])
                
                # Walk left
                if x - 1 >= 0:
                    left_tile = level_rows[y][x - 1]
                    if not is_solid(left_tile):
                        neighbors.append([dist + 1, (x - 1, y, -1, 0, 0, False, ropes_used, rope_positions)])
                
                # Grab adjacent ladder (move horizontally onto a ladder)
                if x + 1 <= max_x and level_rows[y][x + 1] == ladder_char:
                    neighbors.append([dist + 1, (x + 1, y, -1, 0, 0, True, ropes_used, rope_positions)])
                if x - 1 >= 0 and level_rows[y][x - 1] == ladder_char:
                    neighbors.append([dist + 1, (x - 1, y, -1, 0, 0, True, ropes_used, rope_positions)])
                
                # Throw rope upward (if we have ropes left)
                if ropes_used < max_ropes:
                    # Find the nearest solid tile above
                    rope_end = None
                    for check_y in range(y - 1, -1, -1):
                        if is_solid(level_rows[check_y][x]):
                            rope_end = check_y + 1  # Rope attaches just below the solid
                            break
                    
                    # Only place rope if there's a ceiling to attach to and rope would be useful
                    if rope_end is not None and rope_end < y:
                        new_rope_positions = rope_positions | frozenset([(x, rope_end, y)])
                        # Move onto the rope immediately
                        neighbors.append([dist + 1, (x, y, -1, 0, 0, True, ropes_used + 1, new_rope_positions)])
                
                # Ledge grab / Wall climb - In Spelunky, you can climb up walls
                # Check if there's a wall next to us and we can climb up it
                # Right wall climb - check if wall to the right and space above the wall
                if x + 1 <= max_x and y >= 2:
                    # Wall next to us at same level or above
                    if is_solid(level_rows[y][x + 1]) or is_solid(level_rows[y - 1][x + 1]):
                        # Check if we can stand on top of the wall (2 blocks up from current position must be empty)
                        if not is_solid(level_rows[y - 2][x + 1]):
                            # Also check the space we'll move through
                            if not is_solid(level_rows[y - 1][x]) or not is_solid(level_rows[y - 1][x + 1]):
                                neighbors.append([dist + 2, (x + 1, y - 2, -1, 0, 0, False, ropes_used, rope_positions)])
                
                # Left wall climb
                if x - 1 >= 0 and y >= 2:
                    if is_solid(level_rows[y][x - 1]) or is_solid(level_rows[y - 1][x - 1]):
                        if not is_solid(level_rows[y - 2][x - 1]):
                            if not is_solid(level_rows[y - 1][x]) or not is_solid(level_rows[y - 1][x - 1]):
                                neighbors.append([dist + 2, (x - 1, y - 2, -1, 0, 0, False, ropes_used, rope_positions)])
                
                # Start jumps
                for jump_idx in range(len(jump_diffs)):
                    dx_right = jump_diffs[jump_idx][0][0]
                    dy = jump_diffs[jump_idx][0][1]
                    
                    # Jump right
                    if x + dx_right <= max_x and y + dy >= 0:
                        if not is_solid(level_rows[y + dy][x + dx_right]):
                            neighbors.append([dist + 1, (x + dx_right, y + dy, jump_idx, 0, 1, False, ropes_used, rope_positions)])
                    
                    # Jump left
                    if x - dx_right >= 0 and y + dy >= 0:
                        if not is_solid(level_rows[y + dy][x - dx_right]):
                            neighbors.append([dist + 1, (x - dx_right, y + dy, jump_idx, 0, -1, False, ropes_used, rope_positions)])
            
            else:
                # In air - falling or can grab onto rope
                # Check if we can grab onto a rope at current position
                if on_rope:
                    # Already on rope, handled above
                    pass
                else:
                    # Falling
                    neighbors.append([dist + 1, (x, y + 1, -1, 0, 0, False, ropes_used, rope_positions)])
                    
                    # Diagonal falling
                    if y + 1 <= max_y:
                        if x + 1 <= max_x and not is_solid(level_rows[y + 1][x + 1]):
                            neighbors.append([dist + 1.4, (x + 1, y + 1, -1, 0, 0, False, ropes_used, rope_positions)])
                        if x - 1 >= 0 and not is_solid(level_rows[y + 1][x - 1]):
                            neighbors.append([dist + 1.4, (x - 1, y + 1, -1, 0, 0, False, ropes_used, rope_positions)])
                    
                    if y + 2 <= max_y:
                        if x + 1 <= max_x and not is_solid(level_rows[y + 2][x + 1]):
                            neighbors.append([dist + 2, (x + 1, y + 2, -1, 0, 0, False, ropes_used, rope_positions)])
                        if x - 1 >= 0 and not is_solid(level_rows[y + 2][x - 1]):
                            neighbors.append([dist + 2, (x - 1, y + 2, -1, 0, 0, False, ropes_used, rope_positions)])
        
        return neighbors
    
    return get_neighbors


def find_paths_spelunky(sub_optimal: int, solids: Set[str], jumps, level_rows, 
                        start: Tuple[int, int], goal: Tuple[int, int],
                        ladder_char: str = 'L', debug: bool = False) -> List[List[Tuple[int, int]]]:
    """
    Find paths in a Spelunky level with ladder support.
    
    Returns list of paths, where each path is a list of (x, y) coordinates.
    """
    visited = set()
    is_solid = make_is_solid(solids)
    get_neighbors = make_get_neighbors_spelunky(jumps, level_rows, visited, is_solid, ladder_char, max_ropes=4)
    
    # Start state: (x, y, jump_idx=-1, frame=0, dir=0, on_ladder=False, ropes_used=0, rope_positions=frozenset())
    start_state = (start[0], start[1], -1, 0, 0, False, 0, frozenset())
    
    if debug:
        print(f"Start state: {start_state}")
        print(f"Start tile: '{level_rows[start[1]][start[0]]}'")
        print(f"Below start: '{level_rows[start[1]+1][start[0]] if start[1]+1 < len(level_rows) else 'OOB'}'")
        print(f"Testing neighbors from start...")
        neighbors = get_neighbors([0, start_state, 0])
        print(f"Found {len(neighbors)} neighbors from start: {neighbors[:5]}")
    
    # Goal check function
    goal_fn = lambda state: (state[0] == goal[0] and state[1] == goal[1])
    
    paths = pathfinding.astar_shortest_path(
        start_state,
        goal_fn,
        get_neighbors,
        sub_optimal,
        lambda state: abs(state[0] - goal[0]) + abs(state[1] - goal[1])  # Manhattan heuristic
    )
    
    # Extract just (x, y) coordinates
    return [[(state[0], state[1]) for state in path] for path in paths]


# ---------------------------
# Core: Spelunky solvability
# ---------------------------
def is_level_solvable(
    level_rows: List[str],
    cfg: Dict,
    sub_optimal: int = 0,
    return_paths: bool = False,
) -> Tuple[bool, Dict]:
    """
    Check Spelunky level solvability with ladder support.
    Returns (solvable, info).
    
    info:
      - 'solvable': bool
      - 'num_paths': number of acceptable paths
      - 'best_length': cost/length of the shortest path
      - 'paths' (optional): list of coordinate paths [(x,y), ...]
      - 'start': (x,y)
      - 'goal': (x,y)
    """
    validate_rect_grid(level_rows)
    
    tiles = cfg["tiles"]
    physics = cfg.get("physics", {})
    solids_list = physics.get("solids", ["1"])
    jumps = physics.get("jumps", [])
    
    # Build solid set (exclude start, goal, ladder)
    solids = set(solids_list)
    solids.discard(tiles.get("start", "D"))
    solids.discard(tiles.get("goal", "E"))
    solids.discard(tiles.get("ladder", "L"))
    solids.discard(tiles.get("empty", "."))
    
    # Find start and goal
    start_xy = find_char(level_rows, tiles.get("start", "D"))
    goal_xy = find_char(level_rows, tiles.get("goal", "E"))
    
    if start_xy is None:
        raise ValueError(f"Start tile '{tiles.get('start')}' not found in level.")
    if goal_xy is None:
        raise ValueError(f"Goal tile '{tiles.get('goal')}' not found in level.")
    
    # Find paths using Spelunky-specific pathfinding
    paths = find_paths_spelunky(
        sub_optimal,
        solids,
        jumps,
        level_rows,
        start_xy,
        goal_xy,
        ladder_char=tiles.get("ladder", "L"),
        debug=False
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


def is_file_level_solvable(
    level_file: str,
    cfg: Dict,
    sub_optimal: int = 0,
    return_paths: bool = False,
) -> Tuple[bool, Dict]:
    with open(level_file, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]
    return is_level_solvable(rows, cfg, sub_optimal, return_paths)


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
    p = argparse.ArgumentParser(description="Check Spelunky level solvability with ladder support.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--level-file", type=str, help="Path to a .txt grid level file.")
    g.add_argument("--all", action="store_true", help="Check every .txt in --levels-dir")
    p.add_argument("--levels-dir", type=str, default="captured_levels",
                   help="Directory to scan when using --all")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config (tiles, physics, jumps).")
    p.add_argument("--sub", type=int, default=0, help="Accept paths within this extra cost above the best.")
    p.add_argument("--show-path", action="store_true", help="Print the shortest path coordinates.")
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
                for step in info["paths"][0]:
                    print(f"    {step}")
    
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
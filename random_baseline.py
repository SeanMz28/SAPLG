"""
Random baseline generator for Spelunky style levels.
- Produces a numpy char grid with tiles from config
- Supports optional solvability check using solvability.is_level_solvable
- Includes ladder generation for vertical navigation
- Can save out to .txt for preview scripts
"""

import numpy as np
import random
import json
import signal
from typing import List, Tuple, Dict
from solvability import is_level_solvable


# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Solvability check timed out")

def with_timeout(func, timeout_seconds=30):
    """Execute function with a timeout. Returns (success, result)."""
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Disable the alarm
        return True, result
    except TimeoutException:
        signal.alarm(0)  # Disable the alarm
        return False, None
    except Exception as e:
        signal.alarm(0)  # Disable the alarm
        return False, None

class GenConfig:
    def __init__(
        self,
        width=32,
        height=32,
        seed=None,
        platform_density=0.7,
        max_segments_per_row=3,
        segment_len_min=2,
        segment_len_max=8,
        ground_rows=1,
        ladder_probability=0.03,
        min_ladder_height=3,
        config_path='configs/spelunky.json'
    ):
        self.width = width
        self.height = height
        self.seed = seed
        self.platform_density = platform_density
        self.max_segments_per_row = max_segments_per_row
        self.segment_len_min = segment_len_min
        self.segment_len_max = segment_len_max
        self.ground_rows = ground_rows
        self.ladder_probability = ladder_probability
        self.min_ladder_height = min_ladder_height
        self.config_path = config_path
        
        # Load config for tiles
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.tiles = self.config['tiles']

# --- GENERATOR ---
def generate_random_level(cfg: GenConfig) -> np.ndarray:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    tiles = cfg.tiles
    empty = tiles['empty']
    platform = tiles['platform']
    ground = tiles['ground']
    ladder = tiles['ladder']
    start_tile = tiles['start']
    goal_tile = tiles['goal']
    
    grid = np.full((cfg.height, cfg.width), empty, dtype='<U1')

    # Add left and right borders if configured
    left_right_border = cfg.config.get('generation', {}).get('left_right_border', True)
    if left_right_border:
        grid[:, 0] = platform
        grid[:, cfg.width - 1] = platform

    # Bottom rows as ground
    for y in range(cfg.height - cfg.ground_rows, cfg.height):
        grid[y, :] = ground

    # Randomly scatter platform segments
    for y in range(cfg.height - cfg.ground_rows - 1):  # exclude ground
        if random.random() < cfg.platform_density:
            nseg = random.randint(1, cfg.max_segments_per_row)
            for _ in range(nseg):
                seg_len = random.randint(cfg.segment_len_min, cfg.segment_len_max)
                # Avoid placing on borders if they exist
                start_margin = 1 if left_right_border else 0
                end_margin = 1 if left_right_border else 0
                if cfg.width - start_margin - end_margin - seg_len > 0:
                    start_x = random.randint(start_margin, cfg.width - end_margin - seg_len)
                    grid[y, start_x:start_x+seg_len] = platform
    
    # Add ladders for vertical navigation
    add_ladders(grid, cfg)
    
    # Place start and goal
    place_start_goal(grid, cfg)
    
    return grid


def add_ladders(grid: np.ndarray, cfg: GenConfig):
    """Add ladders to connect platforms vertically."""
    height, width = grid.shape
    tiles = cfg.tiles
    empty = tiles['empty']
    platform = tiles['platform']
    ground = tiles['ground']
    ladder = tiles['ladder']
    
    # Find platform positions
    for x in range(1, width - 1):
        y = 0
        while y < height - cfg.ground_rows:
            # Check if we're on a platform
            if grid[y, x] in [platform, ground] and y > 0:
                # Check if there's empty space above
                if grid[y - 1, x] == empty:
                    # Randomly decide to place a ladder
                    if random.random() < cfg.ladder_probability:
                        # Find how high the ladder should go
                        ladder_height = 0
                        check_y = y - 1
                        while check_y >= 0 and grid[check_y, x] == empty:
                            ladder_height += 1
                            check_y -= 1
                        
                        # Only place if meets minimum height
                        if ladder_height >= cfg.min_ladder_height:
                            # Place ladder
                            for ladder_y in range(y - ladder_height, y):
                                if 0 <= ladder_y < height:
                                    grid[ladder_y, x] = ladder
            y += 1


def place_start_goal(grid: np.ndarray, cfg: GenConfig):
    """Place start and goal positions on solid ground with good separation."""
    height, width = grid.shape
    tiles = cfg.tiles
    empty = tiles['empty']
    platform = tiles['platform']
    ground = tiles['ground']
    start_tile = tiles['start']
    goal_tile = tiles['goal']
    
    # Find valid positions (on solid ground with empty space above)
    valid_positions = []
    for x in range(1, width - 1):
        for y in range(1, height - cfg.ground_rows):
            # Check if solid below and empty at this position
            if y + 1 < height and grid[y + 1, x] in [platform, ground] and grid[y, x] == empty:
                valid_positions.append((x, y))
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    if len(valid_positions) < 2:
        # Fallback: place far apart on ground level, but different heights if possible
        start_x = 2
        goal_x = width - 3
        # Try to place start higher if possible
        start_y = max(0, height - cfg.ground_rows - 5)  # 5 rows above ground
        goal_y = height - cfg.ground_rows - 1  # Just above ground
        grid[start_y, start_x] = start_tile
        grid[goal_y, goal_x] = goal_tile
    else:
        # Randomly choose gameplay pattern for variety
        pattern = random.choice(['bottom-left_to_top-right', 'top-left_to_bottom-right', 
                                'top-right_to_bottom-left', 'bottom-right_to_top-left'])
        
        left_positions = [pos for pos in valid_positions if pos[0] < width // 2]
        right_positions = [pos for pos in valid_positions if pos[0] >= width // 2]
        top_positions = [pos for pos in valid_positions if pos[1] < height // 2]
        bottom_positions = [pos for pos in valid_positions if pos[1] >= height // 2]
        
        if pattern == 'bottom-left_to_top-right':
            # Start: bottom-left, Goal: top-right
            start_candidates = [p for p in valid_positions if p[0] < width // 2 and p[1] >= height // 2]
            goal_candidates = [p for p in valid_positions if p[0] >= width // 2 and p[1] < height // 2]
        elif pattern == 'top-left_to_bottom-right':
            # Start: top-left, Goal: bottom-right
            start_candidates = [p for p in valid_positions if p[0] < width // 2 and p[1] < height // 2]
            goal_candidates = [p for p in valid_positions if p[0] >= width // 2 and p[1] >= height // 2]
        elif pattern == 'top-right_to_bottom-left':
            # Start: top-right, Goal: bottom-left
            start_candidates = [p for p in valid_positions if p[0] >= width // 2 and p[1] < height // 2]
            goal_candidates = [p for p in valid_positions if p[0] < width // 2 and p[1] >= height // 2]
        else:  # bottom-right_to_top-left
            # Start: bottom-right, Goal: top-left
            start_candidates = [p for p in valid_positions if p[0] >= width // 2 and p[1] >= height // 2]
            goal_candidates = [p for p in valid_positions if p[0] < width // 2 and p[1] < height // 2]
        
        # Fallback if specific quadrants don't have positions
        if not start_candidates:
            start_candidates = left_positions if left_positions else valid_positions
        if not goal_candidates:
            goal_candidates = right_positions if right_positions else valid_positions
        
        start_pos = random.choice(start_candidates)
        
        # Filter goal candidates to ensure good distance from start
        far_goal_candidates = [pos for pos in goal_candidates 
                              if manhattan_distance(start_pos, pos) > width // 2]
        
        if far_goal_candidates:
            goal_pos = random.choice(far_goal_candidates)
        elif goal_candidates:
            goal_pos = random.choice(goal_candidates)
        else:
            # Find furthest position from start
            goal_pos = max(valid_positions, key=lambda p: manhattan_distance(start_pos, p))
        
        grid[start_pos[1], start_pos[0]] = start_tile
        grid[goal_pos[1], goal_pos[0]] = goal_tile

def save_txt(grid: np.ndarray, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for row in grid:
            f.write("".join(row.tolist()) + "\n")

def grid_to_lines(grid: np.ndarray) -> List[str]:
    return ["".join(row.tolist()) for row in grid]

# --- SOLVABILITY CHECK ---
def is_solvable(grid: np.ndarray, config: Dict, sub_optimal=0, timeout=30) -> Tuple[bool, dict]:
    """Check if level is solvable using Spelunky-specific pathfinding with timeout."""
    level_rows = grid_to_lines(grid)
    
    def check():
        return is_level_solvable(level_rows, config, sub_optimal=sub_optimal)
    
    success, result = with_timeout(check, timeout_seconds=timeout)
    
    if success and result:
        solvable, info = result
        return solvable, info
    else:
        # Timeout or error - treat as unsolvable
        return False, {"error": "timeout or error", "num_paths": 0}

def generate_until_solvable(cfg: GenConfig, max_attempts=100, sub_optimal=0, timeout=30, verbose=True):
    """Generate random levels until finding a solvable one."""
    for attempt in range(1, max_attempts + 1):
        if verbose and attempt > 1:
            print(f"  Attempt {attempt}/{max_attempts}...", end='\r')
        
        grid = generate_random_level(cfg)
        ok, info = is_solvable(grid, cfg.config, sub_optimal=sub_optimal, timeout=timeout)
        
        if ok:
            if verbose and attempt > 1:
                print()  # New line after progress
            return grid, info
        
        if cfg.seed is not None:
            cfg.seed += 1
    
    raise RuntimeError(f"Could not generate solvable level after {max_attempts} attempts")

# --- CLI ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Random Spelunky level generator with ladders")
    ap.add_argument("--out", type=str, required=True, help="Path to save generated .txt level")
    ap.add_argument("--config", type=str, default="configs/spelunky.json", 
                    help="Path to config file")
    ap.add_argument("--width", type=int, default=None, help="Override width from config")
    ap.add_argument("--height", type=int, default=None, help="Override height from config")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--platform-density", type=float, default=0.7, 
                    help="Probability of platforms per row")
    ap.add_argument("--ladder-probability", type=float, default=0.03,
                    help="Probability of placing ladders")
    ap.add_argument("--ensure-solvable", action="store_true", 
                    help="Keep generating until solvable")
    ap.add_argument("--max-attempts", type=int, default=100,
                    help="Max attempts for solvable generation")
    ap.add_argument("--timeout", type=int, default=30,
                    help="Timeout in seconds for each solvability check (default: 30)")
    args = ap.parse_args()

    # Load config to get dimensions if not specified
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    width = args.width if args.width is not None else config.get('width', 32)
    height = args.height if args.height is not None else config.get('height', 32)
    
    cfg = GenConfig(
        width=width, 
        height=height, 
        seed=args.seed,
        platform_density=args.platform_density,
        ladder_probability=args.ladder_probability,
        config_path=args.config
    )

    if args.ensure_solvable:
        print(f"Generating solvable level (max {args.max_attempts} attempts, {args.timeout}s timeout per check)...")
        try:
            grid, info = generate_until_solvable(
                cfg, 
                max_attempts=args.max_attempts, 
                timeout=args.timeout,
                verbose=True
            )
            print(f"✅ Generated solvable level: best_length={info['best_length']}, paths={info['num_paths']}")
        except RuntimeError as e:
            print(f"❌ Failed: {e}")
            print(f"💡 Try: Lower platform-density, increase ladder-probability, or increase max-attempts/timeout")
            exit(1)
    else:
        grid = generate_random_level(cfg)
        print(f"Generated level (not checked for solvability)")

    save_txt(grid, args.out)
    print(f"💾 Saved -> {args.out}")


# Usage examples:
# 
# Basic solvable level with recommended parameters:
# python3 random_baseline.py --out "level.txt" --ensure-solvable --platform-density 0.15 --ladder-probability 0.15
# 
# Dense level (may take longer):
# python3 random_baseline.py --out "dense.txt" --ensure-solvable --platform-density 0.7 --ladder-probability 0.03 --timeout 20 --max-attempts 50
# 
# Quick generation without solvability check:
# python3 random_baseline.py --out "quick.txt" --platform-density 0.2 --ladder-probability 0.2
# 
# With specific seed for reproducibility:
# python3 random_baseline.py --out "seeded.txt" --seed 42 --ensure-solvable --timeout 30

#!/usr/bin/env python3
"""Test jump generation from start position"""

import json

# Load level
with open("captured_levels/level_20251023_071250.txt", "r") as f:
    level_rows = [line.rstrip("\n") for line in f]

# Load config
with open("./configs/spelunky.json", "r") as f:
    cfg = json.load(f)

jumps = cfg['physics']['jumps']
print("Jump configurations:")
for i, jump in enumerate(jumps):
    print(f"  Jump {i}: {jump}")
print()

# Start position
start = (29, 6)
x, y = start
max_x = len(level_rows[0]) - 1
max_y = len(level_rows) - 1

print(f"Start: ({x}, {y})")
print(f"Tile at start: '{level_rows[y][x]}'")
print(f"Tile below: '{level_rows[y+1][x]}'")
print()

# Simulate first frame of each jump
for jump_idx, jump in enumerate(jumps):
    dx_right = jump[0][0]
    dy = jump[0][1]
    
    print(f"Jump {jump_idx}: first frame delta = ({dx_right}, {dy})")
    
    # Jump left
    new_x = x - dx_right
    new_y = y + dy
    print(f"  Jump left: ({x}, {y}) -> ({new_x}, {new_y})")
    if new_x >= 0 and new_y >= 0 and new_x <= max_x and new_y <= max_y:
        tile = level_rows[new_y][new_x]
        print(f"    Target tile: '{tile}', is '1'? {tile == '1'}")
    else:
        print(f"    Out of bounds")
    
    # Jump right
    new_x = x + dx_right
    new_y = y + dy
    print(f"  Jump right: ({x}, {y}) -> ({new_x}, {new_y})")
    if new_x >= 0 and new_y >= 0 and new_x <= max_x and new_y <= max_y:
        tile = level_rows[new_y][new_x]
        print(f"    Target tile: '{tile}', is '1'? {tile == '1'}")
    else:
        print(f"    Out of bounds")
    print()

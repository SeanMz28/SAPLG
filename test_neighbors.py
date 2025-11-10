#!/usr/bin/env python3
"""Test neighbor generation for the start position"""

import json
from solvability import is_level_solvable, load_config, make_is_solid, make_get_neighbors_spelunky

# Load level
with open("captured_levels/level_20251023_071250.txt", "r") as f:
    level_rows = [line.rstrip("\n") for line in f]

# Load config
cfg = load_config("./configs/spelunky.json")

tiles = cfg["tiles"]
physics = cfg.get("physics", {})
solids_list = physics.get("solids", ["1"])
jumps = physics.get("jumps", [])

# Build solid set
solids = set(solids_list)
solids.discard(tiles.get("start", "D"))
solids.discard(tiles.get("goal", "E"))
solids.discard(tiles.get("ladder", "L"))
solids.discard(tiles.get("empty", "."))

print(f"Solids: {solids}")
print()

# Start position
start = (29, 6)
print(f"Start position: {start}")
print(f"Start tile: '{level_rows[start[1]][start[0]]}'")
print(f"Tile below: '{level_rows[start[1]+1][start[0]]}'")
print(f"Tile left: '{level_rows[start[1]][start[0]-1]}'")
print(f"Tile right: '{level_rows[start[1]][start[0]+1]}'")
print()

# Create neighbor function
visited = set()
is_solid = make_is_solid(solids)
get_neighbors = make_get_neighbors_spelunky(jumps, level_rows, visited, is_solid, 'L')

# Test from start
start_state = (start[0], start[1], -1, 0, 0, False)
print(f"Start state: {start_state}")
print(f"Is tile below solid? {is_solid(level_rows[start[1]+1][start[0]])}")

# Get neighbors
neighbors = get_neighbors([0, start_state, 0])
print(f"\nNeighbors from start ({len(neighbors)}):")
for i, n in enumerate(neighbors[:10]):
    state = n[1]
    print(f"  {i}: dist={n[0]:.1f}, pos=({state[0]}, {state[1]}), tile='{level_rows[state[1]][state[0]]}'")

if not neighbors:
    print("  NO NEIGHBORS FOUND!")
    print("\nDEBUG INFO:")
    print(f"  Below position: ({start[0]}, {start[1]+1})")
    print(f"  Is below in bounds? {start[1]+1 < len(level_rows)}")
    print(f"  Tile below: '{level_rows[start[1]+1][start[0]]}'")
    print(f"  Is below solid? {is_solid(level_rows[start[1]+1][start[0]])}")

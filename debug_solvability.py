#!/usr/bin/env python3
"""Quick debug script to visualize the level and check positions"""

with open("captured_levels/level_20251023_071250.txt", "r") as f:
    lines = [line.rstrip("\n") for line in f]

print(f"Level dimensions: {len(lines[0])} x {len(lines)}")
print()

# Find D and E
for y, line in enumerate(lines):
    if 'D' in line:
        x = line.index('D')
        print(f"Start (D) at: ({x}, {y})")
        print(f"Row {y}: {line}")
        if y > 0:
            print(f"Above  : {lines[y-1]}")
        if y < len(lines) - 1:
            print(f"Below  : {lines[y+1]}")
        print()
    
    if 'E' in line:
        x = line.index('E')
        print(f"Exit (E) at: ({x}, {y})")
        print(f"Row {y}: {line}")
        if y > 0:
            print(f"Above  : {lines[y-1]}")
        if y < len(lines) - 1:
            print(f"Below  : {lines[y+1]}")
        print()

# Check what's at start position
for y, line in enumerate(lines):
    if 'D' in line:
        x = line.index('D')
        print(f"Tiles around start ({x}, {y}):")
        print(f"  Left:  '{line[x-1] if x > 0 else 'OOB'}'")
        print(f"  Right: '{line[x+1] if x < len(line)-1 else 'OOB'}'")
        print(f"  Above: '{lines[y-1][x] if y > 0 else 'OOB'}'")
        print(f"  Below: '{lines[y+1][x] if y < len(lines)-1 else 'OOB'}'")

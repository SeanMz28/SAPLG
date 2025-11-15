# rainbow_loader.py
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image

# Folders relative to this file / your repo root
PROCESSED_DIR = Path("Rainbow Islands/Processed")
ORIGINAL_DIR  = Path("Rainbow Islands/Original")

# Map characters to integers (tweak as needed for your generator)
# '.' = empty, 'B' = solid block, 'G' = ground/bedrock
TILE_MAP = {'.': 0, 'B': 1, 'G': 2}

def load_level_txt(txt_path: Path, tile_map=TILE_MAP) -> np.ndarray:
    rows: List[List[int]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        # ignore blank lines at file edges
        if not line.strip():
            continue
        rows.append([tile_map.get(ch, 0) for ch in line.rstrip("\n")])
    # pad ragged rows to rectangle if needed
    width = max(len(r) for r in rows)
    grid = np.zeros((len(rows), width), dtype=np.int32)
    for y, r in enumerate(rows):
        grid[y, :len(r)] = r
    return grid  # shape: (rows, cols), top row first

def maybe_load_image(basename: str) -> Optional[Image.Image]:
    # Try a few common extensions
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = ORIGINAL_DIR / f"{basename}{ext}"
        if p.exists():
            return Image.open(p)
    return None

def load_all_levels() -> Dict[str, Tuple[np.ndarray, Optional[Image.Image]]]:
    """Returns {level_name: (grid, original_image_or_None)}"""
    levels: Dict[str, Tuple[np.ndarray, Optional[Image.Image]]] = {}
    for txt_path in sorted(PROCESSED_DIR.glob("*.txt")):
        name = txt_path.stem  # e.g., "Level 25"
        grid = load_level_txt(txt_path)
        img = maybe_load_image(name)
        levels[name] = (grid, img)
    return levels

if __name__ == "__main__":
    levels = load_all_levels()
    print(f"Loaded {len(levels)} Rainbow Islands level(s).")
    # Example: print dimensions for a quick sanity check
    for name, (grid, img) in levels.items():
        h, w = grid.shape
        print(f"{name}: {w}x{h} tiles | image={'yes' if img else 'no'}")

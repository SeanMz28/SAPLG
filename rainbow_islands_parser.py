
# rainbow_islands_parser.py
# Parse Rainbow Islands levels from VGLC-style text grids plus matching images.
# Folder layout (relative to this file or your working directory):
#   ./Rainbow Islands/Processed  -> contains .txt tile grids (e.g., 'Level 25.txt')
#   ./Rainbow Islands/Original   -> contains images     (e.g., 'Level 25.png')
#
# Usage example:
#   from rainbow_islands_parser import load_levels, render_level_ascii
#   levels = load_levels(base_dir='.')  # or path to parent folder containing 'Rainbow Islands'
#   lvl = levels['Level 25']
#   print(lvl['name'], lvl['width'], 'x', lvl['height'])
#   print(render_level_ascii(lvl))
#   # Access numpy grid: lvl['grid']  (dtype='<U1', shape=(H, W))
#   # Access PIL image path (if present): lvl['image_path']
#
# Notes:
# - This parser does not require a JSON legend; it infers the tile set from the text.
# - By default, it treats 'B' and 'G' as solid tiles for convenience (common in VGLC Rainbow Islands).
# - You can override or extend the legend using the 'legend' parameter in load_levels().

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple, Iterable, Any
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    Image = None  # type: ignore

@dataclass
class LevelRecord:
    name: str
    txt_path: str
    image_path: Optional[str]
    grid: np.ndarray  # shape (H, W), dtype '<U1' (characters)
    width: int
    height: int
    tiles: List[str]               # sorted unique tile characters
    legend: Dict[str, List[str]]   # simple legend: char -> list of tags (e.g., ['solid'])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy arrays to Python lists for JSON-serializable dicts (optional)
        d['grid'] = self.grid.tolist()
        return d

# --------- Core parsing helpers ---------
def _read_txt_grid(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    if not lines:
        raise ValueError(f"Empty level file: {path}")
    # Ensure all rows are the same width by right-padding with '.' (empty) if needed
    width = max(len(row) for row in lines)
    norm_rows = [row.ljust(width, '.') for row in lines]
    # Convert to numpy char array (H, W)
    arr = np.array([list(row) for row in norm_rows], dtype='<U1')
    return arr

def _infer_simple_legend(grid: np.ndarray, defaults: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    tiles = sorted({c for c in np.unique(grid)})
    legend = {c: [] for c in tiles}
    # Defaults for Rainbow Islands (can be overridden by caller)
    # '.' -> empty/passable; 'B' -> brick/solid; 'G' -> ground/solid (bottom rows)
    default_tags = {
        '.': ['passable', 'empty'],
        'B': ['solid', 'brick', 'platform'],
        'G': ['solid', 'ground'],
    }
    if defaults:
        default_tags.update(defaults)
    for c in tiles:
        legend[c] = default_tags.get(c, [])
    return legend

def _find_matching_image(original_dir: str, stem: str) -> Optional[str]:
    # Match by stem, accept common image suffixes
    patterns = [os.path.join(original_dir, f"{stem}.*")]
    candidates: List[str] = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    # Filter to images
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'}
    candidates = [p for p in candidates if os.path.splitext(p)[1].lower() in exts]
    if candidates:
        # Prefer png if present
        candidates.sort(key=lambda p: (os.path.splitext(p)[1].lower() != '.png', p.lower()))
        return candidates[0]
    return None

# --------- Public API ---------
def load_levels(
    base_dir: str = '.',
    processed_subdir: str = os.path.join('Rainbow Islands', 'Processed'),
    original_subdir: str = os.path.join('Rainbow Islands', 'Original'),
    legend: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load all .txt levels from Processed and (optionally) attach matching images from Original.

    Returns a dict mapping level name -> level record as a dictionary.
    The key is the filename stem (e.g., 'Level 25').
    """
    processed_dir = os.path.join(base_dir, processed_subdir)
    original_dir  = os.path.join(base_dir, original_subdir)
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    if not os.path.isdir(original_dir):
        # It's okay if images are absent; just warn via comment in code
        original_dir = None  # type: ignore

    out: Dict[str, Dict[str, Any]] = {}
    for txt_path in sorted(glob.glob(os.path.join(processed_dir, '*.txt'))):
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        grid = _read_txt_grid(txt_path)
        H, W = grid.shape
        simple_legend = _infer_simple_legend(grid, legend)
        tiles = sorted(simple_legend.keys())
        image_path = _find_matching_image(original_dir, stem) if original_dir else None
        rec = LevelRecord(
            name=stem,
            txt_path=txt_path,
            image_path=image_path,
            grid=grid,
            width=W,
            height=H,
            tiles=tiles,
            legend=simple_legend,
        )
        out[stem] = {
            'name': rec.name,
            'txt_path': rec.txt_path,
            'image_path': rec.image_path,
            'grid': rec.grid,
            'width': rec.width,
            'height': rec.height,
            'tiles': rec.tiles,
            'legend': rec.legend,
        }
    return out

# --------- Convenience utilities ---------
def get_solid_mask(level: Dict[str, Any], solid_tags: Tuple[str, ...] = ('solid',)) -> np.ndarray:
    """Return a boolean mask of solid tiles, using legend tags (default: any tile with 'solid')."""
    grid: np.ndarray = level['grid']
    legend: Dict[str, List[str]] = level['legend']
    H, W = grid.shape
    mask = np.zeros((H, W), dtype=bool)
    for tile_char, tags in legend.items():
        if any(tag in tags for tag in solid_tags):
            mask |= (grid == tile_char)
    return mask

def render_level_ascii(level: Dict[str, Any]) -> str:
    """Return a single string showing the grid for quick inspection."""
    grid: np.ndarray = level['grid']
    return '\n'.join(''.join(row.tolist()) for row in grid)

def save_level_preview(level: Dict[str, Any], path: str, scale: int = 2) -> Optional[str]:
    """Render a simple image preview from the grid using a hardcoded palette.

    Colors (modifiable): '.'=dark teal, 'B'=yellow, 'G'=green.
    """
    if not _HAS_PIL:
        return None
    palette = {
        '.': (12, 44, 56),     # background
        'B': (228, 205, 31),   # bricks/platforms
        'G': (40, 220, 90),    # ground/grass
    }
    grid: np.ndarray = level['grid']
    H, W = grid.shape
    img = Image.new('RGB', (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            px[x, y] = palette.get(grid[y, x], (200, 200, 200))
    if scale > 1:
        img = img.resize((W*scale, H*scale), resample=Image.NEAREST)
    img.save(path)
    return path

if __name__ == '__main__':
    # Quick manual test: load and summarize
    levels = load_levels(base_dir='.')
    print(f"Loaded {len(levels)} level(s). Keys: {list(levels.keys())[:5]}{'...' if len(levels)>5 else ''}")
    # Show one level summary
    if levels:
        name = sorted(levels.keys())[0]
        lvl = levels[name]
        print(f"Level: {lvl['name']}  size: {lvl['width']}x{lvl['height']}  tiles: {lvl['tiles']}")
        print(render_level_ascii(lvl).splitlines()[:5])  # show first 5 rows
        # Optional: save a preview PNG next to the script
        if _HAS_PIL:
            preview_path = f"{lvl['name']}_preview.png"
            save_level_preview(lvl, preview_path, scale=2)
            print(f"Saved preview -> {preview_path}")

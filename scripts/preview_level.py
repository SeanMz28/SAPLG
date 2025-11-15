import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.rainbow_islands_parser import load_levels, render_level_ascii, get_solid_mask, save_level_preview

levels = load_levels(base_dir='.')
lvl = levels['Level 10']            # key is the filename stem
print(lvl['name'], lvl['width'], 'x', lvl['height'])
print(render_level_ascii(lvl))      # raw grid as text

solid = get_solid_mask(lvl)         # boolean mask using legend tags
save_level_preview(lvl, 'level25_preview.png', scale=3)

# visualize_generated.py
"""
Visualize generated levels for research paper.
Creates individual level visualizations and combined figures.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import json
from src.core.structural_features import Physics, build_segment_graph, structural_metrics

def visualize_all_generated(generated_dir='generated_levels', output_dir='research_paper/figures'):
    """Create publication-quality visualizations of all generated levels."""
    
    generated_path = Path(generated_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tile colors for visualization
    tile_colors = {
        '0': [1, 1, 1],       # White (air)
        '.': [1, 1, 1],       # White (air)
        '1': [0.2, 0.2, 0.2], # Dark gray (solid)
        'D': [0, 0.8, 0],     # Green (entrance)
        'E': [0.8, 0, 0],     # Red (exit)
        'L': [0, 0.4, 0.8],   # Blue (ladder)
    }
    
    # Get all generated fi2pop levels
    gen_files = sorted(generated_path.glob('fi2pop_*.txt'))
    
    print(f"Found {len(gen_files)} generated levels")
    
    # Create individual visualizations
    for gen_file in gen_files:
        with open(gen_file, 'r') as f:
            rows = [line.strip() for line in f if line.strip()]
        
        # Convert to image
        img = level_to_image(rows, tile_colors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, interpolation='nearest', aspect='equal')
        ax.set_title(gen_file.stem.replace('fi2pop_', '').replace('_', ' ').title(), 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save individual
        output_file = output_path / f'{gen_file.stem}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"[OK] Saved {output_file}")
    
    # Create combined 3x3 grid figure for paper
    create_combined_figure(gen_files, tile_colors, output_path)
    
    print(f"\n💾 All visualizations saved to {output_path}/")


def create_combined_figure(gen_files, tile_colors, output_path):
    """Create a 3x3 grid of all 9 style profiles."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, gen_file in enumerate(gen_files[:9]):
        with open(gen_file, 'r') as f:
            rows = [line.strip() for line in f if line.strip()]
        
        img = level_to_image(rows, tile_colors)
        
        axes[i].imshow(img, interpolation='nearest', aspect='equal')
        title = gen_file.stem.replace('fi2pop_', '').replace('_', ' ').title()
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    output_file = output_path / 'all_profiles_grid.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved combined grid: {output_file}")


def level_to_image(rows, tile_colors):
    """Convert text level to RGB image."""
    if not rows:
        return np.ones((32, 32, 3))
    
    H = len(rows)
    W = max(len(row) for row in rows)
    img = np.ones((H, W, 3))  # Default white
    
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            img[i, j] = tile_colors.get(char, [0.5, 0.5, 0.5])  # Gray for unknown
    
    return img


def extract_metrics_from_generated(generated_dir='generated_levels', config_path='configs/spelunky.json'):
    """Extract structural metrics from all generated levels."""
    
    # Load physics config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    physics = Physics(
        solids=set(config['physics']['solids']),
        jumps=[[(int(dx), int(dy)) for dx, dy in arc] 
               for arc in config['physics']['jumps']]
    )
    
    generated_path = Path(generated_dir)
    gen_files = sorted(generated_path.glob('fi2pop_*.txt'))
    
    results = {}
    
    for gen_file in gen_files:
        with open(gen_file, 'r') as f:
            rows = [line.strip() for line in f if line.strip()]
        
        try:
            G, id2seg = build_segment_graph(rows, physics)
            metrics = structural_metrics(G, id2seg, max_len_for_style=None)
            
            style_name = gen_file.stem.replace('fi2pop_', '')
            results[style_name] = {
                'room_count': metrics['room_count'],
                'branching': metrics['branching'],
                'linearity': metrics['linearity'],
                'dead_end_rate': metrics['dead_end_rate'],
                'loop_complexity': metrics['loop_complexity'],
            }
            
            print(f"{style_name:20s} | R={metrics['room_count']:.1f} B={metrics['branching']:.2f} "
                  f"L={metrics['linearity']:.2f} D={metrics['dead_end_rate']:.2f} Q={metrics['loop_complexity']:.1f}")
        except Exception as e:
            print(f"WARNING: Error extracting metrics from {gen_file.stem}: {e}")
    
    # Save to JSON
    output_file = generated_path / 'generated_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved metrics to {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize generated levels for paper")
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--metrics', action='store_true', help='Extract metrics')
    parser.add_argument('--all', action='store_true', help='Do both')
    
    args = parser.parse_args()
    
    if args.all or args.visualize:
        visualize_all_generated()
    
    if args.all or args.metrics:
        extract_metrics_from_generated()
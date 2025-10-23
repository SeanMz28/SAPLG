# visualize_generated.py
"""
Visualize generated levels vs real levels.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

def visualize_levels(generated_dir='generated_levels', real_dir='captured_levels', num_samples=6):
    """Compare generated vs real levels side-by-side."""
    
    generated_path = Path(generated_dir)
    real_path = Path(real_dir)
    
    gen_files = list(generated_path.glob('*.txt'))[:num_samples]
    real_files = random.sample(list(real_path.glob('*.txt')), num_samples)
    
    tile_colors = {
        '0': [1, 1, 1],      # White (air)
        '1': [0.3, 0.3, 0.3], # Dark gray (solid)
        'D': [0, 1, 0],      # Green (door)
        'E': [1, 0, 0],      # Red (exit)
        'L': [0, 0, 1],      # Blue (ladder)
    }
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 2))
    
    for i, (gen_file, real_file) in enumerate(zip(gen_files, real_files)):
        # Load levels
        with open(gen_file, 'r') as f:
            gen_rows = [line.strip() for line in f]
        
        with open(real_file, 'r') as f:
            real_rows = [line.strip() for line in f]
        
        # Convert to RGB images
        gen_img = level_to_image(gen_rows, tile_colors)
        real_img = level_to_image(real_rows, tile_colors)
        
        # Plot
        axes[i, 0].imshow(gen_img, interpolation='nearest')
        axes[i, 0].set_title(f'Generated: {gen_file.stem}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(real_img, interpolation='nearest')
        axes[i, 1].set_title(f'Real: {real_file.stem}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(generated_path / 'comparison.png', dpi=150)
    print(f"💾 Saved visualization to {generated_path / 'comparison.png'}")
    plt.show()


def level_to_image(rows, tile_colors):
    """Convert text level to RGB image."""
    H, W = len(rows), len(rows[0])
    img = np.zeros((H, W, 3))
    
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            img[i, j] = tile_colors.get(char, [0.5, 0.5, 0.5])  # Gray for unknown
    
    return img


if __name__ == "__main__":
    visualize_levels()
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import json

from src.generators.generator import StyleAwareGenerator


def generate_levels(
    checkpoint_path,
    style_vectors,
    num_samples_per_style=5,
    output_dir='generated_levels',
    device='cuda'
):
    """
    Generate levels from trained generator.
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pt file)
        style_vectors: List of style vectors to generate from
        num_samples_per_style: Number of variations per style
        output_dir: Where to save generated levels
        device: 'cuda' or 'cpu'
    """
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create generator
    generator = StyleAwareGenerator(
        style_dim=6,
        noise_dim=100,
        num_tiles=5
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print(f"[OK] Loaded model from epoch {checkpoint['epoch']}")
    
    # Tile mapping (reverse of tile_to_idx)
    idx_to_tile = {0: '0', 1: '1', 2: 'D', 3: 'E', 4: 'L'}
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate levels
    generated_count = 0
    
    for style_idx, style_vector in enumerate(style_vectors):
        print(f"\n Style {style_idx + 1}/{len(style_vectors)}: {style_vector}")
        
        # Convert to tensor
        style_tensor = torch.tensor([style_vector], dtype=torch.float32).to(device)
        
        for sample_idx in range(num_samples_per_style):
            with torch.no_grad():
                # Generate with random noise
                fake_level = generator(style_tensor)  # (1, 5, 32, 32)
                
                # Convert logits to tile indices (argmax across tile dimension)
                level_indices = torch.argmax(fake_level[0], dim=0).cpu().numpy()  # (32, 32)
                
                # Convert to text
                level_text = indices_to_text(level_indices, idx_to_tile)
                
                # Save
                filename = output_path / f"generated_style{style_idx}_sample{sample_idx}.txt"
                with open(filename, 'w') as f:
                    f.write('\n'.join(level_text))
                
                generated_count += 1
                print(f"  [OK] Generated: {filename.name}")
    
    print(f"\n Generated {generated_count} levels in {output_dir}/")
    return output_path


def indices_to_text(level_indices, idx_to_tile):
    """Convert index array to text level."""
    H, W = level_indices.shape
    level_text = []
    
    for i in range(H):
        row = ''.join([idx_to_tile[idx] for idx in level_indices[i]])
        level_text.append(row)
    
    return level_text


def load_real_style_vectors(metrics_file, num_samples=5):
    """Load real style vectors from your dataset for comparison."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Filter successful extractions
    valid_metrics = [
        v for v in metrics.values() 
        if v.get('success', False)
    ]
    
    # Sample random styles
    import random
    sampled = random.sample(valid_metrics, min(num_samples, len(valid_metrics)))
    
    style_vectors = []
    for m in sampled:
        style_vectors.append([
            m['room_count'],
            m['branching'],
            m['linearity'],
            m['dead_end_rate'],
            m['loop_complexity'],
            m['segment_size_variance']
        ])
    
    return style_vectors


def main():
    parser = argparse.ArgumentParser(description='Generate levels from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoints/epoch_0100.pt)')
    parser.add_argument('--num_styles', type=int, default=5,
                        help='Number of different styles to try')
    parser.add_argument('--samples_per_style', type=int, default=3,
                        help='Number of samples per style')
    parser.add_argument('--output_dir', type=str, default='generated_levels',
                        help='Output directory')
    parser.add_argument('--metrics_file', type=str, default='spelunky_metrics.json',
                        help='Use real style vectors from this file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[STATS] Loading style vectors from {args.metrics_file}")
    style_vectors = load_real_style_vectors(args.metrics_file, args.num_styles)
    
    # Generate levels
    generate_levels(
        checkpoint_path=args.checkpoint,
        style_vectors=style_vectors,
        num_samples_per_style=args.samples_per_style,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == "__main__":
    main()
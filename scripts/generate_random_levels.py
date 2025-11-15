"""
Pre-generate random levels for FI-2POP initial population.
Creates a library of random levels that can be sampled during evolution.
Only saves levels that pass solvability checking.
Adds variation through randomized platform density.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import numpy as np
from tqdm import tqdm
from src.generators.random_baseline import generate_until_solvable, GenConfig, grid_to_lines

def generate_and_save_random_levels(
    output_dir: str = "random_levels",
    num_levels: int = 100,
    width: int = 32,
    height: int = 32,
    max_attempts_per_level: int = 10,
    timeout: int = 10,
    platform_density_range: tuple = (0.1, 0.7)
):
    """
    Generate random levels with variation and save to directory.
    Only saves solvable levels.
    
    Args:
        output_dir: Directory to save levels
        num_levels: Target number of solvable levels to generate
        width: Level width
        height: Level height
        max_attempts_per_level: Max attempts to generate a solvable level before moving on
        timeout: Timeout in seconds for solvability check
        platform_density_range: (min, max) for platform density variation
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load physics config for tiles
    with open('configs/spelunky.json', 'r') as f:
        physics_config = json.load(f)
    tiles = physics_config['tiles']
    
    print(f"Generating {num_levels} solvable random levels with variation...")
    print(f"Output directory: {output_dir}")
    print(f"Solvability timeout: {timeout}s per level")
    print(f"Max attempts per level: {max_attempts_per_level}")
    print(f"Platform density range: {platform_density_range}")
    print("=" * 60)
    
    successful = 0
    total_attempts = 0
    failed_solvability = 0
    failed_generation = 0
    
    # Track seed if needed
    current_seed = None
    
    pbar = tqdm(total=num_levels, desc="Generating solvable levels")
    
    while successful < num_levels:
        try:
            # Randomize platform density for variation
            platform_density = random.uniform(platform_density_range[0], platform_density_range[1])
            
            # Create config with randomized platform density
            gen_cfg = GenConfig(
                width=width,
                height=height,
                ground_rows=2,
                platform_density=platform_density,
                seed=current_seed
            )
            
            # Use generate_until_solvable which has built-in retry logic
            grid, info = generate_until_solvable(
                gen_cfg,
                max_attempts=max_attempts_per_level,
                sub_optimal=0,
                timeout=timeout,
                verbose=False  # Don't print individual attempts
            )
            
            # Convert to lines
            level_rows = grid_to_lines(grid)
            
            # Save to file
            output_file = output_path / f"random_{successful:04d}.txt"
            with open(output_file, 'w') as f:
                for row in level_rows:
                    f.write(row + '\n')
            
            successful += 1
            pbar.update(1)
            
            # Update seed for next generation
            if current_seed is not None:
                current_seed += max_attempts_per_level
            
        except RuntimeError as e:
            # Could not generate solvable level after max_attempts
            failed_solvability += max_attempts_per_level
            total_attempts += max_attempts_per_level
            pbar.write(f"WARNING: Skipping after {max_attempts_per_level} failed attempts")
        except Exception as e:
            failed_generation += 1
            pbar.write(f"WARNING: Generation error: {e}")
    
    pbar.close()
    
    print("\n" + "=" * 60)
    print(f"Successfully generated {successful} solvable random levels")
    if failed_solvability > 0 or failed_generation > 0:
        print(f"Statistics:")
        print(f"   Failed attempts:       {failed_solvability + failed_generation}")
        print(f"   Failed solvability:    {failed_solvability}")
        print(f"   Failed generation:     {failed_generation}")
    print(f"💾 Saved to: {output_dir}/")
    
    return successful


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate solvable random levels with variation for FI-2POP")
    parser.add_argument('--output', type=str, default='random_levels',
                       help='Output directory')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of solvable levels to generate')
    parser.add_argument('--width', type=int, default=32,
                       help='Level width')
    parser.add_argument('--height', type=int, default=32,
                       help='Level height')
    parser.add_argument('--max-attempts', type=int, default=10,
                       help='Max attempts to generate each solvable level')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout in seconds for solvability check')
    
    # Variation parameters
    parser.add_argument('--platform-density-min', type=float, default=0.1,
                       help='Minimum platform density')
    parser.add_argument('--platform-density-max', type=float, default=0.7,
                       help='Maximum platform density (max 0.7)')
    
    args = parser.parse_args()
    
    # Clamp max density to 0.7
    max_density = min(args.platform_density_max, 0.7)
    
    generate_and_save_random_levels(
        output_dir=args.output,
        num_levels=args.count,
        width=args.width,
        height=args.height,
        max_attempts_per_level=args.max_attempts,
        timeout=args.timeout,
        platform_density_range=(args.platform_density_min, max_density)
    )

"""
Generate 100 constructive levels with variation in step length and step overlap.
"""
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import random
from src.generators.constructive import generate

def main():
    # Configuration
    config_path = "configs/spelunky.json"
    output_dir = "constructive_levels"
    num_levels = 100
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Variation ranges
    step_length_range = (3, 8)  # Min and max step length
    step_overlap_range = (0, 3)  # Min and max step overlap
    
    print(f"Generating {num_levels} constructive levels...")
    print(f"Step length range: {step_length_range}")
    print(f"Step overlap range: {step_overlap_range}")
    print(f"Output directory: {output_dir}")
    print()
    
    for i in range(num_levels):
        # Randomly select step parameters for variation
        step_length = random.randint(*step_length_range)
        step_overlap = random.randint(*step_overlap_range)
        
        # Ensure overlap is less than step length
        if step_overlap >= step_length:
            step_overlap = step_length - 1
        
        # Generate output filename
        out_path = os.path.join(output_dir, f"constructive_level_{i+1:03d}.txt")
        
        # Generate the level
        try:
            info, grid, cfg = generate(
                out_path=out_path,
                cfg_path=config_path,
                step_length=step_length,
                step_overlap=step_overlap,
                ground_rows=1,
                wall_thickness=1,
                step_vertical=3,
                left_margin=2,
                right_margin=2
            )
            
            print(f"Level {i+1:3d}/{num_levels}: step_length={step_length}, step_overlap={step_overlap} -> {out_path}")
            
        except Exception as e:
            print(f"Error generating level {i+1}: {e}")
            continue
    
    print()
    print(f"Successfully generated {num_levels} levels in {output_dir}/")

if __name__ == "__main__":
    main()

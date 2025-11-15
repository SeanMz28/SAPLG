"""
Convenience script to run FI-2POP with different target styles.

PREREQUISITES:
  Run prepare_initial_population.py first to generate level libraries:
    python prepare_initial_population.py

This will create:
  - random_levels/       (random generated levels)
  - constructive_levels/ (constructive generated levels)
  - captured_levels/     (already exists with Spelunky levels)
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.generators.fi2pop_generator import FI2POPGenerator, FI2POPConfig

# Check if level libraries exist
required_dirs = ['captured_levels', 'constructive_levels', 'random_levels']
missing_dirs = [d for d in required_dirs if not Path(d).exists()]

if missing_dirs:
    print("WARNING: ERROR: Missing level libraries!")
    print(f"   Missing: {', '.join(missing_dirs)}")
    print()
    print("Please run the preparation script first:")
    print("  python prepare_initial_population.py")
    print()
    print("This will generate the required level libraries.")
    exit(1)

# Load your extracted metrics to get realistic targets
with open('spelunky_metrics_summary.json', 'r') as f:
    summary = json.load(f)

# Load physics config
with open('configs/spelunky.json', 'r') as f:
    physics_config = json.load(f)

# Define target styles
target_styles = {
    'high_branching': summary['sample_targets']['high_branching'],
    'high_linearity': summary['sample_targets']['high_linearity'],
    'balanced': summary['sample_targets']['balanced'],

    # New styles
    'maze_complex': {
        'room_count': 9.0,
        'branching': 2.27,
        'linearity': 0.24,
        'dead_end_rate': 0.46,
        'loop_complexity': 12.0,
        'segment_size_variance': 4.66
    },
    'open_sparse': {
        'room_count': 1.0,
        'branching': 1.80,
        'linearity': 0.33,
        'dead_end_rate': 0.07,
        'loop_complexity': 0.0,
        'segment_size_variance': 1.57
    },
    'dense_network': {
        'room_count': 5.0,
        'branching': 2.27,
        'linearity': 0.24,
        'dead_end_rate': 0.18,
        'loop_complexity': 12.0,
        'segment_size_variance': 3.11
    },
    'web_branching': {
        'room_count': 2.0,
        'branching': 2.67,        # Max branching
        'linearity': 0.16,        # Min linearity
        'dead_end_rate': 0.07,    # Min dead ends (everything connects)
        'loop_complexity': 24.0,  # Max loops
        'segment_size_variance': 7.76
    },

    'vertical_focused': {
        'room_count': 3.0,
        'branching': 2.07,        # 25th percentile (less horizontal branching)
        'linearity': 0.16,        # Min linearity (more vertical movement)
        'dead_end_rate': 0.18,    # 25th percentile
        'loop_complexity': 7.0,   # 25th percentile
        'segment_size_variance': 9.75  # Max variance (varied platform sizes)
    },

    'minimalist': {
    'room_count': 1.0,        # Min
    'branching': 1.80,        # Min
    'linearity': 0.54,        # Max (very linear)
    'dead_end_rate': 0.18,    # 25th percentile
    'loop_complexity': 0.0,   # No loops
    'segment_size_variance': 1.57  # Min variance
    }

}

print(f" FI-2POP Generator - OPTIMIZED RUN")
print(f"=" * 60)
print(f"Population sources:")
print(f"  - ~33% Captured Spelunky levels (captured_levels/)")
print(f"  - ~33% Constructive levels (constructive_levels/)")
print(f"  - ~33% Random levels (random_levels/)")
print(f"=" * 60)
print(f"\n OPTIMIZED: Pop=50, Gen=30, High Mutation, Styles=9")
print(f"=" * 60)

# Run FI-2POP for each style
for style_name, target_metrics in target_styles.items():
    print(f"\n{'='*60}")
    print(f"Generating level with style: {style_name.upper()}")
    print(f"Target metrics: {target_metrics}")
    print(f"{'='*60}\n")
    
    cfg = FI2POPConfig(
        width=32,
        height=32,
        population_size=50,      # Smaller, faster
        max_generations=20,      # Reduced - no improvement after gen 10 anyway
        mutation_rate=0.15,      # HIGHER mutation for diversity
        crossover_rate=0.7       # Slightly lower crossover
    )
    
    generator = FI2POPGenerator(target_metrics, physics_config, cfg)
    best_level, best_fitness = generator.evolve()
    
    output_path = f"generated_levels/fi2pop_{style_name}.txt"
    generator.save_best(output_path)
    
    print(f"\n[OK] Generated {style_name} level with fitness {best_fitness:.4f}")
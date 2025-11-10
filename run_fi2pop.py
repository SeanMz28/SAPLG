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

import json
from pathlib import Path
from fi2pop_generator import FI2POPGenerator, FI2POPConfig

# Check if level libraries exist
required_dirs = ['captured_levels', 'constructive_levels', 'random_levels']
missing_dirs = [d for d in required_dirs if not Path(d).exists()]

if missing_dirs:
    print("⚠️ ERROR: Missing level libraries!")
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
    'balanced': summary['sample_targets']['balanced']
}

print(f"🎮 FI-2POP Generator - SMALL SCALE TEST")
print(f"=" * 60)
print(f"Population sources:")
print(f"  - ~33% Captured Spelunky levels (captured_levels/)")
print(f"  - ~33% Constructive levels (constructive_levels/)")
print(f"  - ~33% Random levels (random_levels/)")
print(f"=" * 60)
print(f"\n⚡ TEST MODE: Small population (20) and few generations (20)")
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
        population_size=100,  
        max_generations=100
    )
    
    generator = FI2POPGenerator(target_metrics, physics_config, cfg)
    best_level, best_fitness = generator.evolve()
    
    output_path = f"generated_levels/fi2pop_{style_name}.txt"
    generator.save_best(output_path)
    
    print(f"\n✅ Generated {style_name} level with fitness {best_fitness:.4f}")
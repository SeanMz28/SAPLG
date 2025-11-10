"""
Test script to verify the new folder-based level loading works.
This is a quick test before running full FI-2POP evolution.
"""

import json
from pathlib import Path
from fi2pop_generator import FI2POPGenerator, FI2POPConfig

# Check prerequisites
print("Checking prerequisites...")
print("=" * 60)

required_dirs = {
    'captured_levels': 'Real Spelunky levels',
    'constructive_levels': 'Pre-generated constructive levels',
    'random_levels': 'Pre-generated random levels'
}

missing = []
for dir_name, description in required_dirs.items():
    path = Path(dir_name)
    if path.exists():
        count = len(list(path.glob("*.txt")))
        print(f"✅ {dir_name:20s} - {count} files found")
    else:
        print(f"❌ {dir_name:20s} - NOT FOUND")
        missing.append(dir_name)

if missing:
    print("\n⚠️ Missing directories. Run this first:")
    print("   python prepare_initial_population.py")
    exit(1)

print("\n" + "=" * 60)
print("Running quick FI-2POP test...")
print("=" * 60 + "\n")

# Load configs
with open('spelunky_metrics_summary.json', 'r') as f:
    summary = json.load(f)

with open('configs/spelunky.json', 'r') as f:
    physics_config = json.load(f)

# Use balanced target for test
target_metrics = summary['sample_targets']['balanced']

# Create generator with small population for quick test
cfg = FI2POPConfig(
    width=40,
    height=32,
    population_size=15,   # Small for quick test
    max_generations=3     # Just a few generations to verify it works
)

generator = FI2POPGenerator(target_metrics, physics_config, cfg)

# This will load from folders
best_level, best_fitness = generator.evolve()

print("\n" + "=" * 60)
print("✅ TEST SUCCESSFUL!")
print("=" * 60)
print(f"Best fitness achieved: {best_fitness:.4f}")
print(f"Final population: {len(generator.feasible_pop)} feasible, "
      f"{len(generator.infeasible_pop)} infeasible")

# Save test result
generator.save_best("test_fi2pop_folder_loading.txt")

print("\n🎉 Folder-based level loading works correctly!")
print("You can now run: python run_fi2pop.py")


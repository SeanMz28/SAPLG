"""
Quick test to verify captured levels can be loaded into FI-2POP
"""
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.generators.fi2pop_generator import FI2POPGenerator, FI2POPConfig

# Load config
with open('configs/spelunky.json', 'r') as f:
    physics_config = json.load(f)

# Simple target metrics (doesn't matter for this test)
target_metrics = {
    'branching': 0.5,
    'linearity': 0.5,
    'dead_end_rate': 0.3,
    'loop_complexity': 0.2,
    'room_count': 10
}

# Create small test population
cfg = FI2POPConfig(
    width=40,
    height=32,
    population_size=20,  # Small for quick test
    max_generations=1
)

# Test loading captured levels
generator = FI2POPGenerator(target_metrics, physics_config, cfg)
print("\n Testing captured level loading...")
print(f"Target population size: {cfg.population_size}")
print(f"Captured levels directory: captured_levels/")

# Initialize with captured levels
generator.initialize_population(captured_levels_dir='captured_levels')

print(f"\n[OK] Results:")
print(f"   Feasible population: {len(generator.feasible_pop)} levels")
print(f"   Infeasible population: {len(generator.infeasible_pop)} levels")
print(f"   Total: {len(generator.feasible_pop) + len(generator.infeasible_pop)} levels")

# Show some stats about captured levels
if generator.feasible_pop:
    print(f"\n[STATS] Feasible level fitness scores:")
    for i, (level, fitness) in enumerate(generator.feasible_pop[:5]):
        print(f"   Level {i+1}: fitness={fitness:.4f}, shape={level.shape}")

if generator.infeasible_pop:
    print(f"\n[STATS] Infeasible level fitness scores:")
    for i, (level, fitness) in enumerate(generator.infeasible_pop[:5]):
        print(f"   Level {i+1}: fitness={fitness:.4f}, shape={level.shape}")

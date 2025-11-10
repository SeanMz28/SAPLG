# FI-2POP Level Generator - Setup Guide

This guide explains how to prepare and run the FI-2POP genetic algorithm for style-aware platformer level generation.

## Overview

The FI-2POP generator uses three types of levels in its initial population:
1. **Captured levels** - Real Spelunky levels (already in `captured_levels/`)
2. **Constructive levels** - Generated using the constructive algorithm (guaranteed solvable)
3. **Random levels** - Randomly generated levels for diversity

## Quick Start

### Step 1: Prepare Level Libraries (One-time setup)

Generate the random and constructive level libraries:

```bash
python prepare_initial_population.py
```

This creates:
- `random_levels/` - 100 random levels
- `constructive_levels/` - 100 constructive levels

**Options:**
```bash
# Generate more levels
python prepare_initial_population.py --random 200 --constructive 200

# Custom dimensions
python prepare_initial_population.py --width 50 --height 40
```

### Step 2: Run FI-2POP Generator

Once the libraries are prepared, run the evolution:

```bash
python run_fi2pop.py
```

This generates levels for three target styles:
- High branching
- High linearity  
- Balanced

Output files saved to `generated_levels/`:
- `fi2pop_high_branching.txt`
- `fi2pop_high_linearity.txt`
- `fi2pop_balanced.txt`

## Manual Level Generation

### Generate Random Levels Only

```bash
python generate_random_levels.py --count 100 --output random_levels
```

### Generate Constructive Levels Only

```bash
python generate_constructive_levels.py --count 100 --output constructive_levels
```

## Directory Structure

```
SAPLG/
├── captured_levels/          # Real Spelunky levels (200+)
├── random_levels/            # Generated random levels
├── constructive_levels/      # Generated constructive levels
├── generated_levels/         # FI-2POP output
│
├── prepare_initial_population.py   # Master setup script
├── generate_random_levels.py       # Random level generator
├── generate_constructive_levels.py # Constructive level generator
├── run_fi2pop.py                   # Run FI-2POP evolution
└── fi2pop_generator.py             # FI-2POP implementation
```

## Workflow Benefits

✅ **Fast iterations** - Pre-generate once, reuse many times  
✅ **Reproducibility** - Same starting population across runs  
✅ **Debugging** - Can inspect initial population levels  
✅ **Flexibility** - Generate large libraries offline, sample during evolution  

## Advanced Usage

### Custom Target Metrics

Edit `run_fi2pop.py` and modify the `target_styles` dictionary:

```python
target_styles = {
    'my_style': {
        'branching': 1.5,
        'linearity': 0.8,
        'dead_end_rate': 0.2,
        'loop_complexity': 0.3,
        'room_count': 4.0
    }
}
```

### Adjust Evolution Parameters

In `run_fi2pop.py`, modify the `FI2POPConfig`:

```python
cfg = FI2POPConfig(
    width=40,
    height=32,
    population_size=100,      # Larger population
    max_generations=200,      # More generations
    mutation_rate=0.1,        # Higher mutation
    crossover_rate=0.9        # More crossover
)
```

### Using Different Population Sources

In `fi2pop_generator.py`, modify `initialize_population()` call:

```python
# Use only captured and constructive (no random)
generator.initialize_population(
    captured_dir="captured_levels",
    constructive_dir="constructive_levels",
    random_dir=None  # Skip random levels
)
```

## Troubleshooting

**Missing level libraries error:**
```
Run: python prepare_initial_population.py
```

**Not enough levels loaded:**
- Check that directories exist and contain .txt files
- Generator will auto-generate random levels to fill shortage

**Constructive generation fails:**
- Ensure `configs/spelunky.json` exists
- Check that constructive.py is working: `python constructive.py test_out.txt`

## Performance Tips

- Pre-generate larger libraries (200-500 levels each) for better diversity
- Use smaller population size (20-30) for quick testing
- Increase generations (200-500) for better convergence to target metrics
- Monitor feasible population count - should stay above 10-20

## Next Steps

1. ✅ Prepare libraries: `python prepare_initial_population.py`
2. ✅ Run evolution: `python run_fi2pop.py`
3. ✅ Inspect results in `generated_levels/`
4. ✅ Visualize levels: `python preview_level.py generated_levels/fi2pop_balanced.txt`
5. ✅ Test solvability: `python solvability.py generated_levels/fi2pop_balanced.txt`

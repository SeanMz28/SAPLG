# Level Generation Variation Guide

## Overview

Both **random** and **constructive** level generators now include **randomized variation** to create diverse levels while maintaining guaranteed solvability.

## Random Level Variation

Each random level is generated with a randomized **platform density**.

### Platform Density
- **Default range:** 0.1 - 0.7 (max capped at 0.7)
- **Effect:** Controls how many platforms appear per row
- **Examples:**
  - `density=0.1`: Sparse, minimal platforms (harder)
  - `density=0.4`: Moderate platform count (balanced)
  - `density=0.7`: Dense platforms (easier navigation)

## Constructive Level Variation

## Constructive Level Variation

Each constructive level is generated with randomized values for staircase parameters:

### 1. **Step Length** (platform width)
- **Default range:** 3-8 tiles
- **Effect:** Longer steps = easier jumps, wider platforms
- **Example:** 
  - `step_length=3`: `###` (tight platforms)
  - `step_length=8`: `########` (wide platforms)

### 2. **Step Vertical** (height between platforms)
- **Default range:** 2-4 tiles
- **Effect:** Higher spacing = more challenging vertical navigation
- **Example:**
  - `step_vertical=2`: Gentle slope (2 tiles up per step)
  - `step_vertical=4`: Steep slope (4 tiles up per step)

### 3. **Step Overlap** (horizontal overlap between steps)
- **Default range:** 0-2 tiles
- **Effect:** More overlap = easier progression, tighter staircase
- **Example:**
  - `overlap=0`: `###  ###` (gap between steps)
  - `overlap=2`: `#####` (steps overlap by 2 tiles)

## Usage

### Quick Start (Default Variation)

**Both random and constructive:**
```bash
# Uses default ranges for all parameters
python3 prepare_initial_population.py
```

Default random variation:
- Platform density: 0.1-0.7

Default constructive variation:
- Step length: 3-8
- Step vertical: 2-4
- Step overlap: 0-2

### Random Levels Only

```bash
# Default variation
python3 generate_random_levels.py --count 100

# Custom density range (sparse levels)
python3 generate_random_levels.py \
  --count 100 \
  --platform-density-min 0.05 \
  --platform-density-max 0.3

# Custom density range (dense levels)
python3 generate_random_levels.py \
  --count 100 \
  --platform-density-min 0.4 \
  --platform-density-max 0.7
```

### Constructive Levels Only

```bash
# Wide platforms, gentle slopes
python3 generate_constructive_levels.py \
  --count 100 \
  --step-length-min 5 \
  --step-length-max 10 \
  --step-vertical-min 2 \
  --step-vertical-max 3 \
  --step-overlap-min 1 \
  --step-overlap-max 3

# Tight platforms, steep slopes
python3 generate_constructive_levels.py \
  --count 100 \
  --step-length-min 2 \
  --step-length-max 4 \
  --step-vertical-min 3 \
  --step-vertical-max 5 \
  --step-overlap-min 0 \
  --step-overlap-max 1
```

### Using prepare_initial_population.py

```bash
# With custom random AND constructive variation
python3 prepare_initial_population.py \
  --random 100 \
  --constructive 100 \
  --platform-density-min 0.15 \
  --platform-density-max 0.6 \
  --step-length-min 4 \
  --step-length-max 7 \
  --step-vertical-min 2 \
  --step-vertical-max 3
```

## Random Variation Presets

### Preset 1: Sparse Levels (Challenging)
```bash
python3 generate_random_levels.py \
  --platform-density-min 0.05 \
  --platform-density-max 0.25
```
- Minimal platforms
- Large gaps to navigate
- **Effect:** High difficulty, requires precise jumps

### Preset 2: Moderate Levels (Balanced)
```bash
python3 generate_random_levels.py \
  --platform-density-min 0.2 \
  --platform-density-max 0.5
```
- Balanced platform count
- Good mix of gaps and platforms
- **Effect:** Medium difficulty (recommended)

### Preset 3: Dense Levels (Easier)
```bash
python3 generate_random_levels.py \
  --platform-density-min 0.4 \
  --platform-density-max 0.7
```
- Many platforms
- Easier navigation
- **Effect:** Lower difficulty, beginner-friendly

### Preset 4: Maximum Diversity
```bash
python3 generate_random_levels.py \
  --platform-density-min 0.05 \
  --platform-density-max 0.7
```
- Full density range
- Maximum structural variety
- **Effect:** Very diverse difficulty levels

## Constructive Variation Presets

### Preset 1: Easy Traversal
```bash
python3 generate_constructive_levels.py \
  --step-length-min 6 --step-length-max 10 \
  --step-vertical-min 2 --step-vertical-max 3 \
  --step-overlap-min 2 --step-overlap-max 3
```
- Wide platforms
- Gentle slopes
- Good overlap
- **Effect:** Easy to navigate, beginner-friendly

### Preset 2: Moderate Challenge
```bash
python3 generate_constructive_levels.py \
  --step-length-min 4 --step-length-max 7 \
  --step-vertical-min 2 --step-vertical-max 4 \
  --step-overlap-min 1 --step-overlap-max 2
```
- Medium platforms
- Variable slopes
- Some overlap
- **Effect:** Balanced difficulty (default)

### Preset 3: High Challenge
```bash
python3 generate_constructive_levels.py \
  --step-length-min 2 --step-length-max 5 \
  --step-vertical-min 3 --step-vertical-max 5 \
  --step-overlap-min 0 --step-overlap-max 1
```
- Narrow platforms
- Steep slopes
- Minimal overlap
- **Effect:** Challenging navigation

### Preset 4: Maximum Diversity
```bash
python3 generate_constructive_levels.py \
  --step-length-min 2 --step-length-max 10 \
  --step-vertical-min 2 --step-vertical-max 5 \
  --step-overlap-min 0 --step-overlap-max 3
```
- Wide range for all parameters
- Maximum structural variety
- **Effect:** Very diverse levels

## How Variation Works

### Random Levels

For each level generated:

1. **Random platform density:** Value chosen from specified range (e.g., 0.1-0.7)
2. **Generation:** Random algorithm uses this density
3. **Solvability check:** Level must pass verification
4. **Retry if needed:** If not solvable, try again with new density

Example generation sequence:
```
Level 0: density=0.23 → solvable ✓
Level 1: density=0.61 → solvable ✓
Level 2: density=0.15 → not solvable, retry with density=0.44 → solvable ✓
...
```

### Constructive Levels

For each level generated:

1. **Random selection:** Parameters are randomly chosen from specified ranges
2. **Generation:** Constructive algorithm uses these parameters
3. **Verification:** Solvability is checked (levels should still be solvable)
4. **Retry if needed:** If verification fails, try again with new random parameters

Example generation sequence:
```
Level 0: step_length=5, step_vertical=3, overlap=1
Level 1: step_length=7, step_vertical=2, overlap=2
Level 2: step_length=3, step_vertical=4, overlap=0
...
```

## Benefits

✅ **Diversity:** Each level (random & constructive) has unique structure  
✅ **Still solvable:** All levels verified for navigability  
✅ **Controllable:** Adjust ranges to get desired difficulty distribution  
✅ **Better initial population:** More varied starting point for FI-2POP  
✅ **Complementary variation:** Random varies density, constructive varies structure  

## Impact on FI-2POP

### Without Variation
- Random levels similar density → limited diversity
- Constructive levels all look identical → minimal structural variety
- Limited exploration of design space

### With Variation
- **Random:** Diverse platform densities (sparse to dense)
- **Constructive:** Diverse staircase structures
- **Combined:** Maximum structural diversity in initial population
- Faster convergence to target metrics
- Better exploration of fitness landscape

## Checking Your Levels

After generation, you can inspect the variety:

```bash
# View a few generated levels
python3 preview_level.py constructive_levels/constructive_0000.txt
python3 preview_level.py constructive_levels/constructive_0010.txt
python3 preview_level.py constructive_levels/constructive_0020.txt

# Extract metrics to see diversity
python3 run_structural_metrics.py
```

## Troubleshooting

### Random: Too many solvability failures

**Problem:** Success rate <5% or taking very long

**Solutions:**
```bash
# Increase density range (more platforms = easier to solve)
--platform-density-min 0.2 --platform-density-max 0.7

# Increase timeout (gives harder levels more time)
--timeout 15

# Increase max attempts
--max-attempts 50
```

### Random: Not enough diversity

**Problem:** All levels feel similar

**Solutions:**
```bash
# Widen density range
--platform-density-min 0.05 --platform-density-max 0.7

# Or use maximum diversity preset
```

### Constructive: Too many solvability failures
```bash
# Reduce vertical range (easier jumps)
--step-vertical-min 2 --step-vertical-max 3

# Increase overlap (easier traversal)
--step-overlap-min 1 --step-overlap-max 3

# Increase platform width
--step-length-min 5 --step-length-max 10

# Increase timeout
--timeout 10
```

### Constructive: Not enough diversity

**Problem:** Levels look too similar

**Solutions:**
```bash
# Widen all ranges
--step-length-min 2 --step-length-max 10
--step-vertical-min 2 --step-vertical-max 5
--step-overlap-min 0 --step-overlap-max 3

# Or use maximum diversity preset (see above)
```

### All levels too easy/hard

**Problem:** Difficulty not matching expectations

**Solutions:**
```bash
# For easier RANDOM levels
--platform-density-min 0.4 --platform-density-max 0.7

# For harder RANDOM levels
--platform-density-min 0.05 --platform-density-max 0.3

# For easier CONSTRUCTIVE levels (see Easy Traversal preset above)
# For harder CONSTRUCTIVE levels (see High Challenge preset above)
```

## Combining Random and Constructive Variation

**Best practice:** Use complementary ranges for diverse initial population

```bash
python3 prepare_initial_population.py \
  --random 100 \
  --constructive 100 \
  --platform-density-min 0.1 --platform-density-max 0.7 \
  --step-length-min 3 --step-length-max 8 \
  --step-vertical-min 2 --step-vertical-max 4
```

This gives you:
- **100 random levels** with varying density (sparse to dense)
- **100 constructive levels** with varying staircase structure
- **Total: 200 diverse solvable levels** for FI-2POP

## Advanced: Understanding Random Density

### Step Length + Overlap
- `length=4, overlap=2`: Platforms share 2 tiles
- `length=8, overlap=0`: Platforms have gaps between them
- **Sweet spot:** `length=5-7, overlap=1-2` for balanced gameplay

### Step Vertical + Level Height
- Higher vertical spacing means fewer total steps fit
- `vertical=2` on `height=32` = ~16 steps
- `vertical=4` on `height=32` = ~8 steps
- **Consider:** Adjust based on level height

### All Three Together
- **Tight, steep, no overlap:** Maximum challenge
- **Wide, gentle, good overlap:** Maximum ease
- **Mix:** Most interesting levels have varied parameters

## Recommended Defaults

For most use cases, the **default ranges work well**:

**Random:**
```python
platform_density_range=(0.1, 0.7)
```

**Constructive:**
```python
step_length_range=(3, 8)
step_vertical_range=(2, 4)
step_overlap_range=(0, 2)
```

These provide:
- Good diversity in both types
- Consistently solvable levels
- Balanced difficulty distribution
- Reasonable generation speed

## Summary Commands

```bash
# Default (recommended) - maximum diversity
python3 prepare_initial_population.py

# High diversity random, moderate constructive
python3 prepare_initial_population.py \
  --platform-density-min 0.05 --platform-density-max 0.7 \
  --step-length-min 4 --step-length-max 7

# Conservative (fast, reliable) for both
python3 prepare_initial_population.py \
  --platform-density-min 0.3 --platform-density-max 0.6 \
  --step-length-min 5 --step-length-max 7 \
  --step-vertical-min 2 --step-vertical-max 3

# Maximum variation for production
python3 prepare_initial_population.py \
  --random 500 \
  --constructive 500 \
  --platform-density-min 0.05 --platform-density-max 0.7 \
  --step-length-min 2 --step-length-max 10 \
  --step-vertical-min 2 --step-vertical-max 5
```

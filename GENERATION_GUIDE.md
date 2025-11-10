# Level Generation Guide with Solvability Verification

## Overview

Both random and constructive level generators now include **solvability verification** to ensure all generated levels in the initial population are actually solvable.

**NEW:** Constructive generator includes **parameter variation** to create diverse levels while maintaining solvability!

## Quick Start

### Generate Both (Recommended)

```bash
python3 prepare_initial_population.py
```

Default settings:
- 100 random solvable levels
- 100 constructive verified levels
- 5 second timeout per solvability check
- Max 10 attempts per level

### Generate Random Levels Only

```bash
python3 generate_random_levels.py
```

### Generate Constructive Levels Only

```bash
python3 generate_constructive_levels.py
```

## Configuration Options

### Basic Settings

```bash
# Generate 200 of each type
python3 prepare_initial_population.py --random 200 --constructive 200

# Custom dimensions
python3 prepare_initial_population.py --width 50 --height 40
```

### Solvability Settings

```bash
# Longer timeout for complex levels (10 seconds)
python3 prepare_initial_population.py --timeout 10

# More attempts per level (useful for random generation)
python3 prepare_initial_population.py --max-attempts 20

# Conservative settings (fast, fewer attempts)
python3 prepare_initial_population.py --timeout 3 --max-attempts 5
```

### Separate Generation with Custom Settings

#### Random Levels

```bash
# High-effort random generation
python3 generate_random_levels.py \
  --count 200 \
  --max-attempts 50 \
  --timeout 10

# Quick random generation  
python3 generate_random_levels.py \
  --count 50 \
  --max-attempts 5 \
  --timeout 3
```

#### Constructive Levels

```bash
# Standard constructive with verification
python3 generate_constructive_levels.py \
  --count 100 \
  --timeout 5

# Skip verification (not recommended but faster)
python3 generate_constructive_levels.py \
  --count 100 \
  --no-verify
```

## How It Works

### Random Level Generation

1. **Generate** random level using random_baseline
2. **Check solvability** with timeout
3. **If solvable**: Save to `random_levels/`
4. **If not solvable**: Try again (up to max_attempts)
5. **Move on** after max_attempts reached

**Success rate:** Typically 5-20% depending on random generation parameters

### Constructive Level Generation

1. **Generate** level using constructive algorithm with **randomized step parameters**
2. **Verify solvability** (even though constructive should be solvable)
3. **If verified**: Save to `constructive_levels/`
4. **If failed**: Retry with new random parameters (up to max_attempts)

**Variation:** Each level uses different:
- Step length (platform width): 3-8 tiles
- Step vertical (height spacing): 2-4 tiles  
- Step overlap (horizontal overlap): 0-2 tiles

**Success rate:** Typically 90-100% (constructive is designed to be solvable)

## Performance Tips

### For Random Levels

Random generation has **lower success rate**, so:

- Increase `--max-attempts` (e.g., 20-50)
- Use longer `--timeout` (e.g., 10s) for complex levels
- Generate more than needed and sample the best ones
- Expect ~5-20 attempts per successful level

### For Constructive Levels

Constructive has **high success rate**, so:

- Lower `--max-attempts` (e.g., 3-5) is usually enough
- Standard `--timeout` (5s) is fine
- Can use `--no-verify` to skip checking (faster, but risky)
- Expect ~1-2 attempts per successful level

## Understanding Output

### Statistics Explained

```
✅ Successfully generated 100 solvable random levels
📊 Statistics:
   Total attempts:         453        ← Total generation attempts
   Success rate:           22.1%      ← Percentage that passed
   Failed solvability:     320        ← Didn't pass solvability check
   Failed generation:      33         ← Generation error
   Avg attempts per level: 4.5        ← Average tries needed
```

### What's Good?

**Random Levels:**
- Success rate >10%: Good
- Avg attempts <10: Excellent
- Avg attempts <20: Acceptable
- Avg attempts >30: Consider adjusting parameters

**Constructive Levels:**
- Success rate >80%: Good
- Avg attempts <3: Excellent
- Avg attempts >5: Something might be wrong

## Troubleshooting

### Random: Too Many Failed Attempts

**Problem:** Success rate <5% or avg attempts >50

**Solutions:**
```bash
# Increase timeout (gives harder levels more time)
python3 generate_random_levels.py --timeout 15

# Increase max attempts
python3 generate_random_levels.py --max-attempts 100

# Or generate a larger batch and accept lower target
python3 generate_random_levels.py --count 50 --max-attempts 20
```

### Constructive: Verification Failures

**Problem:** Constructive levels failing solvability

**Possible causes:**
- Bug in constructive generator
- Timeout too short for generated level complexity
- Physics config mismatch

**Solutions:**
```bash
# Increase timeout
python3 generate_constructive_levels.py --timeout 10

# Skip verification (not recommended)
python3 generate_constructive_levels.py --no-verify

# Check constructive generator
python3 constructive.py test_output.txt
python3 solvability.py test_output.txt
```

### Generation Too Slow

**Problem:** Taking hours to generate levels

**Solutions:**
```bash
# Reduce timeout
--timeout 3

# Reduce max attempts
--max-attempts 5

# Generate fewer levels initially
--random 50 --constructive 50

# For random, accept that it's slow (solvability checking is expensive)
# Consider running overnight for large batches
```

## Recommended Presets

### Quick Test (5-10 minutes)

```bash
python3 prepare_initial_population.py \
  --random 20 \
  --constructive 20 \
  --timeout 3 \
  --max-attempts 5
```

### Standard (30-60 minutes)

```bash
python3 prepare_initial_population.py \
  --random 100 \
  --constructive 100 \
  --timeout 5 \
  --max-attempts 10
```

### High Quality (2-4 hours)

```bash
python3 prepare_initial_population.py \
  --random 200 \
  --constructive 200 \
  --timeout 10 \
  --max-attempts 20
```

### Production (4-8 hours)

```bash
python3 prepare_initial_population.py \
  --random 500 \
  --constructive 500 \
  --timeout 10 \
  --max-attempts 30
```

## What Gets Generated

### File Structure

```
random_levels/
├── random_0000.txt    ← All verified solvable
├── random_0001.txt
├── ...
└── random_0099.txt

constructive_levels/
├── constructive_0000.txt    ← All verified solvable
├── constructive_0001.txt
├── ...
└── constructive_0099.txt
```

### Level Properties

**Random levels:**
- ✅ Verified solvable
- 🎲 High diversity
- ⚡ Variable difficulty
- 🏗️ Natural-looking structure

**Constructive levels:**
- ✅ Guaranteed solvable by design
- ✅ Double-verified through checking
- 🎯 Consistent quality
- 📐 Well-structured paths
- 🎲 **Varied staircase parameters** for diversity

## Next Steps

After generation:

1. **Check status**: `python3 check_fi2pop_status.py`
2. **Test setup**: `python3 test_captured_in_fi2pop.py`
3. **Run FI-2POP**: `python3 run_fi2pop.py`

## Advanced: Parallel Generation

Generate random and constructive in parallel:

```bash
# Terminal 1
python3 generate_random_levels.py --count 200 --max-attempts 20

# Terminal 2  
python3 generate_constructive_levels.py --count 200 --timeout 5
```

This can cut total generation time significantly!

## Variation Control

Control constructive level diversity (see `VARIATION_GUIDE.md` for details):

```bash
# Maximum diversity
python3 prepare_initial_population.py \
  --step-length-min 2 --step-length-max 10 \
  --step-vertical-min 2 --step-vertical-max 5

# Conservative (fast, less varied)
python3 prepare_initial_population.py \
  --step-length-min 5 --step-length-max 7 \
  --step-vertical-min 2 --step-vertical-max 3
```

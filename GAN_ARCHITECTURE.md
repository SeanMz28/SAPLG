# GAN Architecture & Style Vector Documentation

**Project:** SAPLG (Style-Aware Platformer Level Generation)  
**Author:** SeanMz28  
**Date:** 2025-10-23  
**Model Type:** Style-Aware Generative Adversarial Network (GAN)

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [6D Style Vector Explained](#6d-style-vector-explained)
- [How Metrics Work Together](#how-metrics-work-together)
- [Current Issues](#current-issues)
- [Solutions](#solutions)

---

## Overview

This project uses a **Style-Aware Generative Adversarial Network (GAN)** to generate Spelunky-style platformer levels. Unlike search-based procedural content generation (PCG) methods, this approach:

- ✅ **Generates levels in a single forward pass** (~milliseconds)
- ✅ **Learns implicit patterns from real Spelunky levels** (528 training samples)
- ✅ **Conditions generation on 6D style vectors** for controllable output
- ✅ **Uses graph-based feature extraction** to capture level structure

### Method Classification

| Aspect | Type |
|--------|------|
| **Approach** | Learning-based (NOT search-based) |
| **Architecture** | Generative Adversarial Network (GAN) |
| **Generation** | Constructive/Generative (neural network forward pass) |
| **Style Control** | Conditional on 6D structural metrics |

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Real Levels (32×32) + Style Vectors (6D)                │
│            ↓                    ↓                         │
│     ┌──────────────┐    ┌──────────────┐                │
│     │  Generator   │←───│ Noise (100D) │                │
│     │              │←───│ Style (6D)   │                │
│     └──────┬───────┘    └──────────────┘                │
│            │                                              │
│            ↓ Generated Level (32×32)                     │
│            │                                              │
│     ┌──────┴───────────────────────────┐                │
│     │      Discriminator (Dual-Head)    │                │
│     ├───────────────────────────────────┤                │
│     │  Head 1: Real/Fake Classification │                │
│     │  Head 2: Style Vector Prediction  │                │
│     └──────┬────────────────────┬───────┘                │
│            │                    │                         │
│            ↓                    ↓                         │
│    Adversarial Loss      Style Loss                      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Generator

- **Input:** 
  - Noise vector (100D, sampled from N(0,1))
  - Style vector (6D, structural metrics)
- **Architecture:** 
  - Fully connected layer (projects to 256×8×8)
  - ConvTranspose2d layers (upsampling)
  - Output: 32×32 grid with 10 tile types
- **Parameters:** ~2.6M

### Discriminator

- **Input:** 32×32 level grid
- **Architecture:**
  - Convolutional layers
  - **Dual heads:**
    1. Real/Fake classifier (1D output)
    2. Style predictor (6D output)
- **Parameters:** ~1.1M

### Loss Function

```python
# Generator loss
G_loss = G_adv + (style_weight * G_style)

# Discriminator loss
D_loss = D_real + D_fake + (style_weight * D_style)
```
---

## 6D Style Vector Explained

Each Spelunky level is represented by a 6-dimensional vector capturing its structural properties:

```python
style_vector = [
    room_count,           # Dimension 1
    branching,            # Dimension 2
    linearity,            # Dimension 3
    dead_end_rate,        # Dimension 4
    loop_complexity,      # Dimension 5
    segment_size_variance # Dimension 6
]
```

**Example from real levels:**
```python
Style A: [6.0, 2.085, 0.357, 0.243, 9.0, 2.633]  # Moderate complexity
Style B: [5.0, 2.233, 0.217, 0.333, 12.0, 8.936] # High loop complexity
```

---

### Metric 1: `room_count` 🏠

**What it measures:** Total number of connected platform segments/rooms

**Graph definition:** Number of nodes in the level's connectivity graph

**Range:** 3-10 (typical Spelunky levels)

**Impact:**
- **Low (3-4):** Simple, linear levels
- **Medium (5-7):** Moderate complexity
- **High (8+):** Complex, multi-path levels

**Example:**
```
room_count = 4:
[Start] → [Room1] → [Room2] → [Exit]

room_count = 8:
         ┌→ [Room2] → [Room4] ┐
[Start] →│→ [Room3] → [Room5] │→ [Exit]
         └→ [Room6] → [Room7] ┘
```

---

### Metric 2: `branching` 🌳

**What it measures:** Average number of connections per room (graph degree)

**Formula:**
```python
branching = total_edges / total_nodes
```

**Range:** 1.0-3.0

**Impact:**
- **1.0-1.5:** Linear paths, minimal choice
- **1.5-2.5:** Some branching (typical)
- **2.5+:** Heavy branching, many paths

**Example:**
```
branching = 1.0 (linear):
A → B → C → D

branching = 2.0 (branching):
    ┌→ B ┐
A →│→ C │→ E
    └→ D ┘
```

**Interpretation:**
- **≈ 1.0:** Chain structure (each room connects to 1 other)
- **≈ 2.0:** Each room connects to ~2 others (typical Spelunky)
- **> 3.0:** Hub-and-spoke or dense mesh

---

### Metric 3: `linearity` 📏

**What it measures:** How direct the path from start to exit is

**Formula:**
```python
linearity = shortest_path_length / total_rooms
```

**Range:** 0.0-1.0

**Impact:**
- **High (0.8-1.0):** Very linear, direct path
- **Medium (0.4-0.7):** Some exploration
- **Low (0.1-0.3):** Very exploratory, many detours

**Example:**
```
linearity = 1.0 (perfectly linear):
[Start] → [R1] → [R2] → [R3] → [Exit]
Shortest = 5, Total = 5

linearity = 0.5 (exploratory):
[Start] → [R1] ┐
          [R2] → [R3] → [Exit]
          [R4] → [R5] ┘
Shortest = 4, Total = 8
```

**Gameplay:**
- **High:** Speedrun-friendly, straightforward
- **Low:** Exploration-focused, optional content

---

### Metric 4: `dead_end_rate` 🚫

**What it measures:** Proportion of rooms with no exit (excluding goal)

**Formula:**
```python
dead_end_rate = dead_end_rooms / total_rooms
```

**Range:** 0.0-0.5

**Impact:**
- **Low (0.0-0.2):** Few dead ends, rewarding exploration
- **Medium (0.2-0.4):** Balanced (typical Spelunky)
- **High (0.5+):** Punishing, many wasted paths

**Example:**
```
dead_end_rate = 0.33 (2 of 6 rooms):

[Start] → [R1] → [R2] → [Exit]
           ↓      ↓
        [Dead1] [Dead2]  ← Dead ends
```

**Gameplay:**
- **Low:** Encourages exploration (everything connects)
- **High:** Discourages exploration (risk of wasted time)

---

### Metric 5: `loop_complexity` 🔄

**What it measures:** Number of cycles/loops in the level graph

**Formula (Cyclomatic Complexity):**
```python
loop_complexity = num_edges - num_nodes + 1
```

**Range:** 0-15+

**Impact:**
- **0:** Tree structure (no loops)
- **1-3:** Minimal loops
- **4-8:** Some loops
- **9+:** Highly interconnected

**Example:**
```
loop_complexity = 0 (tree):
    A
   / \
  B   C
 /     \
D       E

loop_complexity = 2 (two cycles):
A → B → C
↓   ↓   ↓
D → E → F
↓       ↓
G ←─────┘
```

**Gameplay:**
- **0:** No backtracking, one-way progression
- **High:** Multiple approach angles, escape routes

---

### Metric 6: `segment_size_variance` 📐

**What it measures:** Statistical variance in platform segment sizes

**Formula:**
```python
segment_size_variance = variance([len(seg) for seg in segments])
```

**Range:** 0-10+

**Impact:**
- **Low (0-2):** Uniform platform sizes
- **Medium (3-6):** Mix of sizes
- **High (7+):** Very diverse (tiny + huge)

**Example:**
```
Low variance (1.2):
Platforms: [5, 6, 5, 6, 5, 6]  ← All similar

High variance (8.9):
Platforms: [2, 10, 3, 12, 1, 15]  ← Very different
```

**Gameplay:**
- **Low:** Predictable jumps
- **High:** Varied platforming challenges

---

## How Metrics Work Together

### Example Style Profiles

#### **Linear Speedrun Style**
```python
[4.0, 1.5, 0.8, 0.25, 2.0, 3.5]
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| room_count | 4.0 | Few rooms |
| branching | 1.5 | Minimal branching |
| linearity | 0.8 | Very direct path |
| dead_end_rate | 0.25 | Some dead ends |
| loop_complexity | 2.0 | Almost no loops |
| segment_variance | 3.5 | Moderate variety |

**Result:** Simple, straightforward level good for speedrunning

**Visual:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     S → → → → → → → E
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

#### **Exploratory Maze Style**
```python
[8.0, 2.5, 0.2, 0.35, 12.0, 7.0]
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| room_count | 8.0 | Many rooms |
| branching | 2.5 | High branching |
| linearity | 0.2 | Very non-linear |
| dead_end_rate | 0.35 | Many dead ends |
| loop_complexity | 12.0 | Many loops |
| segment_variance | 7.0 | Very varied platforms |

**Result:** Complex maze with lots of exploration

**Visual:**
```
    ╔═══╗     ╔═══╗
    ║ R1║─────║ R4║
    ╚═══╝     ╚═══╝
      │         │
╔═══╗ │   ╔═══╗ │   ╔═══╗
║ S ║─┼───║ R2║─┼───║ E ║
╚═══╝ │   ╚═══╝ │   ╚═══╝
      │         │
    ╔═══╗     ╔═══╗
    ║ R3║─────║ R5║
    ╚═══╝     ╚═══╝
```

---

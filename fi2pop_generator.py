"""
FI-2POP Genetic Algorithm for Style-Aware Level Generation

Implements the methodology from your research proposal:
- Two populations: feasible (solvable) and infeasible (not solvable)
- Fitness function combines solvability + structural metric similarity
- Genetic operators: crossover, mutation
- Evolves toward target structural metrics while maintaining solvability
"""

from __future__ import annotations
import random
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Your existing modules
from structural_features import Physics, build_segment_graph, structural_metrics
from solvability import is_level_solvable
from constructive import generate as generate_constructive
from random_baseline import generate_random_level, GenConfig, grid_to_lines


@dataclass
class FI2POPConfig:
    """Configuration for FI-2POP generator."""
    width: int = 32
    height: int = 32
    population_size: int = 50
    max_generations: int = 200
    mutation_rate: float = 0.05
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    
    # Weights for fitness function
    metric_weight: float = 1.0
    solvability_weight: float = 2.0


class FI2POPGenerator:
    """
    FI-2POP Genetic Algorithm for platformer level generation.
    
    Maintains two populations:
    - Feasible: solvable levels
    - Infeasible: unsolvable levels (potential for repair)
    """
    
    def __init__(
        self,
        target_metrics: Dict[str, float],
        physics_config: Dict,
        config: FI2POPConfig = None
    ):
        self.target = target_metrics
        self.physics_config = physics_config
        self.cfg = config or FI2POPConfig()
        
        # Setup physics for structural metrics
        self.physics = Physics(
            solids=set(physics_config["physics"]["solids"]),
            jumps=[[(int(dx), int(dy)) for dx, dy in arc] 
                   for arc in physics_config["physics"]["jumps"]]
        )
        
        self.tiles = physics_config["tiles"]
        
        # Populations: list of (level, fitness) tuples
        self.feasible_pop: List[Tuple[np.ndarray, float]] = []
        self.infeasible_pop: List[Tuple[np.ndarray, float]] = []
        
        # Track best ever
        self.best_level = None
        self.best_fitness = -float('inf')
        
        # History
        self.history = {
            'gen': [],
            'best_fitness': [],
            'avg_fitness': [],
            'feasible_count': [],
            'metric_distance': []
        }
    
    def initialize_population(
        self,
        captured_dir: str = "captured_levels",
        constructive_dir: str = "constructive_levels",
        random_dir: str = "random_levels"
    ):
        """
        Create initial population by loading from pre-generated level folders.
        
        Args:
            captured_dir: Directory with captured Spelunky levels
            constructive_dir: Directory with pre-generated constructive levels
            random_dir: Directory with pre-generated random levels
        """
        print("🌱 Initializing population from level libraries...")
        
        # Calculate target numbers for each source (aim for equal distribution)
        target_per_source = self.cfg.population_size // 3
        
        # Load levels from each source
        all_levels = []
        
        # 1. Load captured levels
        captured_levels = self._load_levels_from_dir(
            captured_dir, 
            pattern="level_*.txt",
            max_count=target_per_source,
            source_name="captured"
        )
        all_levels.extend(captured_levels)
        
        # 2. Load constructive levels
        constructive_levels = self._load_levels_from_dir(
            constructive_dir,
            pattern="constructive_*.txt", 
            max_count=target_per_source,
            source_name="constructive"
        )
        all_levels.extend(constructive_levels)
        
        # 3. Load random levels
        random_levels = self._load_levels_from_dir(
            random_dir,
            pattern="random_*.txt",
            max_count=target_per_source,
            source_name="random"
        )
        all_levels.extend(random_levels)
        
        # If we don't have enough levels, fill with newly generated random ones
        if len(all_levels) < self.cfg.population_size:
            shortage = self.cfg.population_size - len(all_levels)
            print(f"WARNING: Only loaded {len(all_levels)} levels, generating {shortage} more random levels...")
            for _ in tqdm(range(shortage), desc="Generating additional random levels"):
                all_levels.append(self._generate_random_level())
        
        # Shuffle to mix sources
        random.shuffle(all_levels)
        
        # Limit to population size if we loaded too many
        all_levels = all_levels[:self.cfg.population_size]
        
        print(f"[STATS] Total levels loaded: {len(all_levels)}")
        print("Computing fitness for initial population (all pre-verified as solvable)...")
        
        # All pre-generated levels are guaranteed solvable, so skip expensive solvability checks
        # Just compute fitness metrics and add directly to feasible population
        for level in tqdm(all_levels, desc="Computing fitness"):
            level_rows = grid_to_lines(level)
            fitness = self._compute_fitness(level_rows)
            self.feasible_pop.append((level, fitness))
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_level = level.copy()
        
        print(f"[OK] Initial population: {len(self.feasible_pop)} feasible levels ready")
    
    def _load_levels_from_dir(
        self,
        directory: str,
        pattern: str = "*.txt",
        max_count: int = None,
        source_name: str = "levels"
    ) -> List[np.ndarray]:
        """Load levels from a directory."""
        levels = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"WARNING: Directory not found: {directory}")
            return levels
        
        # Find all matching files
        level_files = list(dir_path.glob(pattern))
        
        if not level_files:
            print(f"WARNING: No {source_name} levels found in {directory}")
            return levels
        
        # Sample if we have too many
        if max_count and len(level_files) > max_count:
            level_files = random.sample(level_files, max_count)
        
        print(f"[DIR] Loading {len(level_files)} {source_name} levels from {directory}/")
        
        # Load each file
        for level_file in level_files:
            try:
                level = self._load_level_from_file(str(level_file))
                levels.append(level)
            except Exception as e:
                print(f"WARNING: Could not load {level_file.name}: {e}")
        
        return levels
    
    def _load_level_from_file(self, filepath: str) -> np.ndarray:
        """Load a level from a text file and convert to numpy grid."""
        with open(filepath, 'r') as f:
            rows = [line.strip() for line in f if line.strip()]
        
        # Convert to numpy grid
        grid = np.array([list(row) for row in rows])
        
        # Ensure correct dimensions (pad or crop if needed)
        if grid.shape[0] != self.cfg.height or grid.shape[1] != self.cfg.width:
            # Create new grid with correct dimensions
            new_grid = np.full((self.cfg.height, self.cfg.width), 
                              self.tiles['empty'], dtype=grid.dtype)
            
            # Copy what fits
            h = min(grid.shape[0], self.cfg.height)
            w = min(grid.shape[1], self.cfg.width)
            new_grid[:h, :w] = grid[:h, :w]
            
            grid = new_grid
        
        return grid
    
    def _generate_random_level(self) -> np.ndarray:
        """Generate level using random method."""
        gen_cfg = GenConfig(
            width=self.cfg.width,
            height=self.cfg.height,
            seed=None,
            platform_density=0.7,
            max_segments_per_row=3,
            segment_len_min=2,
            segment_len_max=8,
            ground_rows=2,
            ladder_probability=0.03,
            min_ladder_height=3,
            config_path='configs/spelunky.json'
        )
        
        grid = generate_random_level(gen_cfg)
        
        # Add start/goal if not present
        if self.tiles['start'] not in grid:
            grid[self.cfg.height - 4, 2] = self.tiles['start']
        if self.tiles['goal'] not in grid:
            grid[4, self.cfg.width - 3] = self.tiles['goal']
        
        return grid
    
    def _evaluate_and_add(self, level: np.ndarray):
        """Evaluate level and add to appropriate population."""
        level_rows = grid_to_lines(level)
        
        # Check solvability with timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Solvability check timeout")
        
        solvable = False
        try:
            # Set 5 second timeout for solvability check
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            solvable, info = is_level_solvable(
                level_rows,
                self.physics_config,
                sub_optimal=0
            )
            
            signal.alarm(0)  # Cancel alarm
        except (TimeoutError, Exception):
            signal.alarm(0)  # Cancel alarm
            solvable = False
            info = {}
        
        if solvable:
            fitness = self._compute_fitness(level_rows)
            self.feasible_pop.append((level, fitness))
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_level = level.copy()
        else:
            # Infeasible fitness: encourage platform connectivity
            fitness = self._infeasible_fitness(level_rows)
            self.infeasible_pop.append((level, fitness))
    
    def _compute_fitness(self, level_rows: List[str]) -> float:
        """
        Fitness for FEASIBLE levels: similarity to target metrics.
        Higher = better (closer to target).
        """
        try:
            # Extract structural metrics
            G, id2seg = build_segment_graph(level_rows, self.physics)
            metrics = structural_metrics(G, id2seg, max_len_for_style=None)
            
            # Compute distance to target for each metric
            distance = 0.0
            metric_keys = ['branching', 'linearity', 'dead_end_rate', 
                          'loop_complexity', 'room_count']
            
            for key in metric_keys:
                if key in self.target and key in metrics:
                    target_val = self.target[key]
                    actual_val = metrics[key]
                    
                    # Normalized squared error
                    if target_val != 0:
                        error = ((actual_val - target_val) / target_val) ** 2
                    else:
                        error = (actual_val - target_val) ** 2
                    
                    distance += error
            
            # Convert distance to fitness (closer = higher fitness)
            fitness = 1.0 / (1.0 + distance)
            
            return fitness
            
        except Exception as e:
            # If metric extraction fails, penalize
            return 0.1
    
    def _infeasible_fitness(self, level_rows: List[str]) -> float:
        """
        Fitness for INFEASIBLE levels: heuristic based on structure.
        Encourages platform connectivity and complexity.
        """
        try:
            G, id2seg = build_segment_graph(level_rows, self.physics)
            
            # Reward more platforms and connections
            num_platforms = G.number_of_nodes()
            num_connections = G.number_of_edges()
            
            # Normalize
            fitness = (num_platforms / 50.0) + (num_connections / 100.0)
            return min(fitness, 0.5)  # Cap below feasible fitness
            
        except:
            return 0.01
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Crossover: combine two parents to create offspring.
        Uses horizontal or vertical split.
        """
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy()
        
        child = parent1.copy()
        
        if random.random() < 0.5:
            # Horizontal crossover
            cut = random.randint(1, self.cfg.height - 1)
            child[cut:, :] = parent2[cut:, :]
        else:
            # Vertical crossover
            cut = random.randint(1, self.cfg.width - 1)
            child[:, cut:] = parent2[:, cut:]
        
        return child
    
    def _mutate(self, level: np.ndarray) -> np.ndarray:
        """
        Mutation: randomly modify tiles.
        """
        mutated = level.copy()
        
        for i in range(self.cfg.height):
            for j in range(self.cfg.width):
                if random.random() < self.cfg.mutation_rate:
                    # Skip start/goal/borders
                    if (mutated[i, j] in [self.tiles['start'], self.tiles['goal']] or
                        i == 0 or i >= self.cfg.height - 2 or
                        j == 0 or j >= self.cfg.width - 1):
                        continue
                    
                    # Randomly choose: empty or platform
                    mutated[i, j] = random.choice([
                        self.tiles['empty'],
                        self.tiles['platform'],
                        self.tiles['platform']  # Bias toward platforms
                    ])
        
        return mutated
    
    def _tournament_selection(self, population: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Select individual using tournament selection."""
        if not population:
            return self._generate_random_level()
        
        tournament = random.sample(population, 
                                  min(self.cfg.tournament_size, len(population)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0].copy()
    
    def evolve(self):
        """Run FI-2POP evolution for specified generations."""
        self.initialize_population()
        
        print(f"\n🧬 Starting FI-2POP evolution for {self.cfg.max_generations} generations...")
        
        # Track stagnation for adaptive mutation
        last_best_fitness = -float('inf')
        stagnation_counter = 0
        
        for gen in range(self.cfg.max_generations):
            # Sort populations by fitness
            self.feasible_pop.sort(key=lambda x: x[1], reverse=True)
            self.infeasible_pop.sort(key=lambda x: x[1], reverse=True)
            
            # Adaptive mutation: increase if stuck
            current_best = self.feasible_pop[0][1] if self.feasible_pop else 0
            if abs(current_best - last_best_fitness) < 0.001:
                stagnation_counter += 1
                # Temporarily boost mutation
                if stagnation_counter > 3:
                    old_mutation = self.cfg.mutation_rate
                    self.cfg.mutation_rate = min(0.3, old_mutation * 1.5)
            else:
                stagnation_counter = 0
                # Reset to normal
            last_best_fitness = current_best
            
            # Generate offspring
            offspring = []
            
            # Primarily breed from feasible population
            if len(self.feasible_pop) >= 2:
                for _ in range(self.cfg.population_size // 2):
                    parent1 = self._tournament_selection(self.feasible_pop[:20])
                    parent2 = self._tournament_selection(self.feasible_pop[:20])
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    offspring.append(child)
            else:
                # Not enough feasible, use infeasible
                for _ in range(self.cfg.population_size // 2):
                    if self.infeasible_pop:
                        parent = self._tournament_selection(self.infeasible_pop[:20])
                        child = self._mutate(parent)
                        offspring.append(child)
                    else:
                        offspring.append(self._generate_random_level())
            
            # Also try to repair infeasible with higher mutation
            for _ in range(self.cfg.population_size // 2):
                if self.infeasible_pop:
                    parent = self._tournament_selection(self.infeasible_pop[:10])
                    child = parent.copy()
                    # Apply more aggressive mutation for repair
                    for _ in range(3):
                        child = self._mutate(child)
                    offspring.append(child)
                else:
                    offspring.append(self._generate_random_level())
            
            # DIVERSITY INJECTION: Add fresh random levels every 5 generations
            if (gen + 1) % 5 == 0:
                num_fresh = self.cfg.population_size // 10  # 10% fresh blood
                for _ in range(num_fresh):
                    offspring.append(self._generate_random_level())
            
            # Evaluate offspring
            for child in tqdm(offspring, desc=f"Gen {gen+1} eval", leave=False):
                self._evaluate_and_add(child)
            
            # Elitism: keep best from feasible
            elite_count = min(self.cfg.elite_size, len(self.feasible_pop))
            
            # Trim populations
            self.feasible_pop = self.feasible_pop[:self.cfg.population_size]
            self.infeasible_pop = self.infeasible_pop[:self.cfg.population_size]
            
            # Record history
            if self.feasible_pop:
                avg_fit = np.mean([f for _, f in self.feasible_pop])
                best_fit = self.feasible_pop[0][1]
                
                # Compute metric distance for best
                best_level_rows = grid_to_lines(self.feasible_pop[0][0])
                try:
                    G, id2seg = build_segment_graph(best_level_rows, self.physics)
                    metrics = structural_metrics(G, id2seg)
                    metric_dist = sum([
                        abs(metrics.get(k, 0) - self.target.get(k, 0))
                        for k in ['branching', 'linearity', 'dead_end_rate']
                    ])
                except:
                    metric_dist = 999
            else:
                avg_fit = 0
                best_fit = 0
                metric_dist = 999
            
            self.history['gen'].append(gen)
            self.history['best_fitness'].append(best_fit)
            self.history['avg_fitness'].append(avg_fit)
            self.history['feasible_count'].append(len(self.feasible_pop))
            self.history['metric_distance'].append(metric_dist)
            
            # Progress report
            if (gen + 1) % 10 == 0:
                print(f"\nGen {gen + 1}/{self.cfg.max_generations}:")
                print(f"  Best fitness: {best_fit:.4f}")
                print(f"  Avg fitness: {avg_fit:.4f}")
                print(f"  Feasible: {len(self.feasible_pop)}")
                print(f"  Metric distance: {metric_dist:.4f}")
        
        return self.best_level, self.best_fitness
    
    def get_best_level(self) -> Optional[np.ndarray]:
        """Return the best level found."""
        if self.best_level is not None:
            return self.best_level
        elif self.feasible_pop:
            return self.feasible_pop[0][0]
        else:
            return None
    
    def save_best(self, output_path: str):
        """Save the best level to file."""
        best = self.get_best_level()
        if best is None:
            print("WARNING: No feasible level found!")
            return
        
        with open(output_path, 'w') as f:
            for row in grid_to_lines(best):
                f.write(row + '\n')
        
        print(f"💾 Saved best level to {output_path}")
        
        # Also print metrics
        best_rows = grid_to_lines(best)
        try:
            G, id2seg = build_segment_graph(best_rows, self.physics)
            metrics = structural_metrics(G, id2seg)
            
            print("\n[STATS] Best Level Metrics:")
            for key in ['branching', 'linearity', 'dead_end_rate', 'loop_complexity', 'room_count']:
                target_val = self.target.get(key, 'N/A')
                actual_val = metrics.get(key, 'N/A')
                print(f"  {key:20s}: target={target_val:6.3f}  actual={actual_val:6.3f}")
        except Exception as e:
            print(f"WARNING: Could not extract metrics: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FI-2POP Generator for Style-Aware Level Generation")
    parser.add_argument('--config', type=str, default='configs/spelunky.json',
                       help='Path to physics config')
    parser.add_argument('--target-metrics', type=str, required=True,
                       help='JSON file with target metrics')
    parser.add_argument('--output', type=str, default='fi2pop_generated.txt',
                       help='Output file path')
    parser.add_argument('--generations', type=int, default=200,
                       help='Number of generations')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        physics_config = json.load(f)
    
    # Load target metrics
    with open(args.target_metrics, 'r') as f:
        target_metrics = json.load(f)
    
    # Create generator config
    cfg = FI2POPConfig(
        width=physics_config.get('width', 32),
        height=physics_config.get('height', 32),
        population_size=args.population,
        max_generations=args.generations
    )
    
    # Run generator
    generator = FI2POPGenerator(target_metrics, physics_config, cfg)
    best_level, best_fitness = generator.evolve()
    
    # Save result
    generator.save_best(args.output)
    
    print(f"\n[OK] Evolution complete! Best fitness: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
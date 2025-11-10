"""
Master script to prepare all initial population levels for FI-2POP.
Generates random and constructive levels with solvability verification.
Constructive levels include variation in step parameters for diversity.
"""

from generate_random_levels import generate_and_save_random_levels
from generate_constructive_levels import generate_and_save_constructive_levels

def prepare_all_levels(
    num_random: int = 100,
    num_constructive: int = 100,
    width: int = 40,
    height: int = 32,
    max_attempts: int = 10,
    timeout: int = 5,
    platform_density_range: tuple = (0.1, 0.7),
    step_length_range: tuple = (3, 8),
    step_vertical_range: tuple = (2, 4),
    step_overlap_range: tuple = (0, 2)
):
    """
    Generate both random and constructive level libraries.
    
    Args:
        num_random: Number of random solvable levels to generate
        num_constructive: Number of constructive levels to generate
        width: Level width
        height: Level height
        max_attempts: Max attempts per level before moving on
        timeout: Timeout in seconds for solvability check
        platform_density_range: (min, max) for random platform density variation
        step_length_range: (min, max) for constructive step length variation
        step_vertical_range: (min, max) for constructive vertical spacing variation
        step_overlap_range: (min, max) for constructive overlap variation
    """
    
    print("=" * 60)
    print("🎮 PREPARING INITIAL POPULATION FOR FI-2POP")
    print("=" * 60)
    print(f"Target: {num_random} random + {num_constructive} constructive")
    print(f"Dimensions: {width}x{height}")
    print(f"Solvability timeout: {timeout}s")
    print(f"Max attempts per level: {max_attempts}")
    print(f"Random variation:")
    print(f"  Platform density: {platform_density_range}")
    print(f"Constructive variation:")
    print(f"  Step length: {step_length_range}")
    print(f"  Step vertical: {step_vertical_range}")
    print(f"  Step overlap: {step_overlap_range}")
    print()
    
    # Generate random levels
    print("STEP 1: Random Levels (with solvability verification and variation)")
    print("-" * 60)
    random_count = generate_and_save_random_levels(
        output_dir="random_levels",
        num_levels=num_random,
        width=width,
        height=height,
        max_attempts_per_level=max_attempts,
        timeout=timeout,
        platform_density_range=platform_density_range
    )
    
    print("\n" + "=" * 60)
    
    # Generate constructive levels
    print("\nSTEP 2: Constructive Levels (with solvability verification and variation)")
    print("-" * 60)
    constructive_count = generate_and_save_constructive_levels(
        output_dir="constructive_levels",
        num_levels=num_constructive,
        config_path="configs/spelunky.json",
        width=width,
        height=height,
        max_attempts_per_level=max_attempts,
        timeout=timeout,
        verify_solvability=True,
        step_length_range=step_length_range,
        step_vertical_range=step_vertical_range,
        step_overlap_range=step_overlap_range
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Random levels:        {random_count}/{num_random} (solvable)")
    print(f"Constructive levels:  {constructive_count}/{num_constructive} (verified solvable)")
    print(f"Total generated:      {random_count + constructive_count}")
    print()
    print("📁 Level libraries:")
    print("   - random_levels/")
    print("   - constructive_levels/")
    print("   - captured_levels/ (existing)")
    print()
    print("✅ Ready to run FI-2POP!")
    print("   All levels are verified solvable!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare all initial population levels for FI-2POP"
    )
    parser.add_argument('--random', type=int, default=100,
                       help='Number of random levels to generate')
    parser.add_argument('--constructive', type=int, default=100,
                       help='Number of constructive levels to generate')
    parser.add_argument('--width', type=int, default=40,
                       help='Level width')
    parser.add_argument('--height', type=int, default=32,
                       help='Level height')
    parser.add_argument('--max-attempts', type=int, default=10,
                       help='Max attempts per level before moving on')
    parser.add_argument('--timeout', type=int, default=5,
                       help='Timeout in seconds for solvability check')
    
    # Random variation parameters
    parser.add_argument('--platform-density-min', type=float, default=0.1,
                       help='Minimum platform density for random levels')
    parser.add_argument('--platform-density-max', type=float, default=0.7,
                       help='Maximum platform density for random levels (max 0.7)')
    
    # Constructive variation parameters
    parser.add_argument('--step-length-min', type=int, default=3,
                       help='Minimum step length for constructive levels')
    parser.add_argument('--step-length-max', type=int, default=8,
                       help='Maximum step length for constructive levels')
    parser.add_argument('--step-vertical-min', type=int, default=2,
                       help='Minimum vertical spacing for constructive levels')
    parser.add_argument('--step-vertical-max', type=int, default=4,
                       help='Maximum vertical spacing for constructive levels')
    parser.add_argument('--step-overlap-min', type=int, default=0,
                       help='Minimum overlap for constructive levels')
    parser.add_argument('--step-overlap-max', type=int, default=2,
                       help='Maximum overlap for constructive levels')
    
    args = parser.parse_args()
    
    # Clamp platform density max to 0.7
    max_density = min(args.platform_density_max, 0.7)
    
    prepare_all_levels(
        num_random=args.random,
        num_constructive=args.constructive,
        width=args.width,
        height=args.height,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        platform_density_range=(args.platform_density_min, max_density),
        step_length_range=(args.step_length_min, args.step_length_max),
        step_vertical_range=(args.step_vertical_min, args.step_vertical_max),
        step_overlap_range=(args.step_overlap_min, args.step_overlap_max)
    )

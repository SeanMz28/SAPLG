#!/usr/bin/env python3
"""
Check the status of FI-2POP level libraries and system readiness.
"""

from pathlib import Path
import json

def count_files(directory, pattern="*.txt"):
    """Count files matching pattern in directory."""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))

def check_file(filepath):
    """Check if file exists."""
    return Path(filepath).exists()

def main():
    print("=" * 70)
    print("FI-2POP SYSTEM STATUS")
    print("=" * 70)
    
    # Check level libraries
    print("\n[DIR] LEVEL LIBRARIES:")
    print("-" * 70)
    
    libraries = {
        'captured_levels': ('level_*.txt', 'Real Spelunky levels'),
        'constructive_levels': ('constructive_*.txt', 'Pre-generated constructive'),
        'random_levels': ('random_*.txt', 'Pre-generated random')
    }
    
    all_ready = True
    total_levels = 0
    
    for dir_name, (pattern, description) in libraries.items():
        count = count_files(dir_name, pattern)
        total_levels += count
        status = "[OK]" if count > 0 else "[FAIL]"
        print(f"{status} {dir_name:25s} {count:4d} files  ({description})")
        if count == 0:
            all_ready = False
    
    print("-" * 70)
    print(f"   Total available levels: {total_levels}")
    
    # Check configuration files
    print("\n⚙️  CONFIGURATION FILES:")
    print("-" * 70)
    
    configs = {
        'configs/spelunky.json': 'Physics configuration',
        'spelunky_metrics_summary.json': 'Target metrics'
    }
    
    for filepath, description in configs.items():
        exists = check_file(filepath)
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {filepath:35s} ({description})")
        if not exists:
            all_ready = False
    
    # Check if generated_levels directory exists
    print("\n📂 OUTPUT DIRECTORY:")
    print("-" * 70)
    
    gen_dir = Path('generated_levels')
    if gen_dir.exists():
        gen_count = count_files('generated_levels')
        print(f"[OK] generated_levels/              {gen_count} generated levels")
    else:
        print(f"WARNING:  generated_levels/              (will be created automatically)")
    
    # Check core scripts
    print("\n CORE SCRIPTS:")
    print("-" * 70)
    
    scripts = [
        'prepare_initial_population.py',
        'fi2pop_generator.py',
        'run_fi2pop.py',
        'structural_features.py',
        'solvability.py'
    ]
    
    for script in scripts:
        exists = check_file(script)
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {script}")
        if not exists:
            all_ready = False
    
    # Summary and next steps
    print("\n" + "=" * 70)
    
    if all_ready:
        print("[OK] SYSTEM READY!")
        print("=" * 70)
        print("\nYou can now run FI-2POP:")
        print("  python run_fi2pop.py")
        print("\nOr run a quick test first:")
        print("  python test_captured_in_fi2pop.py")
        
    else:
        print("WARNING:  SETUP REQUIRED")
        print("=" * 70)
        
        # Determine what's missing
        captured = count_files('captured_levels', 'level_*.txt')
        constructive = count_files('constructive_levels', 'constructive_*.txt')
        random = count_files('random_levels', 'random_*.txt')
        
        if captured > 0 and (constructive == 0 or random == 0):
            print("\nMissing level libraries. Run:")
            print("  python prepare_initial_population.py")
            print("\nThis will generate random and constructive level libraries.")
            
        elif not check_file('configs/spelunky.json'):
            print("\nMissing physics configuration.")
            print("Please ensure configs/spelunky.json exists.")
            
        elif not check_file('spelunky_metrics_summary.json'):
            print("\nMissing metrics summary.")
            print("Run structural metrics extraction first.")
        
        else:
            print("\nPlease check the missing files/directories above.")
    
    print()

if __name__ == "__main__":
    main()

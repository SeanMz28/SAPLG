# evaluate.py
"""
Evaluate generated levels using structural metrics.
"""

import json
from pathlib import Path
import numpy as np
from structural_features import Physics, build_segment_graph, structural_metrics

def evaluate_generated_levels(generated_dir, config_file='configs/spelunky.json'):
    """
    Evaluate all generated levels.
    
    Returns:
        dict: Metrics for all generated levels
    """
    
    # Load physics config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    physics = Physics(
        solids=set(config['physics']['solids']),
        jumps=config['physics']['jumps']
    )
    
    generated_path = Path(generated_dir)
    level_files = list(generated_path.glob('*.txt'))
    
    print(f"[STATS] Evaluating {len(level_files)} generated levels...\n")
    
    results = {}
    valid_count = 0
    
    for level_file in level_files:
        with open(level_file, 'r') as f:
            rows = [line.strip() for line in f]
        
        try:
            # Extract features
            G, id2seg = build_segment_graph(rows, physics)
            metrics = structural_metrics(G, id2seg)
            
            results[level_file.stem] = {
                'success': True,
                **metrics
            }
            valid_count += 1
            
        except Exception as e:
            results[level_file.stem] = {
                'success': False,
                'error': str(e)
            }
    
    print(f"[OK] Valid levels: {valid_count}/{len(level_files)}")
    
    # Calculate statistics
    if valid_count > 0:
        valid_metrics = [r for r in results.values() if r['success']]
        
        print(f"\n Statistics:")
        for key in ['room_count', 'branching', 'linearity', 'dead_end_rate', 'loop_complexity']:
            values = [m[key] for m in valid_metrics]
            print(f"  {key}:")
            print(f"    Mean: {np.mean(values):.2f}")
            print(f"    Std:  {np.std(values):.2f}")
            print(f"    Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    # Save results
    output_file = generated_path / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    generated_dir = sys.argv[1] if len(sys.argv) > 1 else 'generated_levels'
    evaluate_generated_levels(generated_dir)
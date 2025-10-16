# run_structural_metrics.py
import json, argparse
from structural_features import Physics, build_segment_graph, structural_metrics

def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, help="Path to .txt grid file")
    ap.add_argument("--config", required=True, help="Path to JSON config with tiles/physics/jumps")
    args = ap.parse_args()

    # 1) load inputs
    rows = read_rows(args.level)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 2) get physics (solids + jumps) from your config
    solids = set(cfg.get("physics", {}).get("solids", []))
    jumps  = cfg.get("physics", {}).get("jumps", [])
    # make tuples for steps
    jumps = [[(int(dx), int(dy)) for dx, dy in arc] for arc in jumps]

    phys = Physics(solids=solids, jumps=jumps)

    # 3) build the platform-segment graph and compute metrics
    G, id2seg = build_segment_graph(rows, phys)
    m = structural_metrics(G, id2seg)

    # 4) print a tidy summary
    print(f"Level: {args.level}")
    print(f"Nodes (segments): {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print("Metrics:")
    for k in ["room_count","branching","linearity","dead_end_rate","loop_complexity","segment_size_variance"]:
        print(f"  {k:24s} {m[k]}")

if __name__ == "__main__":
    main()

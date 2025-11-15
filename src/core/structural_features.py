from __future__ import annotations
"""
Structural feature extraction for 2D tile platformer levels.

Key ideas
---------
• Represent each *standable platform* as a graph node (a maximal horizontal run
  of solid tiles with AIR directly above – where the avatar's feet stand).
• Add undirected edges for feasible movement between platforms via:
  - Walking off an edge and *falling* to a lower platform.
  - Following a discrete *jump arc* (sequence of (dx, dy) in tile units).
• Compute graph-level structural metrics used by the style-aware generator.

Notable implementation choices
------------------------------
• We place each node on the AIR row above its solid run (y-1). This matches the
  position the avatar occupies while standing.
• Falls are only possible when starting from AIR. We therefore test walk-offs
  from the AIR cells just *beyond* each platform edge (x0-1, x1+1) and also from
  any AIR cell above the platform that has a gap immediately below the AIR row.
• Jumps are mirrored left/right automatically so configs only need to define
  rightward arcs if desired.

API
---
- extract_platform_segments(level_rows, solids, ignore_frame=True) -> list[Segment]
- build_segment_graph(level_rows, physics, mirror_arcs=True, enable_falls=True,
                      subsample=1) -> (nx.Graph, dict[int, Segment])
- structural_metrics(G, id2seg) -> dict[str, float]

Dependencies: networkx, numpy
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import networkx as nx


@dataclass
class Physics:
    solids: Set[str]
    # Each arc is a list of (dx, dy) steps in tile units, dy < 0 means upward
    jumps: List[List[Tuple[int, int]]]


@dataclass
class Segment:
    id: int
    y: int   # AIR row where avatar stands on this platform (row above the solids)
    x0: int  # inclusive solid run start (in ground row)
    x1: int  # inclusive solid run end   (in ground row)
    length: int


# ------------------------- helpers -------------------------

def _is_inside(grid: List[str], x: int, y: int) -> bool:
    H, W = len(grid), len(grid[0])
    return 0 <= x < W and 0 <= y < H


def extract_platform_segments(
    level_rows: List[str],
    solids: Set[str],
    ignore_frame: bool = True,
) -> List[Segment]:
    """Find maximal horizontal runs of solid with AIR directly above.

    If ignore_frame is True, skip runs that touch the outer frame to avoid
    border/wall platforms dominating statistics.
    """
    assert level_rows, "level_rows must be non-empty"
    H, W = len(level_rows), len(level_rows[0])
    segs: List[Segment] = []
    sid = 0

    for y in range(1, H):  # y refers to the *solid* row; we will use y-1 for AIR row
        x = 0
        while x < W:
            if level_rows[y][x] in solids and level_rows[y - 1][x] not in solids:
                # Start of a standable run
                x0 = x
                while x < W and level_rows[y][x] in solids and level_rows[y - 1][x] not in solids:
                    x += 1
                x1 = x - 1
                air_y = y - 1
                if ignore_frame:
                    if air_y <= 0 or air_y >= H - 1 or x0 <= 0 or x1 >= W - 1:
                        continue
                segs.append(Segment(id=sid, y=air_y, x0=x0, x1=x1, length=x1 - x0 + 1))
                sid += 1
            else:
                x += 1
    return segs


def _index_segments_by_row(segs: List[Segment]) -> Dict[int, List[Segment]]:
    by_row: Dict[int, List[Segment]] = {}
    for s in segs:
        by_row.setdefault(s.y, []).append(s)
    for row in by_row.values():
        row.sort(key=lambda s: s.x0)
    return by_row


def _segment_at(by_row: Dict[int, List[Segment]], x: int, y: int) -> Optional[int]:
    """Return the segment id whose [x0..x1] covers x on AIR row y, else None."""
    row = by_row.get(y)
    if not row:
        return None
    lo, hi = 0, len(row) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = row[mid]
        if x < s.x0:
            hi = mid - 1
        elif x > s.x1:
            lo = mid + 1
        else:
            return s.id
    return None


def _path_clear_air(
    level_rows: List[str],
    steps: List[Tuple[int, int]],
    x0: int,
    y0: int,
    solids: Set[str],
) -> Optional[Tuple[int, int]]:
    """Simulate an arc through AIR. Return landing AIR cell (x,y) if the path
    remains in air and ends with solid just below; otherwise None.
    """
    H, W = len(level_rows), len(level_rows[0])
    x, y = x0, y0
    for dx, dy in steps:
        x2, y2 = x + dx, y + dy
        if not (0 <= x2 < W and 0 <= y2 < H):
            return None
        if level_rows[y2][x2] in solids:
            return None
        x, y = x2, y2
    # landing condition: AIR at (x,y) and SOLID just below
    if level_rows[y][x] in solids:
        return None
    y_below = y + 1
    if y_below < H and level_rows[y_below][x] in solids:
        return (x, y)
    return None


# ------------------------- main graph builder -------------------------

def build_segment_graph(
    level_rows: List[str],
    physics: Physics,
    mirror_arcs: bool = True,
    enable_falls: bool = True,
    subsample: int = 1,
    ignore_frame_for_segments: bool = True,
    max_fall: Optional[int] = 6,
    allow_hole_falls: bool = False,
) -> Tuple[nx.Graph, Dict[int, Segment]]:
    """Create a platform reachability graph.

    Args
    ----
    level_rows : list of equal-length strings
    physics    : Physics(solids, jumps)
    mirror_arcs: if True, auto-add leftward mirrors of jumps
    enable_falls: if True, attempt walk-off (and optionally hole) falls
    subsample  : take every Nth start x on a segment when simulating arcs (speed)
    ignore_frame_for_segments : if True, exclude frame-touching segments
    max_fall   : maximum tiles you allow to drop during a fall (None = unlimited)
    allow_hole_falls : also fall from holes along the platform (not just edges)
    """
    assert level_rows, "level_rows must be non-empty"
    H, W = len(level_rows), len(level_rows[0])

    segs = extract_platform_segments(level_rows, physics.solids, ignore_frame_for_segments)
    by_row = _index_segments_by_row(segs)
    id2seg = {s.id: s for s in segs}

    G = nx.Graph()
    for s in segs:
        G.add_node(s.id, length=s.length)

    # Prepare arcs (mirror for left jumps automatically)
    all_arcs = list(physics.jumps)
    if mirror_arcs:
        all_arcs += [[(-dx, dy) for (dx, dy) in arc] for arc in physics.jumps]

    def try_fall_from(x_start: int, y_start: int, sid_from: int) -> None:
        # Start must be in AIR
        if not _is_inside(level_rows, x_start, y_start):
            return
        if level_rows[y_start][x_start] in physics.solids:
            return
        y = y_start
        fallen = 0
        while y + 1 < H and level_rows[y + 1][x_start] not in physics.solids:
            y += 1
            fallen += 1
            if max_fall is not None and fallen > max_fall:
                return  # too far to safely drop; skip edge
        sid_land = _segment_at(by_row, x_start, y)
        if sid_land is not None and sid_land != sid_from:
            G.add_edge(sid_from, sid_land)

    for s in segs:
        # ---- FALLS ----
        if enable_falls:
            # (1) Walk-off from each edge
            left_air_x = s.x0 - 1
            right_air_x = s.x1 + 1
            if _is_inside(level_rows, left_air_x, s.y):
                try_fall_from(left_air_x, s.y, s.id)
            if _is_inside(level_rows, right_air_x, s.y):
                try_fall_from(right_air_x, s.y, s.id)

            # (2) Optional: vertical-gap falls from above the platform
            if allow_hole_falls:
                for x in range(s.x0, s.x1 + 1, max(1, subsample)):
                    if s.y + 1 < H and level_rows[s.y + 1][x] not in physics.solids:
                        try_fall_from(x, s.y, s.id)

        # ---- JUMPS ----
        for x in range(s.x0, s.x1 + 1, max(1, subsample)):
            start_x, start_y = x, s.y  # AIR cell where avatar stands
            for arc in all_arcs:
                land = _path_clear_air(level_rows, arc, start_x, start_y, physics.solids)
                if land is None:
                    continue
                sid = _segment_at(by_row, land[0], land[1])
                if sid is not None and sid != s.id:
                    G.add_edge(s.id, sid)

    return G, id2seg


# ------------------------- metrics -------------------------

def structural_metrics(
    G: nx.Graph,
    id2seg: Dict[int, Segment],
    max_len_for_style: Optional[int] = None,
) -> Dict[str, float]:
    """Compute structural metrics from a platform reachability graph.

    If max_len_for_style is set, metrics are computed on a filtered view that
    removes nodes with segment length > max_len_for_style (e.g., long base).
    """
    if max_len_for_style is not None:
        keep_nodes = [n for n in G.nodes() if id2seg[n].length <= max_len_for_style]
        G = G.subgraph(keep_nodes).copy()
        if G.number_of_nodes() == 0:
            return dict(
                room_count=0.0, branching=0.0, linearity=0.0,
                dead_end_rate=0.0, loop_complexity=0.0, segment_size_variance=0.0,
            )
        # recompute variance on kept nodes only
        seg_sizes = [id2seg[n].length for n in G.nodes()]
        seg_var = float(np.var(seg_sizes)) if seg_sizes else 0.0
    else:
        seg_sizes = [id2seg[n].length for n in G.nodes()]
        seg_var = float(np.var(seg_sizes)) if seg_sizes else 0.0

    if G.number_of_nodes() == 0:
        return dict(
            room_count=0.0,
            branching=0.0,
            linearity=0.0,
            dead_end_rate=0.0,
            loop_complexity=0.0,
            segment_size_variance=0.0,
        )

    comps = list(nx.connected_components(G))
    room_count = float(len(comps))
    branching = float(np.mean([deg for _, deg in G.degree()])) if G.number_of_nodes() > 0 else 0.0
    dead_end_rate = float(sum(1 for _, deg in G.degree() if deg == 1) / G.number_of_nodes())
    loop_complexity = float(G.number_of_edges() - G.number_of_nodes() + len(comps))

    # Linearity: approximate with diameter over components (undirected)
    def comp_diameter(nodes: Set[int]) -> int:
        Hsub = G.subgraph(nodes).copy()
        try:
            return nx.diameter(Hsub)
        except nx.NetworkXError:
            diam = 0
            for u in Hsub.nodes():
                lengths = nx.single_source_shortest_path_length(Hsub, u)
                diam = max(diam, max(lengths.values()))
            return diam

    diam = max(comp_diameter(c) for c in comps)
    linearity = float((diam + 1) / max(1, G.number_of_nodes()))

    return dict(
        room_count=float(room_count),
        branching=float(branching),
        linearity=float(linearity),
        dead_end_rate=float(dead_end_rate),
        loop_complexity=float(loop_complexity),
        segment_size_variance=float(seg_var),
    )

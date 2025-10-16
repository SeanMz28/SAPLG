# test_level.py
# (Drop-in updated to support explicit Start and Goal)

import pathfinding

def makeIsSolid(solids):
    def isSolid(tile):
        return tile in solids
    return isSolid

def makeGetNeighbors(jumps, levelStr, visited, isSolid):
    maxX = len(levelStr[0]) - 1
    maxY = len(levelStr) - 1
    # Convert absolute jump frames to per-frame deltas
    jumpDiffs = []
    for jump in jumps:
        jumpDiff = [jump[0]]
        for ii in range(1, len(jump)):
            jumpDiff.append((jump[ii][0] - jump[ii-1][0], jump[ii][1] - jump[ii-1][1]))
        jumpDiffs.append(jumpDiff)
    jumps = jumpDiffs

    def getNeighbors(pos):
        # NOTE: pathfinding.astar_shortest_path passes a structure to this callback
        # that includes distance and the underlying state. This function expects that shape.
        dist = pos[0] - pos[2]
        pos = pos[1]  # now pos is the state tuple: (x, y, jump_idx or -1, frame_idx?, dir?)
        visited.add((pos[0], pos[1]))
        below = (pos[0], pos[1] + 1)
        neighbors = []

        if below[1] > maxY:
            return []

        if pos[2] != -1:
            ii = pos[3] + 1
            jump = pos[2]
            if ii < len(jumps[jump]):
                if not (pos[0] + pos[4] * jumps[jump][ii][0] > maxX or
                        pos[0] + pos[4] * jumps[jump][ii][0] < 0 or
                        pos[1] + jumps[jump][ii][1] < 0) and \
                   not isSolid(levelStr[pos[1] + jumps[jump][ii][1]][pos[0] + pos[4] * jumps[jump][ii][0]]):
                    neighbors.append([dist + 1, (pos[0] + pos[4] * jumps[jump][ii][0],
                                                 pos[1] + jumps[jump][ii][1], jump, ii, pos[4])])
                if pos[1] + jumps[jump][ii][1] < 0 and \
                   not isSolid(levelStr[pos[1] + jumps[jump][ii][1]][pos[0] + pos[4] * jumps[jump][ii][0]]):
                    neighbors.append([dist + 1, (pos[0] + pos[4] * jumps[jump][ii][0],
                                                 0, jump, ii, pos[4])])

        if isSolid(levelStr[below[1]][below[0]]):
            if pos[0] + 1 <= maxX and not isSolid(levelStr[pos[1]][pos[0] + 1]):
                neighbors.append([dist + 1, (pos[0] + 1, pos[1], -1)])
            if pos[0] - 1 >= 0 and not isSolid(levelStr[pos[1]][pos[0] - 1]):
                neighbors.append([dist + 1, (pos[0] - 1, pos[1], -1)])

            for jump in range(len(jumps)):
                ii = 0
                if not (pos[0] + jumps[jump][ii][0] > maxX or pos[1] < 0) and \
                   not isSolid(levelStr[pos[1] + jumps[jump][ii][1]][pos[0] + jumps[jump][ii][0]]):
                    neighbors.append([dist + ii + 1, (pos[0] + jumps[jump][ii][0],
                                                      pos[1] + jumps[jump][ii][1], jump, ii, 1)])

                if not (pos[0] - jumps[jump][ii][0] < 0 or pos[1] < 0) and \
                   not isSolid(levelStr[pos[1] + jumps[jump][ii][1]][pos[0] - jumps[jump][ii][0]]):
                    neighbors.append([dist + ii + 1, (pos[0] - jumps[jump][ii][0],
                                                      pos[1] + jumps[jump][ii][1], jump, ii, -1)])

        else:
            neighbors.append([dist + 1, (pos[0], pos[1] + 1, -1)])
            if pos[1] + 1 <= maxY:
                if pos[0] + 1 <= maxX and not isSolid(levelStr[pos[1] + 1][pos[0] + 1]):
                    neighbors.append([dist + 1.4, (pos[0] + 1, pos[1] + 1, -1)])
                if pos[0] - 1 >= 0 and not isSolid(levelStr[pos[1] + 1][pos[0] - 1]):
                    neighbors.append([dist + 1.4, (pos[0] - 1, pos[1] + 1, -1)])
            if pos[1] + 2 <= maxY:
                if pos[0] + 1 <= maxX and not isSolid(levelStr[pos[1] + 2][pos[0] + 1]):
                    neighbors.append([dist + 2, (pos[0] + 1, pos[1] + 2, -1)])
                if pos[0] - 1 >= 0 and not isSolid(levelStr[pos[1] + 2][pos[0] - 1]):
                    neighbors.append([dist + 2, (pos[0] - 1, pos[1] + 2, -1)])
        return neighbors

    return getNeighbors

def findPaths(subOptimal, solids, jumps, levelStr, start=None, goal_xy=None):
    """
    subOptimal: int band for near-optimal paths
    solids: set/list of solid tile chars
    jumps: list of jump arcs (as in your config)
    levelStr: list[str] level rows
    start: (x, y) tile coords; default = (2,2)
    goal_xy: (x, y) tile coords to reach; default = rightmost column
    """
    visited = set()
    isSolid = makeIsSolid(solids)
    getNeighbors = makeGetNeighbors(jumps, levelStr, visited, isSolid)

    maxX = len(levelStr[0]) - 1

    # Seed start state: (x, y, -1) means "not in a jump"
    if start is None:
        start_state = (2, 2, -1)
    else:
        start_state = (start[0], start[1], -1)

    # Goal predicate: exact goal if provided, else reach rightmost column
    if goal_xy is not None:
        gx, gy = goal_xy
        goal_fn = lambda pos: (pos[0] == gx and pos[1] == gy)
    else:
        goal_fn = lambda pos: pos[0] == maxX

    paths = pathfinding.astar_shortest_path(
        start_state,
        goal_fn,
        getNeighbors,
        subOptimal,
        lambda pos: 0  # heuristic left as zero (you had this before)
    )
    # Return only (x,y) coordinates for each step
    return [[(p[0], p[1]) for p in path] for path in paths]

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 3:
        print('Usage: {} <platformer json> <level text filename>'.format(sys.argv[0]))
        exit()

    levelFilename = sys.argv[2]
    level = []
    with open(levelFilename) as level_file:
        for line in level_file:
            level.append(line.rstrip())
    with open(sys.argv[1]) as data_file:
        platformerDescription = json.load(data_file)
    paths = findPaths(
        10,
        platformerDescription['solid'],      # keep your old file's structure for CLI compat
        platformerDescription['jumps'],
        level
    )
    for p in paths:
        print(p)

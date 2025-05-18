# Các hàm tiện ích dùng chung cho Puzzle 8

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def get_pos(x, y):
    return 30 + x * 150, 30 + y * 150

def move(state, direction):
    new_state = [row[:] for row in state]
    moves = {
        'up': (1, 0),
        'down': (-1, 0),
        'left': (0, 1),
        'right': (0, -1),
    }
    empty_pos = next((y, x) for y in range(3) for x in range(3) if state[y][x] == 0)
    y, x = empty_pos
    dy, dx = moves[direction]
    ny, nx = y + dy, x + dx
    if 0 <= ny < 3 and 0 <= nx < 3:
        new_state[y][x], new_state[ny][nx] = new_state[ny][nx], new_state[y][x]
        return new_state
    return None

def misplaced_tiles(state):
    return sum(1 for y in range(3) for x in range(3) if state[y][x] != goal_state[y][x] and state[y][x] != 0)

def manhattan_distance(state):
    distance = 0
    for y in range(3):
        for x in range(3):
            value = state[y][x]
            if value != 0:
                goal_positions = [(gy, gx) for gy in range(3) for gx in range(3) if goal_state[gy][gx] == value]
                if goal_positions:
                    goal_y, goal_x = goal_positions[0]
                    distance += abs(y - goal_y) + abs(x - goal_x)
    return distance

def heuristic(state):
    return manhattan_distance(state) + misplaced_tiles(state)

def get_neighbors(state):
    neighbors = []
    for direction in ['up', 'down', 'left', 'right']:
        neighbor = move(state, direction)
        if neighbor:
            neighbors.append(neighbor)
    return neighbors

def get_unique_neighbors(state, visited):
    neighbors = get_neighbors(state)
    return [n for n in neighbors if not any(are_states_equal(n, v) for v in visited)]

def are_states_equal(state1, state2):
    if state1 is None or state2 is None:
        return False
    for i in range(3):
        for j in range(3):
            if state1[i][j] != state2[i][j]:
                return False
    return True

def is_solvable(state):
    puzzle = [number for row in state for number in row if number != 0]
    inversions = 0
    for i in range(len(puzzle)):
        for j in range(i + 1, len(puzzle)):
            if puzzle[i] > puzzle[j]:
                inversions += 1
    return inversions % 2 == 0

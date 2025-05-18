import random
import math
from collections import deque
from queue import PriorityQueue

from utils import (
    heuristic,
    get_neighbors,
    get_unique_neighbors,
    are_states_equal,
    is_solvable
)

# BFS
def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = {str(start_state)}
    
    while queue:
        state, path = queue.popleft()
        if heuristic(state) == 0:
            return path + [state]
        
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                queue.append((neighbor, path + [state]))
    
    return [start_state]

# DFS
def dfs(start_state, max_depth=100):  # Tăng max_depth mặc định lên 50
    stack = [(start_state, [], 0)]
    visited = set()
    while stack:
        state, path, depth = stack.pop()
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        if heuristic(state) == 0:
            return path + [state]
        if depth < max_depth:
            for neighbor in get_neighbors(state):
                n_str = str(neighbor)
                if n_str not in visited:
                    stack.append((neighbor, path + [state], depth + 1))
    return []  # Trả về rỗng nếu không tìm thấy lời giải

# Iterative Deepening Search (IDS)
def ids(start_state, max_depth=50):
    def dls(state, path, depth, visited):
        if heuristic(state) == 0:
            return path + [state]
        if depth == 0:
            return None
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                result = dls(neighbor, path + [state], depth - 1, visited)
                if result:
                    return result
        return None

    for depth in range(1, max_depth + 1):
        visited = {str(start_state)}
        result = dls(start_state, [], depth, visited)
        if result:
            return result
    return [start_state]

# Greedy Best-First Search
def greedy_search(start_state):
    pq = PriorityQueue()
    pq.put((heuristic(start_state), start_state, []))
    visited = {str(start_state)}
    
    while not pq.empty():
        _, state, path = pq.get()
        if heuristic(state) == 0:
            return path + [state]
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                pq.put((heuristic(neighbor), neighbor, path + [state]))
    return [start_state]

# Hill Climbing đơn giản (với sideways move)
def simple_hill_climbing(start_state, max_iterations=1000):
    current = start_state
    path = [current]
    visited = [current]
    sideways_moves = 0
    max_sideways_moves = 5  # Cho phép tối đa 5 bước đi sideways
    
    for _ in range(max_iterations):
        neighbors = get_unique_neighbors(current, visited)
        
        # Nếu không còn lân cận nào chưa xét, kết thúc
        if not neighbors:
            break
        
        # Tính điểm heuristic cho trạng thái hiện tại
        current_value = heuristic(current)
        
        # Nếu đạt được trạng thái mục tiêu, trả về đường đi
        if current_value == 0:
            return path
        
        # Tìm lân cận tốt nhất
        neighbor_values = [(n, heuristic(n)) for n in neighbors]
        neighbor_values.sort(key=lambda x: x[1])  # Sắp xếp theo heuristic tăng dần
        
        if not neighbor_values:
            break
            
        best_neighbor, best_value = neighbor_values[0]
        
        # Nếu không tìm thấy lân cận tốt hơn
        if best_value > current_value:
            break  # Đã đạt cực đại địa phương
        
        # Cho phép sideways move (ngang bằng) với giới hạn
        if best_value == current_value:
            sideways_moves += 1
            if sideways_moves > max_sideways_moves:
                break
        else:
            sideways_moves = 0  # Reset counter khi tìm thấy lân cận tốt hơn
        
        # Di chuyển đến lân cận tốt nhất
        current = best_neighbor
        path.append(current)
        visited.append(current)
    
    return path

# Hill Climbing dốc nhất (Steepest Ascent) với sideways moves
def steepest_hill_climbing(start_state, max_iterations=1000):
    current = start_state
    path = [current]
    visited = [current]
    sideways_moves = 0
    max_sideways_moves = 5  # Cho phép tối đa 5 bước đi sideways
    
    for _ in range(max_iterations):
        neighbors = get_unique_neighbors(current, visited)
        
        # Nếu không còn lân cận nào chưa xét, kết thúc
        if not neighbors:
            break
        
        # Tính điểm heuristic cho trạng thái hiện tại
        current_value = heuristic(current)
        
        # Nếu đạt được trạng thái mục tiêu, trả về đường đi
        if current_value == 0:
            return path
        
        # Tìm tất cả lân cận tốt hơn hoặc bằng
        better_neighbors = []
        equal_neighbors = []
        
        for neighbor in neighbors:
            neighbor_value = heuristic(neighbor)
            if neighbor_value < current_value:
                better_neighbors.append((neighbor, neighbor_value))
            elif neighbor_value == current_value:
                equal_neighbors.append((neighbor, neighbor_value))
        
        # Nếu không có lân cận tốt hơn, thử sideways move
        if not better_neighbors:
            if equal_neighbors and sideways_moves < max_sideways_moves:
                # Chọn một lân cận có cùng giá trị
                sideways_moves += 1
                equal_neighbors.sort(key=lambda x: x[1])
                current = equal_neighbors[0][0]
            else:
                break  # Đã đạt cực đại địa phương và không còn sideways moves
        else:
            # Reset counter và di chuyển đến lân cận tốt nhất
            sideways_moves = 0
            better_neighbors.sort(key=lambda x: x[1])
            current = better_neighbors[0][0]
        
        path.append(current)
        visited.append(current)
    
    return path

# Hill Climbing ngẫu nhiên (Random Restart)
def random_hill_climbing(start_state, max_iterations=1000, max_restarts=10):
    best_path = None
    best_value = float('inf')
    all_visited = []  # Để tránh trùng lặp các restart
    
    # Lưu trạng thái ban đầu
    original_start = [row[:] for row in start_state]
    
    for current_restart in range(max_restarts):
        # Bắt đầu từ trạng thái ban đầu cho lần đầu tiên
        if current_restart == 0:
            current = [row[:] for row in original_start]
        else:
            # Tạo trạng thái mới bằng cách thực hiện các bước di chuyển ngẫu nhiên
            current = [row[:] for row in original_start]
            for _ in range(20):  # 20 bước di chuyển ngẫu nhiên
                neighbors = get_neighbors(current)
                if neighbors:
                    current = random.choice(neighbors)
            
            # Kiểm tra xem trạng thái này đã từng được xét chưa
            if any(are_states_equal(current, v) for v in all_visited):
                continue
        
        # Thêm vào danh sách đã xét
        all_visited.append(current)
        
        # Thực hiện hill climbing từ trạng thái bắt đầu
        path = [current]
        visited = [current]
        sideways_moves = 0
        max_sideways_moves = 5
        
        for _ in range(max_iterations // max_restarts):
            neighbors = get_unique_neighbors(current, visited)
            if not neighbors:
                break
                
            current_value = heuristic(current)
            
            if current_value == 0:
                return path  # Đã tìm thấy giải pháp
            
            # Tìm lân cận tốt nhất
            neighbor_values = [(n, heuristic(n)) for n in neighbors]
            neighbor_values.sort(key=lambda x: x[1])
            
            if not neighbor_values:
                break
                
            best_neighbor, best_value_local = neighbor_values[0]
            
            # Xử lý di chuyển
            if best_value_local < current_value:
                sideways_moves = 0
                current = best_neighbor
            elif best_value_local == current_value and sideways_moves < max_sideways_moves:
                sideways_moves += 1
                current = best_neighbor
            else:
                break  # Không thể tiến tới nữa
            
            path.append(current)
            visited.append(current)
        
        # Cập nhật đường đi tốt nhất
        final_value = heuristic(path[-1])
        if best_path is None or final_value < best_value:
            best_value = final_value
            best_path = path
        
        # Nếu đã tìm được giải pháp, trả về ngay
        if best_value == 0:
            return best_path
    
    return best_path if best_path else [start_state]

# Beam Search
def beam_search(start_state, beam_width=3, max_iterations=1000):
    beam = [(start_state, heuristic(start_state))]
    visited = {str(start_state)}
    paths = {str(start_state): [start_state]}
    
    for _ in range(max_iterations):
        candidates = []
        for state, _ in beam:
            if heuristic(state) == 0:
                return paths[str(state)]
                
            neighbors = get_neighbors(state)
            for neighbor in neighbors:
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    visited.add(neighbor_str)
                    h = heuristic(neighbor)
                    candidates.append((neighbor, h))
                    paths[neighbor_str] = paths[str(state)] + [neighbor]
        
        if not candidates:
            break
            
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]
        
    return paths[str(min(beam, key=lambda x: x[1])[0])]

# A* Search
def astar_search(start_state, max_iterations=1000):
    pq = PriorityQueue()
    g_scores = {str(start_state): 0}
    pq.put((heuristic(start_state), 0, start_state, []))
    visited = {str(start_state)}
    
    while not pq.empty():
        _, g_score, state, path = pq.get()
        if heuristic(state) == 0:
            return path + [state]
            
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            new_g = g_score + 1
            if n_str not in visited or new_g < g_scores[n_str]:
                visited.add(n_str)
                g_scores[n_str] = new_g
                f_score = new_g + heuristic(neighbor)
                pq.put((f_score, new_g, neighbor, path + [state]))
    return [start_state]

# Uniform Cost Search
def uniform_cost_search(start_state, max_iterations=1000):
    pq = PriorityQueue()
    pq.put((0, start_state, []))
    visited = {str(start_state)}
    
    while not pq.empty():
        cost, state, path = pq.get()
        if heuristic(state) == 0:
            return path + [state]
            
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                pq.put((cost + 1, neighbor, path + [state]))
    return [start_state]

# IDA* Search
def ida_star_search(start_state, max_iterations=1000):
    def search(path, g, bound):
        node = path[-1]
        f = g + heuristic(node)
        if f > bound:
            return f, None
        if heuristic(node) == 0:
            return -1, path
            
        min_bound = float('inf')
        for neighbor in get_neighbors(node):
            if neighbor not in path:
                path.append(neighbor)
                t, sol = search(path, g + 1, bound)
                if t == -1:
                    return -1, sol
                if t < min_bound:
                    min_bound = t
                path.pop()
        return min_bound, None

    bound = heuristic(start_state)
    path = [start_state]
    for _ in range(max_iterations):
        t, sol = search(path, 0, bound)
        if t == -1:
            return sol
        if t == float('inf'):
            return [start_state]
        bound = t
    return [start_state]

# Simulated Annealing
def simulated_annealing(start_state, max_iterations=1000):
    current = start_state
    path = [current]
    temperature = 1.0
    cooling_rate = 0.995
    
    for _ in range(max_iterations):
        if heuristic(current) == 0:
            return path
            
        temperature *= cooling_rate
        if temperature <= 0.01:
            break
            
        neighbors = get_neighbors(current)
        if not neighbors:
            break
            
        next_state = random.choice(neighbors)
        delta_e = heuristic(next_state) - heuristic(current)
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current = next_state
            path.append(current)
            
    return path

# Backtracking với Forward Checking cho 8-puzzle
def backtracking_search(start_state, max_depth=50):
    def is_goal(state):
        return heuristic(state) == 0

    def backtrack(state, path, visited, depth):
        if is_goal(state):
            return path + [state]
        if depth == 0:
            return None
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                result = backtrack(neighbor, path + [state], visited, depth - 1)
                if result:
                    return result
                visited.remove(n_str)
        return None

    visited = {str(start_state)}
    result = backtrack(start_state, [], visited, max_depth)
    return result if result else [start_state]

def backtracking_forward_search(start_state, max_depth=50):
    def is_goal(state):
        return heuristic(state) == 0

    def forward_check(state, visited):
        # Nếu không còn neighbor nào chưa thăm, trả về False (fail sớm)
        for neighbor in get_neighbors(state):
            if str(neighbor) not in visited:
                return True
        return False

    def backtrack(state, path, visited, depth):
        if is_goal(state):
            return path + [state]
        if depth == 0:
            return None
        for neighbor in get_neighbors(state):
            n_str = str(neighbor)
            if n_str not in visited:
                visited.add(n_str)
                # Forward checking: nếu không còn neighbor nào cho bước tiếp theo thì bỏ qua
                if not forward_check(neighbor, visited):
                    visited.remove(n_str)
                    continue
                result = backtrack(neighbor, path + [state], visited, depth - 1)
                if result:
                    return result
                visited.remove(n_str)
        return None

    visited = {str(start_state)}
    result = backtrack(start_state, [], visited, max_depth)
    return result if result else [start_state]

# Min-Conflicts cho 8-puzzle (phiên bản heuristic, tránh lặp lại 2 trạng thái trước đó)
def min_conflicts_search(start_state, max_steps=1000, tabu_size=5):
    current = [row[:] for row in start_state]
    path = [current]
    tabu_list = []  # Store recent states to avoid cycles
    
    for _ in range(max_steps):
        if heuristic(current) == 0:
            return path
            
        # Get all possible next states
        neighbors = get_neighbors(current)
        if not neighbors:
            break
            
        # Filter out states in tabu list
        valid_neighbors = [n for n in neighbors if not any(are_states_equal(n, t) for t in tabu_list)]
        if not valid_neighbors:
            valid_neighbors = neighbors  # If all neighbors are tabu, ignore tabu list
            
        # Find neighbor with minimum conflicts
        best_neighbor = min(valid_neighbors, key=lambda x: heuristic(x))
        
        # Update tabu list
        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
            
        # Move to best neighbor
        current = best_neighbor
        path.append(current)
        
        # Check if we're stuck in a cycle despite tabu list
        if len(path) >= 3 and are_states_equal(path[-1], path[-3]):
            break
            
    return path if path else [start_state]

# Genetic Algorithm cho 8-puzzle
def genetic_algorithm_search(start_state, population_size=50, generations=200, mutation_rate=0.2, max_attempts=10):
    import random
    from utils import heuristic, is_solvable, get_neighbors, are_states_equal

    def flatten(state):
        return [num for row in state for num in row]

    def unflatten(flat):
        return [flat[i*3:(i+1)*3] for i in range(3)]

    def fitness(state):
        return -heuristic(state)  # càng cao càng tốt

    def crossover(parent1, parent2):
        cut = random.randint(1, 7)
        child = parent1[:cut] + [x for x in parent2 if x not in parent1[:cut]]
        return child

    def mutate(state):
        s = state[:]
        i, j = random.sample(range(9), 2)
        s[i], s[j] = s[j], s[i]
        return s

    base = flatten(start_state)
    best_state = None
    best_score = float('-inf')

    for attempt in range(max_attempts):
        # Khởi tạo lại quần thể mỗi lần thử
        population = []
        for _ in range(population_size):
            ind = base[:]
            random.shuffle(ind)
            while not is_solvable(unflatten(ind)):
                random.shuffle(ind)
            population.append(ind)

        for _ in range(generations):
            scored = sorted([(fitness(unflatten(ind)), ind) for ind in population], reverse=True)
            if -scored[0][0] == 0:
                best_state = unflatten(scored[0][1])
                break
            # Lưu trạng thái tốt nhất hiện tại
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_state = unflatten(scored[0][1])
            selected = [ind for _, ind in scored[:population_size//2]]
            children = []
            while len(children) < population_size:
                p1, p2 = random.sample(selected, 2)
                child = crossover(p1, p2)
                if random.random() < mutation_rate:
                    child = mutate(child)
                if len(set(child)) == 9 and is_solvable(unflatten(child)):
                    children.append(child)
            population = children
        else:
            # Nếu không break (không tìm thấy lời giải), cập nhật best_state nếu cần
            scored = sorted([(fitness(unflatten(ind)), ind) for ind in population], reverse=True)
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_state = unflatten(scored[0][1])
        # Nếu đã tìm được trạng thái đích, thoát vòng lặp lớn
        if best_state is not None and heuristic(best_state) == 0:
            break

    # Tìm đường đi từ start_state đến best_state bằng BFS
    from collections import deque
    def bfs_path(start, goal):
        queue = deque([(start, [])])
        visited = {str(start)}
        while queue:
            state, path = queue.popleft()
            if are_states_equal(state, goal):
                return path + [state]
            for neighbor in get_neighbors(state):
                n_str = str(neighbor)
                if n_str not in visited:
                    visited.add(n_str)
                    queue.append((neighbor, path + [state]))
        return [start]

    return bfs_path(start_state, best_state)

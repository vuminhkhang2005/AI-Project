import pygame
import sys

# Import các thuật toán từ file Algorithms.py
from Algorithms import (
    simple_hill_climbing,
    steepest_hill_climbing,
    random_hill_climbing,
    beam_search,
    bfs,
    dfs,
    backtracking_search,
    backtracking_forward_search,
    ids,
    greedy_search,
    uniform_cost_search,
    astar_search,
    ida_star_search,
    simulated_annealing,
    min_conflicts_search,
    genetic_algorithm_search
)

# Import các hàm tiện ích từ utils.py
from utils import (
    heuristic,
    get_neighbors,
    get_unique_neighbors,
    are_states_equal,
    is_solvable,
    misplaced_tiles,
    manhattan_distance
)

# Trạng thái ban đầu
current = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]

# Cài đặt ban đầu
pygame.init()
size = width, height = 800, 600  # Tăng chiều rộng cửa sổ
screen = pygame.display.set_mode(size)
pygame.display.set_caption("8-Puzzle with Hill Climbing")
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)  # Thu nhỏ font chữ cho buttons

# Màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
LIGHT_BLUE = (100, 100, 255)

# Biến trạng thái chung
current_state = None
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
algorithm = None
running = False
solution_path = []
step_index = 0
delay = 500  # Độ trễ giữa các bước hiển thị (ms)

# Tạo vị trí các ô
def get_pos(x, y):
    return 30 + x * 150, 30 + y * 150  # Di chuyển board sang trái

# Vẽ bảng trạng thái hiện tại
def draw_state(state):
    for y in range(3):
        for x in range(3):
            value = state[y][x]
            pos_x, pos_y = get_pos(x, y)
            pygame.draw.rect(screen, WHITE if value else GREEN, (pos_x, pos_y, 140, 140))
            if value:
                text = font.render(str(value), True, BLUE)
                text_rect = text.get_rect(center=(pos_x + 70, pos_y + 70))
                screen.blit(text, text_rect)

# Di chuyển ô trống
def move(state, direction):
    new_state = [row[:] for row in state]  # Tạo bản sao trạng thái
    moves = {
        'up': (1, 0),    # Di chuyển ô trống lên (ô số đi xuống)
        'down': (-1, 0),  # Di chuyển ô trống xuống (ô số đi lên)
        'left': (0, 1),   # Di chuyển ô trống sang trái (ô số đi sang phải)
        'right': (0, -1), # Di chuyển ô trống sang phải (ô số đi sang trái)
    }
    
    # Tìm vị trí ô trống (0)
    empty_pos = next((y, x) for y in range(3) for x in range(3) if state[y][x] == 0)
    y, x = empty_pos
    
    dy, dx = moves[direction]
    ny, nx = y + dy, x + dx
    
    # Kiểm tra tính hợp lệ của vị trí mới
    if 0 <= ny < 3 and 0 <= nx < 3:
        # Hoán đổi ô trống với ô lân cận
        new_state[y][x], new_state[ny][nx] = new_state[ny][nx], new_state[y][x]
        return new_state
    return None

# Vẽ giao diện
def draw_interface():
    screen.fill(BLACK)
    
    if current_state:
        draw_state(current_state)
    
    # Chia các buttons thành 2 cột
    buttons = [
        # Cột 1
        ("simple", "Hill Climbing"),
        ("steepest", "Steepest"),
        ("random", "Random"), 
        ("beam", "Beam Search"),
        ("bfs", "BFS"),
        ("dfs", "DFS"),
        ("backtrack", "Backtrack"),
        ("backtrackf", "Backtrack Fwd"),
        # Cột 2
        ("ids", "IDS"),
        ("greedy", "Greedy"),
        ("ucs", "UCS"),
        ("astar", "A*"),
        ("idastar", "IDA*"),
        ("simann", "Sim. Annealing"),
        ("minconf", "Min-Conflicts"),
        ("genetic", "Genetic Alg.")
    ]
    
    button_width = 120
    button_height = 35
    gap = 10
    
    # Vị trí cho 2 cột
    col1_x = 500
    col2_x = 630
    start_y = 30
    
    # Vẽ các nút thuật toán trong 2 cột
    for i, (alg_name, button_text) in enumerate(buttons):
        # Xác định vị trí cột
        if i < 8:  # 8 buttons đầu vào cột 1
            x = col1_x
            y = start_y + (i % 8) * (button_height + gap)
        else:  # buttons sau vào cột 2
            x = col2_x
            y = start_y + ((i - 8) % 8) * (button_height + gap)
            
        pygame.draw.rect(screen, LIGHT_BLUE if algorithm == alg_name else GRAY,
                        (x, y, button_width, button_height))
        text = small_font.render(button_text, True, BLACK)
        text_rect = text.get_rect(center=(x + button_width//2, y + button_height//2))
        screen.blit(text, text_rect)

    # Nút bắt đầu / dừng - đặt dưới board
    button_color = GREEN if not running else RED
    button_text = "Start" if not running else "Stop"
    pygame.draw.rect(screen, button_color, (150, 500, 200, 50))
    start_text = font.render(button_text, True, BLACK)
    screen.blit(start_text, (215, 510))
    
    # Hiển thị thông tin về trạng thái hiện tại
    if current_state:
        h_value = heuristic(current_state)
        m_value = manhattan_distance(current_state)
        misplaced = misplaced_tiles(current_state)
        
        h_text = small_font.render(f"Heuristic: {h_value} (M:{m_value} + T:{misplaced})", True, WHITE)
        screen.blit(h_text, (30, 470))
    
    # Hiển thị số bước nếu đang chạy giải pháp
    if solution_path and running:
        step_text = small_font.render(f"Step: {step_index}/{len(solution_path)-1}", True, WHITE)
        screen.blit(step_text, (280, 470))

def reset_puzzle():
    global current_state, solution_path, step_index, running
    current_state = current
    solution_path = []
    step_index = 0
    running = False

def main():
    global current_state, algorithm, running, solution_path, step_index
    
    # Khởi tạo trạng thái ban đầu
    reset_puzzle()
    
    # Kiểm tra tính giải được
    if not is_solvable(current_state):
        print("Trạng thái bắt đầu không thể giải được")
        current_state = current
    
    last_update_time = 0
    
    # Vòng lặp chính
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Kiểm tra click chuột cho 2 cột buttons
                button_width = 120
                button_height = 35
                gap = 10
                col1_x = 500
                col2_x = 630
                start_y = 30
                
                buttons = [
                    ("simple", "Hill Climbing"), ("steepest", "Steepest"),
                    ("random", "Random"), ("beam", "Beam Search"),
                    ("bfs", "BFS"), ("dfs", "DFS"),
                    ("backtrack", "Backtrack"),
                    ("backtrackf", "Backtrack Fwd"),
                    ("ids", "IDS"), ("greedy", "Greedy"),
                    ("ucs", "UCS"), ("astar", "A*"),
                    ("idastar", "IDA*"), ("simann", "Sim. Annealing"),
                    ("minconf", "Min-Conflicts"),
                    ("genetic", "Genetic Alg.")
                ]
                
                # Check cả 2 cột
                for i, (alg_name, _) in enumerate(buttons):
                    if i < 8:
                        x = col1_x
                        y = start_y + (i % 8) * (button_height + gap)
                    else:
                        x = col2_x
                        y = start_y + ((i - 8) % 8) * (button_height + gap)
                        
                    if x <= mouse_pos[0] <= x + button_width and y <= mouse_pos[1] <= y + button_height:
                        algorithm = alg_name
                        reset_puzzle()
                        break
                
                # Kiểm tra nút bắt đầu/dừng
                if 150 <= mouse_pos[0] <= 350 and 500 <= mouse_pos[1] <= 550:
                    if algorithm and not running:
                        running = True
                        try:
                            solution_path = {
                                "simple": simple_hill_climbing,
                                "steepest": steepest_hill_climbing,
                                "random": random_hill_climbing,
                                "beam": beam_search,
                                "bfs": bfs,
                                "dfs": dfs,
                                "backtrack": backtracking_search,
                                "backtrackf": backtracking_forward_search,
                                "ids": ids,
                                "greedy": greedy_search,
                                "ucs": uniform_cost_search,
                                "astar": astar_search,
                                "idastar": ida_star_search,
                                "simann": simulated_annealing,
                                "minconf": min_conflicts_search,
                                "genetic": genetic_algorithm_search
                            }[algorithm](current_state)
                            if not solution_path or len(solution_path) == 0:
                                print("No solution found.")
                                running = False
                                step_index = 0
                            else:
                                print(f"Found path: {len(solution_path)} steps")
                                step_index = 0
                        except Exception as e:
                            print(f"Error: {e}")
                            running = False
                            solution_path = []
                            step_index = 0
                    else:
                        running = False
        
        # Vẽ giao diện
        draw_interface()
        
        # Cập nhật hiển thị không in debug
        current_time = pygame.time.get_ticks()
        if running and solution_path and len(solution_path) > 0 and current_time - last_update_time > delay:
            if step_index < len(solution_path) - 1:
                step_index += 1
                current_state = solution_path[step_index]
                last_update_time = current_time
            else:
                running = False

        pygame.display.flip()
        pygame.time.delay(30)

if __name__ == '__main__':
    current_state = current  # Đảm bảo current_state được khởi tạo khi chạy trực tiếp
    main()

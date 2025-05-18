# Puzzle-8

Ứng dụng Python trực quan hóa quá trình giải bài toán 8-Puzzle bằng nhiều thuật toán tìm kiếm khác nhau, hỗ trợ so sánh hiệu quả giữa các chiến lược tìm kiếm.

## 📦 Cấu trúc Dự án

```
.
├── Puzzle_8.py           # Giao diện chính sử dụng pygame, tích hợp thuật toán
├── Algorithms.py         # Cài đặt các thuật toán tìm kiếm
└── utils.py              # Các hàm tiện ích như heuristic, kiểm tra tính khả giải, v.v.
```

## 🚀 Hướng dẫn chạy

1. Cài đặt các thư viện cần thiết:

```bash
pip install pygame
```

2. Chạy chương trình:

```bash
python Puzzle_8.py
```

## 🧠 Thuật toán được hỗ trợ

- Simple Hill Climbing
- Steepest Ascent Hill Climbing
- Random Restart Hill Climbing
- Beam Search
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Backtracking Search
- Backtracking with Forward Checking
- Iterative Deepening Search (IDS)
- Greedy Search
- Uniform Cost Search
- A* Search
- IDA* Search
- Simulated Annealing
- Min-Conflicts Search
- Genetic Algorithm

## 🎮 Giao diện

- Bàn cờ được hiển thị trực quan với các ô số từ 1 đến 8 và 1 ô trống.
- Các nút lựa chọn thuật toán chia thành 2 cột.
- Nút **Start/Stop** để bắt đầu hoặc dừng thuật toán.
- Hiển thị số bước, giá trị heuristic, và các thông tin hữu ích khi chạy giải pháp.

## 🧩 Cài đặt trạng thái

Trạng thái khởi đầu được định nghĩa sẵn trong mã nguồn (`current`):

```python
current = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
```

Mục tiêu (goal state) là:

```python
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
```

## ✅ Tính năng

- Kiểm tra tính khả giải trước khi chạy thuật toán.
- Hiển thị kết quả trực quan từng bước.
- Dễ dàng chuyển đổi giữa các thuật toán để so sánh hiệu quả.

## 📌 Lưu ý

- Một số thuật toán có thể không tìm được lời giải tùy thuộc vào trạng thái đầu vào.
- Giao diện hoạt động tốt với độ phân giải từ 800x600 trở lên.

---

**Tác giả**: _Vũ Minh Khang_  

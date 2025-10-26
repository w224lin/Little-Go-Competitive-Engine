import numpy as np
import os
import random

BOARD_SIZE = 5  # 5x5 棋盘

STONE_EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = 2

class RandomPlayer:
    def __init__(self, side=None, prev_state=None, curr_state=None):
        self.side = side          # 1 表示黑棋，2 表示白棋
        self.prev_state = prev_state  # 上一步棋盘状态（numpy 数组）
        self.curr_state = curr_state  # 当前棋盘状态（numpy 数组）

    def set_side(self, side):
        self.side = side

    def set_prev_board(self, prev_state):
        self.prev_state = prev_state

    def set_board(self, curr_state):
        self.curr_state = curr_state

    def is_valid_move(self, board, i, j, side):
        """
        判断落子是否合法：
        1. 落子位置必须为空；
        2. 模拟落子后己方棋子必须至少有一个气（非自杀着）；
        3. 模拟落子后棋盘状态不能与上一状态完全相同（打劫规则）。
        """
        # 1. 检查该位置是否为空
        if board[i][j] != STONE_EMPTY:
            return False

        # 2. 模拟落子后的状态
        new_board = self.simulate_move(board, i, j, side)
        group, liberties = self.get_group(new_board, i, j)
        if len(liberties) == 0:
            return False  # 自杀着

        # 3. 检查打劫：新状态不能与上一状态完全相同
        if self.prev_state is not None and np.array_equal(new_board, self.prev_state):
            return False

        return True

    def simulate_move(self, board, i, j, side):
        """
        模拟在 (i, j) 落子后的棋盘状态，并执行提子操作。
        board 为 numpy 数组表示的棋盘状态。
        """
        new_board = np.copy(board)
        new_board[i][j] = side
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        # 检查落子点相邻的对方棋子是否因失去所有气而被提掉
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = i + dx, j + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if new_board[nx][ny] == opponent:
                    group, liberties = self.get_group(new_board, nx, ny)
                    if len(liberties) == 0:
                        for (x, y) in group:
                            new_board[x][y] = STONE_EMPTY
        return new_board

    def get_group(self, board_state, i, j):
        """
        获取 board_state（numpy 数组）中 (i, j) 处棋子的连通分组及该分组的所有气（空邻点）。
        """
        color = board_state[i][j]
        visited = set()
        group = set()
        liberties = set()
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            group.add((x, y))
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    if board_state[nx][ny] == STONE_EMPTY:
                        liberties.add((nx, ny))
                    elif board_state[nx][ny] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        return group, liberties

    def move(self):
        """
        执行落子：
        1. 遍历当前棋盘所有位置，判断哪些是合法落子；
        2. 如果存在合法落子，则随机挑选一个并写入 output.txt；
        3. 如果没有合法落子，则写入 "PASS"。
        """
        board = self.curr_state
        legal_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.is_valid_move(board, i, j, self.side):
                    legal_moves.append((i, j))
        if not legal_moves:
            self.save_move("PASS")
        else:
            move = random.choice(legal_moves)
            self.save_move(f"{move[0]},{move[1]}")

    def save_move(self, content):
        """
        将落子点或 PASS 写入 output.txt。
        """
        file_path = "output.txt"
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")
        with open(file_path, "w") as f:
            f.write(content)

if __name__ == "__main__":
    # 解析输入文件 input.txt，格式与 QLearner 相同：
    # 第一行：执棋方 (1:黑, 2:白)
    # 第2-6行：上一步棋盘状态
    # 第7-11行：当前棋盘状态
    player = RandomPlayer()
    file_name = "input.txt"
    with open(file_name, "r") as f:
        lines = f.readlines()
        side = int(lines[0].strip())
        prev_state = [ [int(ch) for ch in line.strip()] for line in lines[1:6]]
        curr_state = [ [int(ch) for ch in line.strip()] for line in lines[6:11]]
        player.set_side(side)
        player.set_prev_board(np.array(prev_state))
        player.set_board(np.array(curr_state))
        player.move()

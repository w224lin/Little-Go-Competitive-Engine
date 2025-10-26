import sys
from pathlib import Path
import os
import numpy as np

# 常量定义
BOARD_SIZE = 5
STONE_EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = 2
KOMI = 2.5
SIDE_BLACK = 1
SIDE_WHITE = 2

# ------------------ Board 类 ------------------
class Board:
    def __init__(self):
        # current 表示当前棋盘状态，prev 表示上一步棋盘状态
        self.current = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.prev = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_count = 0             # 记录操作的总次数（包括 PASS）
        self.consecutive_passes = 0     # 连续 PASS 次数

    def to_input_file(self, turn):
        """
        将当前状态写入 input.txt 文件：
          - 第一行：当前执棋方（"1" 或 "2"）
          - 接下来 5 行：上一步棋盘状态，每行以5个数字构成的字符串，如 "00000"
          - 接下来 5 行：当前棋盘状态，同样格式
        """
        with open("input.txt", "w") as f:
            f.write(str(turn) + "\n")
            for i in range(BOARD_SIZE):
                row = "".join(map(str, self.prev[i]))
                f.write(row + "\n")
            for i in range(BOARD_SIZE):
                row = "".join(map(str, self.current[i]))
                f.write(row + "\n")

    def update(self, move, turn):
        """
        根据玩家落子更新棋盘：
          - move 为字符串格式，如 "i,j"；若为 "PASS" 则不更新棋盘。
          - 更新前先将当前棋盘保存到 prev 中。
          - 无论 move 是否为 "PASS"，均将 move_count 加 1（记录所有操作）。
          - 如果 move 为 "PASS"，则将 consecutive_passes 加 1；
            否则更新棋盘、将 consecutive_passes 重置为 0，并调用 capture_stones 实现提子。
        """
        self.prev = self.current.copy()
        self.move_count += 1
        if move == "PASS":
            self.consecutive_passes += 1
            return False
        else:
            try:
                i, j = map(int, move.split(","))
            except Exception as e:
                print("Error parsing move:", move, e)
                return False
            self.current[i][j] = turn
            self.consecutive_passes = 0
            self.capture_stones(i, j, turn)
            return True

    def capture_stones(self, i, j, turn):
        """
        检查刚落子 (i,j) 后，相邻的对方棋子组是否失去气，如果是则剔除。
        """
        opponent = STONE_BLACK if turn == STONE_WHITE else STONE_WHITE
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = i + dx, j + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if self.current[nx][ny] == opponent:
                    group, liberties = self.get_group(self.current, nx, ny)
                    if len(liberties) == 0:
                        for (x, y) in group:
                            self.current[x][y] = STONE_EMPTY

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

    def game_over(self):
        """
        游戏结束条件：
          1. 当两方总计操作次数达到 24 次（无论落子或 PASS），游戏结束；
          2. 或者当连续两次 PASS 时游戏结束。
        """
        return (self.move_count >= 24) or (self.consecutive_passes >= 2)

    def reset(self):
        self.current = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.prev = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_count = 0
        self.consecutive_passes = 0

    def result(self):
        """
        计算游戏结果：
          - 黑棋得分：棋盘上 STONE_BLACK 的数量
          - 白棋得分：棋盘上 STONE_WHITE 的数量加上 komi
          返回 SIDE_BLACK（黑胜）、SIDE_WHITE（白胜）或 0（平局）。
        """
        count_black = np.sum(self.current == STONE_BLACK)
        count_white = np.sum(self.current == STONE_WHITE) + KOMI
        if count_black > count_white:
            return SIDE_BLACK
        elif count_white > count_black:
            return SIDE_WHITE
        else:
            return 0

# ------------------ 游戏对战逻辑 ------------------
def simulate_game():
    """
    模拟一局游戏：
      1. 初始状态：创建 input.txt 文件，其第一行为 "1"，后面 10 行均为 "00000"（空棋盘）。
      2. 游戏循环：
         - 根据当前内部棋盘状态写入 input.txt（包含执棋方、上一步状态和当前棋盘状态）。
         - 根据当前执棋方（1：黑；2：白）调用对应玩家程序：
              黑子调用 QLearner 程序（my_player3.py），
              白子调用 RandomPlayer 程序（RandomPlayer.py）。
         - 玩家程序读取 input.txt，计算落子，并写入 output.txt。
         - chest 读取 output.txt，将落子应用到内部棋盘，并切换执棋方。
      3. 游戏结束条件：当操作次数达到 24 或连续两次 PASS 时结束。
      4. 返回游戏结果（SIDE_BLACK、SIDE_WHITE 或 0）。
    """
    board = Board()
    turn = SIDE_BLACK  # 黑子先下
    board.to_input_file(turn)

    while not board.game_over():
        board.to_input_file(turn)
        if turn == SIDE_BLACK:
            os.system("python my_player3.py")
        else:
            os.system("python RandomPlayer.py")
        with open("output.txt", "r") as f:
            move = f.read().strip()
        board.update(move, turn)
        turn = SIDE_WHITE if turn == SIDE_BLACK else SIDE_BLACK

    return board.result()

def battle(num_games):
    """
    对战 num_games 局，并统计结果（以 QLearner 为参考）。
    返回一个字典，键为结果（SIDE_BLACK、SIDE_WHITE、0），值为局数。
    """
    stats = {SIDE_BLACK: 0, SIDE_WHITE: 0, 0: 0}
    for _ in range(num_games):
        result = simulate_game()
        stats[result] += 1
        with open("input.txt", "w") as f:
            f.write("1\n")
            for _ in range(10):
                f.write("00000\n")
    return stats

# ------------------ 主程序 ------------------
def main():
    num_train = 100
    num_test = 10

    print("初始生成 input.txt：")
    with open("input.txt", "w") as f:
        f.write("1\n")
        for _ in range(10):
            f.write("00000\n")

    # 训练阶段：确保训练时调用 my_player3.py 时加载并更新共享的 qtable.txt
    train = True
    if train:
        print("开始训练 QLearner 与 RandomPlayer 对战 {} 局...".format(num_train))
        for _ in range(num_train):
            simulate_game()
        print("训练结束，qtable.txt 已被各局 my_player3.py 更新保存。")

    # 测试阶段：对战后统计结果
    print("开始测试对战 {} 局...".format(num_test))
    stats = battle(num_test)
    wins = stats[SIDE_BLACK]
    draws = stats[0]
    losses = stats[SIDE_WHITE]
    total = num_test
    win_rate = wins / total * 100
    draw_rate = draws / total * 100
    loss_rate = losses / total * 100
    print("测试结果：")
    print("  QLearner（黑棋）胜率：{:.1f}%".format(win_rate))
    print("  平局率：{:.1f}%".format(draw_rate))
    print("  QLearner 败率（白棋胜）：{:.1f}%".format(loss_rate))

if __name__ == "__main__":
    main()

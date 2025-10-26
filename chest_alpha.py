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
def simulate_game(alpha_side):
    """
    模拟一局游戏：
      - 参数 alpha_side 指定 AlphabetaPlayer 执棋方（SIDE_BLACK 或 SIDE_WHITE）。
      - 游戏初始时黑棋先下，但根据 alpha_side 在不同走子时调用相应玩家程序：
            当当前走子方等于 alpha_side 时调用 AlphabetaPlayer 程序，
            否则调用 RandomPlayer 程序。
      - 游戏结束条件：当操作次数达到 24 或连续两次 PASS 时结束。
      - 返回游戏结果（SIDE_BLACK、SIDE_WHITE 或 0）。
    """
    board = Board()
    turn = SIDE_BLACK  # 黑棋先下
    board.to_input_file(turn)

    while not board.game_over():
        board.to_input_file(turn)
        # 根据当前执棋方判断调用哪个玩家程序
        if turn == alpha_side:
            os.system("python AlphabetaPlayer.py")
        else:
            os.system("python RandomPlayer.py")
        with open("output.txt", "r") as f:
            move = f.read().strip()
        board.update(move, turn)
        turn = SIDE_WHITE if turn == SIDE_BLACK else SIDE_BLACK

    return board.result()

def battle(alpha_side, num_games):
    """
    对战 num_games 局，统计结果。
      - 如果 alpha_side == SIDE_BLACK，则 AlphabetaPlayer 为黑棋；
        若 alpha_side == SIDE_WHITE，则 AlphabetaPlayer 为白棋。
      - 返回一个字典，键为游戏结果（SIDE_BLACK、SIDE_WHITE、0），值为对应局数。
    """
    stats = {SIDE_BLACK: 0, SIDE_WHITE: 0, 0: 0}
    for _ in range(num_games):
        result = simulate_game(alpha_side)
        stats[result] += 1
        # 每局结束后重置 input.txt 文件为初始状态
        with open("input.txt", "w") as f:
            f.write("1\n")
            for _ in range(10):
                f.write("00000\n")
    return stats

# ------------------ 主程序 ------------------
def main():
    num_test = 20  # 测试局数

    # 初始化 input.txt 文件（初始执棋方为黑棋）
    with open("input.txt", "w") as f:
        f.write("1\n")
        for _ in range(10):
            f.write("00000\n")

    # 模拟 AlphabetaPlayer 执黑对战
    print("开始测试对战 {} 局（AlphabetaPlayer 执黑棋）...".format(num_test))
    stats_black = battle(SIDE_BLACK, num_test)
    wins_black = stats_black[SIDE_BLACK]
    draws_black = stats_black[0]
    losses_black = stats_black[SIDE_WHITE]
    total_black = num_test
    win_rate_black = wins_black / total_black * 100
    print("  AlphabetaPlayer（黑棋）胜率：{:.1f}%".format(win_rate_black))
    print("  平局率：{:.1f}%".format(draws_black / total_black * 100))
    print("  败率（白棋胜）：{:.1f}%".format(losses_black / total_black * 100))

    # 模拟 AlphabetaPlayer 执白对战
    print("\n开始测试对战 {} 局（AlphabetaPlayer 执白棋）...".format(num_test))
    stats_white = battle(SIDE_WHITE, num_test)
    wins_white = stats_white[SIDE_WHITE]
    draws_white = stats_white[0]
    losses_white = stats_white[SIDE_BLACK]
    total_white = num_test
    win_rate_white = wins_white / total_white * 100
    print("  AlphabetaPlayer（白棋）胜率：{:.1f}%".format(win_rate_white))
    print("  平局率：{:.1f}%".format(draws_white / total_white * 100))
    print("  败率（黑棋胜）：{:.1f}%".format(losses_white / total_white * 100))
    

if __name__ == "__main__":
    main()

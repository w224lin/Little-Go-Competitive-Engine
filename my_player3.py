import time
import numpy as np

BOARD_SIZE = 5
STONE_EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = 2
KOMI = 2.5  # 白棋补偿

# --------------------
# 位板辅助函数
# --------------------
def get_cell(bitboard, i, j):
    """返回位板中 (i, j) 处的值（占用2位）。"""
    index = i * BOARD_SIZE + j
    return (bitboard >> (2 * index)) & 0b11

def set_cell(bitboard, i, j, value):
    """设置位板中 (i, j) 处的值，返回更新后的位板。"""
    index = i * BOARD_SIZE + j
    mask = ~(0b11 << (2 * index))
    bitboard = bitboard & mask
    return bitboard | (value << (2 * index))

def bitboard_to_tuple(bitboard):
    """将位板转换为二维元组表示，每行为一个元组，共 BOARD_SIZE 行。"""
    return tuple(tuple(get_cell(bitboard, i, j) for j in range(BOARD_SIZE)) for i in range(BOARD_SIZE))

def rotate_board(board_tuple, k):
    """将二维棋盘元组顺时针旋转 k*90 度。"""
    board = [list(row) for row in board_tuple]
    for _ in range(k):
        board = list(zip(*board[::-1]))
    return tuple(tuple(row) for row in board)

def flip_board(board_tuple):
    """将二维棋盘元组水平翻转。"""
    return tuple(tuple(row[::-1]) for row in board_tuple)

def array_to_bitboard(array):
    """将二维列表或元组转换为位板表示。"""
    bitboard = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = array[i][j]
            index = i * BOARD_SIZE + j
            bitboard |= (value << (2 * index))
    return bitboard

# --------------------
# AlphabetaPlayer 类
# --------------------
class AlphabetaPlayer:
    def __init__(self, side=None, prev_state=None, curr_state=None, base_depth=3, time_limit=5.0):
        self.side = side              # 1：黑，2：白
        self.prev_state = prev_state  # 位板表示的上一局面（用于打劫判断）
        self.curr_state = curr_state  # 位板表示的当前局面
        self.base_depth = base_depth
        self.time_limit = time_limit  # 每步允许的最大计算时间（秒）
        self.transposition_table = {}  # 置换表，key为 (canonical_board, depth, maximizingPlayer)
        self.killer_moves = {}       # killer_moves[depth] = [move1, move2, ...]
        self.history_table = {}      # history_table[(i, j)] = score

    def set_side(self, side):
        self.side = side

    def set_prev_board(self, prev_state):
        self.prev_state = prev_state

    def set_board(self, curr_state):
        self.curr_state = curr_state

    def canonical_board(self, bitboard):
        """利用所有旋转和翻转对称性，返回二维元组中字典序最小的状态作为规范形式。"""
        board_tuple = bitboard_to_tuple(bitboard)
        transformations = []
        for k in range(4):
            rot = rotate_board(board_tuple, k)
            transformations.append(rot)
            transformations.append(flip_board(rot))
        return min(transformations)

    def get_group(self, bitboard, i, j):
        """获取 (i, j) 处棋子的连通分组及其气（空邻点）。"""
        color = get_cell(bitboard, i, j)
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
                    cell_val = get_cell(bitboard, nx, ny)
                    if cell_val == STONE_EMPTY:
                        liberties.add((nx, ny))
                    elif cell_val == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        return group, liberties

    def simulate_move(self, bitboard, i, j, side):
        """模拟在 (i, j) 落子后的局面，并提掉对手无气棋子。"""
        new_board = set_cell(bitboard, i, j, side)
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = i + dx, j + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if get_cell(new_board, nx, ny) == opponent:
                    group, liberties = self.get_group(new_board, nx, ny)
                    if len(liberties) == 0:
                        for (x, y) in group:
                            new_board = set_cell(new_board, x, y, STONE_EMPTY)
        return new_board

    def is_valid_move(self, bitboard, i, j, side):
        """判断 (i, j) 落子是否合法：空点、非自杀且不违反打劫。"""
        if get_cell(bitboard, i, j) != STONE_EMPTY:
            return False
        new_board = self.simulate_move(bitboard, i, j, side)
        group, liberties = self.get_group(new_board, i, j)
        if len(liberties) == 0:
            return False
        if self.prev_state is not None and new_board == self.prev_state:
            return False
        return True

    def generate_legal_moves(self, bitboard, side):
        """遍历棋盘生成所有合法落子（返回 (i, j) 列表）。"""
        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if get_cell(bitboard, i, j) == STONE_EMPTY and self.is_valid_move(bitboard, i, j, side):
                    moves.append((i, j))
        return moves

    def get_board_group_score(self, bitboard, side):
        """
        连通分组评估：对每个分组计算得分 = 棋子数 + 0.5×气数，
        返回己方各组得分减去对手各组得分之差。
        """
        factor = 0.5
        visited = set()
        player_score = 0
        opponent_score = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if get_cell(bitboard, i, j) != STONE_EMPTY and (i, j) not in visited:
                    group, liberties = self.get_group(bitboard, i, j)
                    visited.update(group)
                    group_value = len(group) + factor * len(liberties)
                    if get_cell(bitboard, i, j) == side:
                        player_score += group_value
                    else:
                        opponent_score += group_value
        return player_score - opponent_score

    def compute_influence(self, bitboard, side):
        """
        影响图评估：累加每个己方棋子对全局各点的衰减贡献，再减去对手的贡献。
        """
        player_influence = 0
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        opponent_influence = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cell = get_cell(bitboard, i, j)
                if cell == side:
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
                            dist = abs(i - x) + abs(j - y)
                            player_influence += 1 / (dist + 1)
                elif cell == opponent:
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
                            dist = abs(i - x) + abs(j - y)
                            opponent_influence += 1 / (dist + 1)
        return player_influence - opponent_influence

    def dynamic_weights(self, bitboard):
        """
        根据局面阶段动态调整各项指标权重：
          - 开局（棋子数 < 6）：强调影响图和气值  
          - 中盘（6 <= 棋子数 < 15）：各指标均衡  
          - 残局（棋子数 >= 15）：更注重连通分组和领域评估
        返回字典，包含 "group", "influence", "territory", "liberty", "ko" 五项权重。
        (防守性调整：在各阶段增加对“territory”与“liberty”的权重)
        """
        count = sum(1 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if get_cell(bitboard, i, j) != STONE_EMPTY)
        if count < 6:
            return {"group": 1.0, "influence": 1.0, "territory": 0.7, "liberty": 1.8, "ko": 1.0}
        elif count < 15:
            return {"group": 1.1, "influence": 0.7, "territory": 0.9, "liberty": 1.3, "ko": 1.0}
        else:
            return {"group": 1.3, "influence": 0.5, "territory": 1.1, "liberty": 1.0, "ko": 1.0}

    def evaluate_territory(self, bitboard, side):
        """
        领域评估：对每个空点，根据其正交邻点情况给予加分或扣分，最后乘以固定权重（0.3）。
        """
        territory_score = 0
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if get_cell(bitboard, i, j) == STONE_EMPTY:
                    neighbors = []
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = i+dx, j+dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            val = get_cell(bitboard, nx, ny)
                            if val != STONE_EMPTY:
                                neighbors.append(val)
                    if neighbors:
                        if all(n == side for n in neighbors):
                            territory_score += 1
                        elif all(n == opponent for n in neighbors):
                            territory_score -= 1
        return territory_score * 0.3

    def liberty_quality(self, bitboard, i, j):
        """
        对于一个空气点，根据其周围空邻点数量给予基础分值，
        并根据是否处于角或边进行调整。
        """
        count_empty = 0
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = i+dx, j+dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if get_cell(bitboard, nx, ny) == STONE_EMPTY:
                    count_empty += 1
        quality = 0.5 + 0.25 * count_empty
        if (i == 0 or i == BOARD_SIZE-1) and (j == 0 or j == BOARD_SIZE-1):
            quality *= 0.9
        elif i == 0 or i == BOARD_SIZE-1 or j == 0 or j == BOARD_SIZE-1:
            quality *= 0.95
        return quality

    def evaluate_liberty_quality(self, bitboard, side):
        """
        对己方所有棋子群体，计算每个群体剩余气的平均质量，并累加作为整体气值评分。
        """
        visited = set()
        total_quality = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if get_cell(bitboard, i, j) == side and (i, j) not in visited:
                    group, liberties = self.get_group(bitboard, i, j)
                    visited.update(group)
                    if liberties:
                        group_quality = sum(self.liberty_quality(bitboard, li, lj) for (li, lj) in liberties) / len(liberties)
                        total_quality += group_quality
        return total_quality

    def evaluate_ko_risk(self, bitboard, side):
        """
        检查对手是否存在单子且只有一个气的局面，作为可能的劫风险进行惩罚。
        """
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        visited = set()
        risk = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if get_cell(bitboard, i, j) == opponent and (i, j) not in visited:
                    group, liberties = self.get_group(bitboard, i, j)
                    visited.update(group)
                    if len(group) == 1 and len(liberties) == 1:
                        risk += 1
        return risk

    def pattern_bonus(self, bitboard, move, side):
        """
        模式奖励：结合捕获效果和群体连通性奖励，同时对过度冒险（导致该群体气值显著下降）的走法给予惩罚。
        """
        i, j = move
        new_board = self.simulate_move(bitboard, i, j, side)
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        before = sum(1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
                     if get_cell(bitboard, x, y) == opponent)
        after = sum(1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
                    if get_cell(new_board, x, y) == opponent)
        capture_bonus = (before - after) * 0.8
        connection_bonus = 0
        friendly_neighbors = 0
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i+dx, j+dy
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                if get_cell(bitboard, ni, nj) == side:
                    friendly_neighbors += 1
        if friendly_neighbors >= 2:
            connection_bonus = 0.5
        # 计算新局面中，新落子所在群体的平均气值，如果低于阈值，则认为过于冒险，给予惩罚
        group, liberties = self.get_group(new_board, i, j)
        if liberties:
            avg_liberty = sum(self.liberty_quality(new_board, li, lj) for (li, lj) in liberties) / len(liberties)
            penalty = 0
            if avg_liberty < 0.7:
                penalty = (0.7 - avg_liberty) * 1.0  # 惩罚因子可调
        else:
            penalty = 0
        return capture_bonus + connection_bonus - penalty

    def domain_bonus(self, bitboard, move):
        """
        开局领域奖励：在局面简单时（棋子数较少），对角部落子奖励较高，边缘次之。
        """
        count = sum(1 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
                    if get_cell(bitboard, i, j) != STONE_EMPTY)
        if count < 6:
            i, j = move
            if (i == 0 or i == BOARD_SIZE - 1) and (j == 0 or j == BOARD_SIZE - 1):
                return 1.0
            elif i == 0 or i == BOARD_SIZE - 1 or j == 0 or j == BOARD_SIZE - 1:
                return 0.5
        return 0.0

    def evaluate_board(self, bitboard):
        """
        综合评估函数：结合连通分组得分、影响图、领域评估、气值和劫风险，
        并根据动态权重加权求和，返回总体局面分数。
        """
        base_score = self.get_board_group_score(bitboard, self.side)
        influence_score = self.compute_influence(bitboard, self.side)
        territory_score = self.evaluate_territory(bitboard, self.side)
        liberty_score = self.evaluate_liberty_quality(bitboard, self.side)
        ko_risk = self.evaluate_ko_risk(bitboard, self.side)
        weights = self.dynamic_weights(bitboard)
        total = (weights["group"] * base_score +
                 weights["influence"] * influence_score +
                 weights["territory"] * territory_score +
                 weights["liberty"] * liberty_score -
                 weights["ko"] * ko_risk)
        return total

    def evaluate_immediate_reward(self, bitboard, i, j, side):
        """计算在 (i,j) 落子产生的即时局面变化奖励。"""
        score_before = self.evaluate_board(bitboard)
        new_board = self.simulate_move(bitboard, i, j, side)
        score_after = self.evaluate_board(new_board)
        return score_after - score_before

    def limit_moves(self, bitboard, side, moves, depth):
        """
        动作排序：当合法走法超过10个时，
        综合考虑即时奖励、领域奖励、模式奖励和捕获潜力，
        同时利用置换表中缓存的最佳走法、killer move 以及历史启发给予额外加分，
        最后保留评分最高的前10个走法。
        """
        if len(moves) <= 10:
            return moves

        board_key = self.canonical_board(bitboard)
        cached_best_move = None
        for key in self.transposition_table:
            if key[0] == board_key and key[2] is True:
                _, candidate_move = self.transposition_table[key]
                if candidate_move is not None:
                    cached_best_move = candidate_move
                    break

        scored_moves = []
        opponent = STONE_BLACK if side == STONE_WHITE else STONE_WHITE
        for move in moves:
            immediate = self.evaluate_immediate_reward(bitboard, move[0], move[1], side)
            bonus = self.domain_bonus(bitboard, move)
            pattern = self.pattern_bonus(bitboard, move, side)
            new_board = self.simulate_move(bitboard, move[0], move[1], side)
            capture_count = sum(1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
                                if get_cell(bitboard, x, y) == opponent) - \
                            sum(1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
                                if get_cell(new_board, x, y) == opponent)
            capture_bonus = capture_count * 1.0
            extra = 100 if cached_best_move is not None and move == cached_best_move else 0
            killer_bonus = 0
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                killer_bonus = 200
            history_bonus = self.history_table.get(move, 0)
            total = immediate + bonus + pattern + capture_bonus + extra + killer_bonus + history_bonus
            scored_moves.append((move, total))
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored_moves[:10]]

    def alphabeta(self, bitboard, depth, alpha, beta, maximizingPlayer):
        """
        Alpha-beta 剪枝搜索，同时利用置换表缓存局面评估结果（key采用规范化后的棋盘）。
        同时在搜索过程中利用 killer move 和历史启发更新信息。
        """
        key = (self.canonical_board(bitboard), depth, maximizingPlayer)
        if key in self.transposition_table:
            return self.transposition_table[key]
        if depth == 0:
            eval_val = self.evaluate_board(bitboard)
            self.transposition_table[key] = (eval_val, None)
            return eval_val, None

        current_side = self.side if maximizingPlayer else (STONE_BLACK if self.side == STONE_WHITE else STONE_WHITE)
        legal_moves = self.generate_legal_moves(bitboard, current_side)
        legal_moves = self.limit_moves(bitboard, current_side, legal_moves, depth)
        if not legal_moves:
            eval_val = self.evaluate_board(bitboard)
            self.transposition_table[key] = (eval_val, None)
            return eval_val, None

        best_move = None
        if maximizingPlayer:
            value = -float('inf')
            for move in legal_moves:
                new_board = self.simulate_move(bitboard, move[0], move[1], current_side)
                child_value, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False)
                if child_value > value:
                    value = child_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.killer_moves.setdefault(depth, [])
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(move)
                    self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                    break
            self.transposition_table[key] = (value, best_move)
            return value, best_move
        else:
            value = float('inf')
            for move in legal_moves:
                new_board = self.simulate_move(bitboard, move[0], move[1], current_side)
                child_value, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True)
                if child_value < value:
                    value = child_value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    self.killer_moves.setdefault(depth, [])
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(move)
                    self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                    break
            self.transposition_table[key] = (value, best_move)
            return value, best_move

    def choose_move(self):
        """
        利用迭代加深与 Aspiration Window 搜索：
          - 根据局面复杂度动态确定最大搜索深度；
          - 在允许时间内不断加深搜索，并使用前一层结果构造窄窗口进行搜索，
            若返回值超出窗口则扩大窗口重新搜索。
        """
        start_time = time.time()
        best_move = None
        prev_value = 0
        delta = 10  # 初始窗口宽度

        legal_moves = self.generate_legal_moves(self.curr_state, self.side)
        move_count = len(legal_moves)
        if move_count >= 15:
            max_possible_depth = 4
        else:
            max_possible_depth = 5

        current_depth = 1
        while current_depth <= max_possible_depth:
            if time.time() - start_time > self.time_limit:
                break

            # Aspiration window：以前一层评分为中心
            alpha_bound = prev_value - delta
            beta_bound = prev_value + delta
            while True:
                value, move = self.alphabeta(self.curr_state, current_depth, alpha_bound, beta_bound, True)
                if value <= alpha_bound:
                    alpha_bound -= delta
                elif value >= beta_bound:
                    beta_bound += delta
                else:
                    break
                if time.time() - start_time > self.time_limit:
                    break
            prev_value = value
            if move is not None:
                best_move = move
            current_depth += 1

        if best_move is None:
            return "PASS"
        else:
            return f"{best_move[0]},{best_move[1]}"

    def save_move(self, content):
        print(content)
        with open("output.txt", "w") as f:
            f.write(content)

    def move(self):
        move = self.choose_move()
        self.save_move(move)

    def set_q_table(self, q_table):
        self.q_values = q_table

# --------------------
# 主程序：读取输入，转换为位板表示，并调用 move()
# --------------------
if __name__ == "__main__":
    file_name = "input.txt"
    with open(file_name, "r") as f:
        lines = f.readlines()
        side = int(lines[0].strip())
        prev_array = [[int(ch) for ch in line.strip()] for line in lines[1:6]]
        curr_array = [[int(ch) for ch in line.strip()] for line in lines[6:11]]
    prev_state = array_to_bitboard(prev_array)
    curr_state = array_to_bitboard(curr_array)
    player = AlphabetaPlayer(base_depth=4, time_limit=6.0)
    player.set_side(side)
    player.set_prev_board(prev_state)
    player.set_board(curr_state)
    player.move()

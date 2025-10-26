import numpy as np

# The board size is 5x5. 
BOARD_SIZE = 5
ONGOING = -1
X_WIN = 1
O_WIN = 2

# In a board, 0 stands for an empty point, 1 stands for a Black stone, and 2 stands for a White stone. 
STONE_EMPTY = '0'
STONE_BLACK = '1'
STONE_WHITE = '2'

# In board visualizations, X represents a Black stone, and O represents a White stone. 
VISUAL_EMPTY = ' '
VISUAL_BLACK = 'X'
VISUAL_WHITE = 'O'

# The maximum number of moves allowed
MOVE_MAX = BOARD_SIZE * BOARD_SIZE - 1

# KOMI
KOMI_VALUE = 2.5


# ============= BOARD =============
class Board:
    def __init__(self, state=None, show_board=False):
        # All points are set to STONE_EMPTY for new race.
        if state is None:
            self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        else:
            self.state = state.copy()
        
        self.game_result = ONGOING
        self.show_board  = show_board
    
    # Show the board
    def set_show_board(self, show_board):
        self.show_board = show_board

    # Encode the current state of the board as a string
    def encode_state(self):
        return ''.join([str(self.state[i][j]) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])
    
    # Reset the game.
    def reset(self):
        self.state.fill(STONE_EMPTY)
        self.game_result = ONGOING

    # Decide if a move is valid.
    def is_valid_move(self, row, col):
        return row < BOARD_SIZE and row >= 0 and col < BOARD_SIZE and col >=0 and self.state[row][col] == 0
    
    def move(self, row, col, player):
        """
        Parameters
        ----------
        row : 0, 1, 2, 3, 4
        col : 0, 1, 2, 3, 4
        player: X -> 1, O -> 2

        Returns
        -------
        state: state after the move
        result: game result after the move
        """
        if not self.is_valid_move(row, col):
            print (row, col)
            self.print_board()
            raise ValueError("Invalid Move")

        self.state[row][col] = player

        if self.show_board:
            p = VISUAL_BLACK if player == STONE_BLACK else VISUAL_WHITE
            print('player {} moved: {}, {}'.format(p, row, col))
            self.print_board()

        return self.state, self.game_result
    
    def game_over(self):
        return self.game_result != ONGOING
    
    def print_board(self):
        board = self.encode_state()
        board = board.replace(STONE_EMPTY, VISUAL_EMPTY)
        board = board.replace(STONE_BLACK, VISUAL_BLACK)
        board = board.replace(STONE_WHITE, VISUAL_WHITE)
        print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + ' | ' + board[3] + ' | ' + board[4])
        print('--- --- --- --- ---')
        print(' ' + board[5] + ' | ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
        print('--- --- --- --- ---')
        print(' ' + board[10] + ' | ' + board[11] + ' | ' + board[12] + ' | ' + board[13] + ' | ' + board[14])
        print('--- --- --- --- ---')
        print(' ' + board[15] + ' | ' + board[16] + ' | ' + board[17] + ' | ' + board[18] + ' | ' + board[19])
        print('--- --- --- --- ---')
        print(' ' + board[20] + ' | ' + board[21] + ' | ' + board[22] + ' | ' + board[23] + ' | ' + board[24])
        print('--- --- --- --- ---')
        print()
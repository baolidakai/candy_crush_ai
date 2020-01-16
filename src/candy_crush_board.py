"""Class for Candy Crush game."""
import utils
import numpy as np

class CandyCrushBoard(object):
  def __init__(self, config_file='../config/config1.txt'):
    """Initializes a Candy Crush board.

    Args:
      config_file: A file of M rows, each row contains N digits representing the colors of the board.
        Each digits must be one of 0, 1, ..., 6.
    """
    # _config contains a M x N matrix representing the
    # colors to drop.
    self._config = utils.load_config_from_file(config_file)
    assert self._config is not None
    self._M = len(self._config)
    self._N = len(self._config[0])
    # _ptrs[i] represents the indices of next candy to drop.
    # If _ptrs[5] = 3, next candy to drop for column
    # 5 is _config[3][5].
    self._ptrs = [0] * self._N
    # Total rewards.
    # For simplicity, reward is defined as number of candies that has been eliminated.
    # The candies that are automatically eliminated
    # by screen flush are counted.
    self._reward = 0
    # The board, represented as a three-dimensional matrix.
    self._board = [[[0 for _ in range(self._N)] for _ in range(self._N)] for _ in range(len(utils.COLUMNS))]
    # Fills the board with config.
    self.fill_board()
    # Flush the screen.
    self.flush()

  def advance_ptr(self, col):
    """Advances ptrs for col."""
    self._ptrs[col] += 1
    if self._ptrs[col] == self._M:
      self._ptrs[col] = 0

  def fill_board(self):
    """Fills the board with the config."""
    for col in range(self._N):
      for row in range(self._N):
        color = self._config[self._ptrs[col]][col]
        self.advance_ptr(col)
        self._board[color][row][col] = 1

  def flush(self):
    """Flushes until no move so all existing matches are eliminated and counted towards reward."""
    while True:
      if not self.flush_once():
        break

  def flush_once(self):
    """Flushes once, only eliminates current matches and do no more moves. Returns false if no flush is done."""
    return False

  def get_board(self):
    """Getter for the board."""
    return self._board


"""Class for Candy Crush game."""
import collections
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
    # Historical board states.
    self._histories = []
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
    self._histories.clear()
    while True:
      if not self.flush_once():
        break

  def get_color(self, row, col):
    """Returns the color at (row, col)."""
    for color in range(len(utils.COLORS)):
      if self._board[color][row][col] == 1:
        return color
    return -1

  def flush_once(self):
    """Flushes once, only eliminates current matches and do no more moves. Returns false if no flush is done."""
    # Checks for consecutive blocks of identical color.
    # Either three colors are identical in a row, or in a column.
    seed_row = -1
    seed_col = -1
    seed_color = -1
    found = False
    for row in range(self._N):
      if found:
        break
      for col in range(self._N - 2):
        if self.get_color(row, col) == self.get_color(row, col + 1) == self.get_color(row, col + 2):
          seed_row, seed_col = row, col
          seed_color = self.get_color(row, col)
          found = True
          break
    for col in range(self._N):
      if found:
        break
      for row in range(self._N - 2):
        if self.get_color(row, col) == self.get_color(row + 1, col) == self.get_color(row + 2, col):
          seed_row, seed_col = row, col
          seed_color = self.get_color(row, col)
          found = True
          break
    if seed_row == -1:
      return False
    # Finds the block.
    block = [(seed_row, seed_col)]
    visited = set(block)
    frontier = collections.deque()
    frontier.append((seed_row, seed_col))
    while frontier:
      curr_row, curr_col = frontier.popleft()
      for drow, dcol in utils.DIRS:
        neighbor = (curr_row + drow, curr_col + dcol)
        if 0 <= neighbor[0] < self._N and 0 <= neighbor[1] < self._N:
          if neighbor in visited:
            continue
          visited.add(neighbor)
          if self.get_color(neighbor[0], neighbor[1]) != seed_color:
            continue
          frontier.append(neighbor)
          block.append(neighbor)
    # Eliminates the block.
    block.sort()
    for row, col in block:
      for r in range(row, 0, -1):
        for channel in range(len(utils.COLUMNS)):
          self._board[channel][r][col] = self._board[channel][r - 1][col]
      new_color = self._config[self._ptrs[col]][col]
      self.advance_ptr(col)
      for channel in range(len(utils.COLUMNS)):
        self._board[channel][0][col] = 0
      self._board[new_color][0][col] = 1
    # TODO(bowendeng): Implements correctly.
    self._reward += len(block)
    self._histories.append(self._board)
    return True

  def get_board(self):
    """Getter for the board."""
    return self._board

  def get_histories(self):
    """Getter for the recent histories."""
    return self._histories

  def swap(self, cell1, cell2):
    """Swaps cell1 and cell2."""
    r1, c1 = cell1
    r2, c2 = cell2
    for channel in range(len(utils.COLUMNS)):
      self._board[channel][r1][c1], self._board[channel][r2][c2] = self._board[channel][r2][c2], self._board[channel][r1][c1]
    # TODO(bowendeng): Implements checking on whether this is useful.
    self.flush()


"""Class for Aixiaochu game."""
from collections import deque
import numpy as np

class Aixiaochu(object):
  # TODO(bowendeng): Adds base class used by candy crush.
  def __init__(self, rows=10, cols=10, num_colors=2, unlimited=False):
    self._rows = rows
    self._cols = cols
    self._num_colors = num_colors
    self._colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    if not 2 <= num_colors <= len(self._colors):
      raise ValueError('Number of colors must be a value between 2 and %d' % (len(self._colors,)))
    self._unlimited = unlimited
    # A matrix representing the state.
    # -1 represents empty space.
    self._board = self.initialize_board()
    # Total reward.
    self._total_reward = 0

  def get_random_color(self):
    return np.random.randint(0, self._num_colors)

  def initialize_board(self):
    # Initializes a random board with self.colors different colors.
    board = [[self.get_random_color() for _ in range(self._cols)] for _ in range(self._rows)]
    return board

  def click(self, row, col):
    """Clicks on row and col, returns -1 if the position is not click-able, otherwise, returns the reward, which is square of size of elimination."""
    # Run BFS to find all neighbors with same color.
    neighbors = self.get_neighbors(row, col)
    if len(neighbors) < 2:
      print('WARNING: number of neighbors is %d' % (len(neighbors), ))
      return -1
    reward = len(neighbors) ** 2
    # Update all neighbors to -1.
    for r, c in neighbors:
      self._board[r][c] = -1
    # Apply gravity.
    self.apply_gravity()
    # Compress columns.
    if not self._unlimited:
      self.compress_columns()
    self._total_reward += reward
    return reward

  def compress_columns(self):
    slow = 0
    fast = 0
    while fast < self._cols:
      if self._board[-1][fast] != -1:
        for row in range(self._rows):
          self._board[row][slow] = self._board[row][fast]
        slow += 1
        fast += 1
      else:
        fast += 1
    while slow < self._cols:
      for row in range(self._rows):
        self._board[row][slow] = -1
      slow += 1

  def apply_gravity(self):
    """Applies gravity so everything drops down."""
    for c in range(self._cols):
      slow = self._rows - 1
      fast = self._rows - 1
      while fast >= 0:
        if self._board[fast][c] != -1:
          self._board[slow][c] = self._board[fast][c]
          slow -= 1
          fast -= 1
        else:
          fast -= 1
      while slow >= 0:
        self._board[slow][c] = self.get_random_color() if self._unlimited else -1
        slow -= 1

  def is_valid_cell(self, row, col):
    return 0 <= row < self._rows and 0 <= col < self._cols and self._board[row][col] != -1

  def get_neighbors(self, row, col):
    if not self.is_valid_cell(row, col):
      print('WARNING: Invalid cell (%d, %d)' % (row, col))
      return []
    my_color = self._board[row][col]
    frontier = deque()
    visited = set()
    frontier.append((row, col))
    visited.add((row, col))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while len(frontier):
      r, c = frontier.popleft()
      for dr, dc in dirs:
        nbr = (r + dr, c + dc)
        if self.is_valid_cell(nbr[0], nbr[1]) and self._board[nbr[0]][nbr[1]] == my_color and nbr not in visited:
          frontier.append(nbr)
          visited.add(nbr)
    return list(visited)

  def print_board(self):
    # Prints the board.
    print('Current board:')
    self.print_delimiter()
    for row in self._board:
      print(' '.join(map(lambda c: '%2s' % (c,) if c != -1 else '  ', row)))
    self.print_delimiter()

  def print_total_reward(self):
    print('Total reward: %d' % (self._total_reward))

  def print_delimiter(self):
    print('=' * (self._cols * 3 - 1))

  def get_cells(self):
    cells = []
    for r in range(self._rows):
      for c in range(self._cols):
        if self._board[r][c] != -1:
          cells.append((r, c, self._colors[self._board[r][c]]))
    return cells


if __name__ == '__main__':
  game = Aixiaochu(rows=10, cols=10, num_colors=3, unlimited=True)
  game.print_board()
  for i in range(10):
    print('Reward = %d\n' % (game.click(5, 5),))
    game.print_board()
    game.print_total_reward()



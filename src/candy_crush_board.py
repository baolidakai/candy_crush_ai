"""Class for Candy Crush game."""
import collections
import copy
import utils
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dqn_base import resize, DQN


class CandyCrushBoard(object):
  def __init__(self, config_file):
    """Initializes a Candy Crush board.

    Args:
      config_file: A file of M rows, each row contains N digits representing the colors of the board.
        Each digits must be one of 0, 1, ..., 6.
    """
    # _config contains a M x N matrix representing the
    # colors to drop.
    assert config_file is not None
    self._config_file = config_file
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
    # Number of swaps so far.
    self._swaps = 0
    # The board, represented as a three-dimensional matrix.
    self._board = [[[0 for _ in range(self._N)] for _ in range(self._N)] for _ in range(len(utils.COLUMNS))]
    # Historical board states.
    self._histories = []
    # Fills the board with config.
    self.fill_board()
    # Flush the screen.
    self.flush()
    # Clears the reward.
    self._reward = 0
    # Clears the history.
    self._histories.clear()
    # Initializes DQN.
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self._dqn = None

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

  def get_block(self):
    """Gets the block to eliminate, or returns []."""
    # TODO(bowendeng): Adds other candies.
    # Checks for consecutive blocks of identical color.
    # Either three colors are identical in a row, or in a column.
    seed_row = -1
    seed_col = -1
    found = False
    for row in range(self._N):
      if found:
        break
      for col in range(self._N - 2):
        if self.get_color(row, col) == self.get_color(row, col + 1) == self.get_color(row, col + 2):
          seed_row, seed_col = row, col
          found = True
          break
    for col in range(self._N):
      if found:
        break
      for row in range(self._N - 2):
        if self.get_color(row, col) == self.get_color(row + 1, col) == self.get_color(row + 2, col):
          seed_row, seed_col = row, col
          found = True
          break
    if not found:
      return []
    # Finds the block.
    visited = set([(seed_row, seed_col)])
    frontier = collections.deque()
    frontier.append((seed_row, seed_col))
    seed_color = self.get_color(seed_row, seed_col)
    while frontier:
      curr_row, curr_col = frontier.popleft()
      for drow, dcol in utils.DIRS:
        neighbor = (curr_row + drow, curr_col + dcol)
        if 0 <= neighbor[0] < self._N and 0 <= neighbor[1] < self._N:
          if neighbor in visited:
            continue
          if self.get_color(neighbor[0], neighbor[1]) != seed_color:
            continue
          visited.add(neighbor)
          frontier.append(neighbor)
    # Only keeps the cells who is in a row of at least 3.
    block = []
    for row, col in visited:
      num_h = 1
      num_v = 1
      for c in range(col + 1, self._N):
        if (row, c) not in visited:
          break
        num_h += 1
      for c in range(col - 1, -1, -1):
        if (row, c) not in visited:
          break
        num_h += 1
      for r in range(row + 1, self._N):
        if (r, col) not in visited:
          break
        num_v += 1
      for r in range(row - 1, -1, -1):
        if (r, col) not in visited:
          break
        num_v += 1
      if num_h > 2 or num_v > 2:
        block.append((row, col))
    return block

  def flush_once(self):
    """Flushes once, only eliminates current matches and do no more moves. Returns false if no flush is done."""
    block = self.get_block()
    if not block:
      return False
    # Eliminates the block.
    block.sort()
    intermediate_board = self.copy_board()
    for row, col in block:
      for channel in range(len(utils.COLUMNS)):
        intermediate_board[channel][row][col] = 0
      self.eliminate_cell(row, col)
    self._histories.append(intermediate_board)
    self._histories.append(self.copy_board())
    return True

  def eliminate_cell(self, row, col):
    for r in range(row, 0, -1):
      for channel in range(len(utils.COLUMNS)):
        self._board[channel][r][col] = self._board[channel][r - 1][col]
    new_color = self._config[self._ptrs[col]][col]
    self.advance_ptr(col)
    for channel in range(len(utils.COLUMNS)):
      self._board[channel][0][col] = 0
    self._board[new_color][0][col] = 1
    self._reward += 1

  def copy_board(self):
    """Deep copy of the board."""
    return copy.deepcopy(self._board)

  def get_board(self):
    """Getter for the board."""
    return copy.deepcopy(self._board)

  def get_numpy_board(self):
    """Getter for the board."""
    return np.array(self.get_board())

  def get_histories(self):
    """Getter for the recent histories."""
    return self._histories

  def swap_helper(self, r1, c1, r2, c2):
    for channel in range(len(utils.COLUMNS)):
      self._board[channel][r1][c1], self._board[channel][r2][c2] = self._board[channel][r2][c2], self._board[channel][r1][c1]

  def swap(self, cell1, cell2):
    """Swaps cell1 and cell2."""
    self._swaps += 1
    r1, c1 = cell1
    r2, c2 = cell2
    self.swap_helper(r1, c1, r2, c2)
    block = self.get_block()
    if not block:
      # If no valid block, revert this change.
      self.swap_helper(r1, c1, r2, c2)
    self.flush()

  def is_feasible_swap(self, cell1, cell2):
    """Checks whether a swap is feasible."""
    r1, c1 = cell1
    r2, c2 = cell2
    self.swap_helper(r1, c1, r2, c2)
    block = self.get_block()
    self.swap_helper(r1, c1, r2, c2)
    return len(block) > 0
  
  def is_feasible_action(self, action_index):
   """Checks whether action_index-th action is feasible swap."""
   r1, c1, r2, c2 = self.get_actions()[action_index]
   return self.is_feasible_swap((r1, c1), (r2, c2))

  def num_swaps(self):
    """Getter for number of swaps."""
    return self._swaps

  def get_size(self):
    return self._N

  def get_score(self):
    """Getter for the reward."""
    return self._reward

  def ai_swap(self, method):
    """Use method to swap once, returns the reward."""
    r1, c1, r2, c2 = -1, -1, -1, -1
    if method == 'brute force':
      r1, c1, r2, c2 = self.brute_force_baseline()
    elif method == 'dqn':
      r1, c1, r2, c2 = self.predict_dqn()
    else:
      raise Exception('Invalid method')
    print('Swapping %d, %d and %d, %d' % (r1, c1, r2, c2))
    if r1 == -1:
      self.flush()
      self._swaps += 1
      return 0
    old_reward = self._reward
    self.swap((r1, c1), (r2, c2))
    return self._reward - old_reward


  def brute_force_baseline(self):
    """Returns a brute force action (r1, c1, r2, c2)."""
    # For each cell, search for right or bottom neighbor.
    for r1, c1, r2, c2 in self.get_actions():
      if self.is_feasible_swap((r1, c1), (r2, c2)):
          return r1, c1, r2, c2
    return -1, -1, -1, -1

  def get_action(self, action_index):
    return self.get_actions()[action_index]

  def get_actions(self):
    """Returns all possible (r1, c1, r2, c2)."""
    # For each cell, search for right or bottom neighbor.
    actions = []
    for row in range(self._N):
      for col in range(self._N):
        if row < self._N - 1:
          actions.append((row, col, row + 1, col))
        if col < self._N - 1:
          actions.append((row, col, row, col + 1))
    return actions

  def step(self, action_index):
    """Takes action_index-th action and returns the reward."""
    old_reward = self._reward
    r1, c1, r2, c2 = self.get_action(action_index)
    self.swap((r1, c1), (r2, c2))
    return self._reward - old_reward

  def reset(self):
    self.__init__(self._config_file)

  def is_done(self):
    """Returns whether the game is done."""
    return self._swaps >= utils.MAX_SWAPS

  def get_dqn_state(self):
    raw_state = self.get_numpy_board()
    raw_state = np.ascontiguousarray(raw_state, dtype=np.float32)
    raw_state = torch.from_numpy(raw_state)
    return resize(raw_state).unsqueeze(0).to(self._device)

  def init_dqn(self):
    """Initializes DQN."""
    print('Initializing DQN')
    self.get_dqn_state()
    init_screen = self.get_dqn_state()
    _, _, screen_height, screen_width = init_screen.shape
    actions = self.get_actions()
    n_actions = len(actions)
    target_net = DQN(screen_height, screen_width, n_actions).to(self._device)
    target_net.load_state_dict(torch.load('target_net'))
    target_net.eval()
    self._dqn = target_net
    print('Done')

  def predict_dqn(self):
    """Returns r1, c1, r2, c2 of the DQN."""
    action_scores = self._dqn(self.get_dqn_state())[0].detach().numpy()
    # Gets the best feasible score.
    best_action_index = 0
    max_score = float('-inf')
    for action_index in range(len(self.get_actions())):
      if not self.is_feasible_action(action_index):
        continue
      curr_score = action_scores[action_index]
      if curr_score > max_score:
        best_action_index, max_score = action_index, curr_score
    return self.get_action(best_action_index)


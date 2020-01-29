"""Utility functions and constants."""
# Maximum number of colors allowed.
MAX_COLOR = 7
# Maximum number of swaps allowed.
MAX_SWAPS = 40
# Columns of the board.
# 7 colors + is vertical strip bomb + is horizontal
# strip bomb + is color bomb + is candy bomb.
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
COLUMNS = COLORS + ['v_strip', 'h_strip', 'color_bomb', 'candy']
# Directions of neighbors.
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def load_config_from_file(config_file):
  """Loads the config file into a binary matrix of M x N. Returns None on error."""
  try:
    fin = open(config_file, 'r')
    rows = []
    for line in fin:
      rows.append(list(map(int, line.strip())))
    # Checks if the board is valid.
    if not rows:
      return None
    N = len(rows[0])
    for row in rows:
      if len(row) != N:
        print('Inconsistent number of digits on each line.')
        return None
    for row in rows:
      for digit in row:
        global MAX_COLOR
        if not 0 <= digit < MAX_COLOR:
          print('Invalid digit %d' % (digit,))
    return rows
  except:
    return None


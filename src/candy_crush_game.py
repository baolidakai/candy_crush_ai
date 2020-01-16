"""Pygame controller for the game."""
import sys
import utils
import candy_crush_board

def main():
  config_file = '../config/config1.txt' if len(sys.argv) < 2 else '../config/' + sys.argv[1]
  print('Loading from %s' % (config_file,))
  # Gets human board.
  human_board = candy_crush_board.CandyCrushBoard(config_file=config_file)
  # Gets computer board.
  computer_board = candy_crush_board.CandyCrushBoard(config_file=config_file)


if __name__ == '__main__':
  main()

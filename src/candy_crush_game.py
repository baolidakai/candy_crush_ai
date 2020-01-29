"""Pygame controller for the game."""
import math
import pygame
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
  SIZE = human_board.get_size()
  assert computer_board.get_size() == SIZE
  # Renders the two boards.
  pygame.init()
  screen = pygame.display.set_mode((1000, 1000))
  clock = pygame.time.Clock()
  pygame.time.set_timer(pygame.USEREVENT, 1500)
  CELLSIZE = 32
  H_CELLSIZE = CELLSIZE // 2
  HUMAN_X = 0
  HUMAN_Y = 30
  COMPUTER_X = 500
  COMPUTER_Y = 30
  PAUSE = 500
  def is_from_human(x, y):
    return x < COMPUTER_X
  def human_screen_to_indices(x, y):
    return ((y - HUMAN_Y) // CELLSIZE, (x - HUMAN_X) // CELLSIZE)
  def human_indices_to_screen(row, col):
    return (HUMAN_X + col * CELLSIZE + H_CELLSIZE, HUMAN_Y + row * CELLSIZE + H_CELLSIZE)
  def computer_screen_to_indices(x, y):
    return ((y - COMPUTER_Y) // CELLSIZE, (x - COMPUTER_X) // CELLSIZE)
  def computer_indices_to_screen(row, col):
    return (COMPUTER_X + col * CELLSIZE + H_CELLSIZE, COMPUTER_Y + row * CELLSIZE + H_CELLSIZE)
  def indices_to_screen(row, col, from_human):
    if from_human:
      return human_indices_to_screen(row, col)
    else:
      return computer_indices_to_screen(row, col)
  def draw_board(screen, board, from_human):
    n = len(board[0])
    for row in range(n):
      for col in range(n):
        coords = indices_to_screen(row, col, from_human)
        color = -1
        for i in range(len(utils.COLORS)):
          if board[i][row][col]:
            color = i
            break
        special = ''
        for i in range(len(utils.COLORS), len(utils.COLUMNS)):
          if board[i][row][col]:
            special = utils.COLUMNS[i]
            break
        if color != -1:
          pygame_color = pygame.color.Color(utils.COLORS[color])
          if special == '':
            pygame.draw.circle(screen, pygame_color, coords, H_CELLSIZE, 0)
          elif special == 'v_strip':
            pygame.draw.ellipse(screen, pygame_color, [coords[0] - H_CELLSIZE // 2, coords[1] - H_CELLSIZE, H_CELLSIZE, H_CELLSIZE * 2], 0)
          elif special == 'h_strip':
            pygame.draw.ellipse(screen, pygame_color, [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE // 2, H_CELLSIZE * 2, H_CELLSIZE], 0)
          else:
            pygame.draw.rect(screen, pygame_color, [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE, H_CELLSIZE, H_CELLSIZE], 0)
        elif special == 'color_bomb':
          pygame.draw.arc(screen, (0, 0, 0), [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE, H_CELLSIZE * 2, H_CELLSIZE * 2], 0, math.pi / 2, 2)
          pygame.draw.arc(screen, (0, 255, 0), [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE, H_CELLSIZE * 2, H_CELLSIZE * 2], math.pi / 2, math.pi, 2)
          pygame.draw.arc(screen, (0, 0, 255), [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE, H_CELLSIZE * 2, H_CELLSIZE * 2], math.pi, 3 * math.pi / 2, 2)
          pygame.draw.arc(screen, (255, 0, 0), [coords[0] - H_CELLSIZE, coords[1] - H_CELLSIZE, H_CELLSIZE * 2, H_CELLSIZE * 2], 3 * math.pi / 2, 2 * math.pi, 2)
  previous_row, previous_col = -1, -1
  draw_histories = True
  while True:
    if pygame.event.get(pygame.QUIT):
      break
    for e in pygame.event.get():
      if e.type == pygame.MOUSEBUTTONDOWN:
        x, y = pygame.mouse.get_pos()
        if is_from_human(x, y):
          row, col = human_screen_to_indices(x, y)
          print('Human clicked %d %d' % (row, col))
          if abs(previous_row - row) + abs(previous_col - col) == 1:
            print('Swapping (%d, %d) and (%d, %d)' % (row, col, previous_row, previous_col))
            human_board.swap((row, col), (previous_row, previous_col))
            new_reward = computer_board.ai_swap(method='brute force')
            print('New reward %d' % (new_reward,))
            draw_histories = True
            previous_row, previous_col = -1, -1
          else:
            previous_row, previous_col = row, col
    screen.fill(pygame.color.Color('white'))
    # Draw the histories.
    human_histories = human_board.get_histories()
    computer_histories = computer_board.get_histories()
    if draw_histories:
      time_to_start_draw_histories = pygame.time.get_ticks()
      time_to_stop_draw_histories = time_to_start_draw_histories + PAUSE * max(len(human_histories), len(computer_histories))
      draw_histories = False
    font = pygame.font.Font('freesansbold.ttf', CELLSIZE)
    human_text = font.render('Human score: %d' % (human_board.get_score(),), True, (0, 0, 0))
    human_text_rect = human_text.get_rect()
    human_text_rect.center = ((HUMAN_X + CELLSIZE * (SIZE // 2)), (HUMAN_Y + CELLSIZE * (SIZE + 2)))
    screen.blit(human_text, human_text_rect)
    computer_text = font.render('Computer score: %d' % (computer_board.get_score(),), True, (0, 0, 0))
    computer_text_rect = computer_text.get_rect()
    computer_text_rect.center = ((COMPUTER_X + CELLSIZE * (SIZE // 2)), (COMPUTER_Y + CELLSIZE * (SIZE + 2)))
    screen.blit(computer_text, computer_text_rect)
    move_text = font.render('Number of moves: %d' % (human_board.num_swaps(),), True, (0, 0, 0))
    move_text_rect = move_text.get_rect()
    move_text_rect.center = (COMPUTER_X, H_CELLSIZE)
    screen.blit(move_text, move_text_rect)
    if pygame.time.get_ticks() < time_to_stop_draw_histories:
      idx = (pygame.time.get_ticks() - time_to_start_draw_histories) // PAUSE
      draw_board(screen, human_histories[idx] if idx < len(human_histories) else human_board.get_board(), from_human=True)
      draw_board(screen, computer_histories[idx] if idx < len(computer_histories) else computer_board.get_board(), from_human=False)
      pygame.display.flip()
    else:
      # Draw the board.
      draw_board(screen, human_board.get_board(), from_human=True)
      draw_board(screen, computer_board.get_board(), from_human=False)
      pygame.display.flip()
    clock.tick(60)


if __name__ == '__main__':
  main()

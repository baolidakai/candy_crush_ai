"""Pygame controller for the game."""
import math
import os
import pygame
import sys
import utils
import candy_crush_board


def main():
  demo_base = '../config/demo/'
  config_file = os.path.join(demo_base, 'config1.txt') if len(sys.argv) < 2 else os.path.join(demo_base, sys.argv[1])
  print('Loading from %s' % (config_file,))
  AGENTS = ['human', 'brute force', 'dqn']
  NUM_BOARDS = len(AGENTS)
  # Gets human board and computer boards.
  boards = [candy_crush_board.CandyCrushBoard(config_file=config_file) for _ in range(NUM_BOARDS)]
  SIZE = boards[0].get_size()
  for board in boards:
    assert board.get_size() == SIZE
  for i in range(NUM_BOARDS):
    if AGENTS[i] == 'dqn':
      boards[i].init_dqn()
  # Renders the two boards.
  pygame.init()
  screen = pygame.display.set_mode((1200, 800))
  clock = pygame.time.Clock()
  pygame.time.set_timer(pygame.USEREVENT, 1500)
  CENTER_X = 600
  CELLSIZE = 28
  BOARD_LEN = 300
  H_CELLSIZE = CELLSIZE // 2
  PAUSE = 300
  def get_base_xy(idx):
    return BOARD_LEN * idx, 30
  def get_index(x, y):
    row = 0
    col = x // BOARD_LEN
    idx = row * 2 + col
    idx = max(idx, 0)
    idx = min(idx, NUM_BOARDS)
    return idx
  def screen_to_indices(x, y, idx):
    base_x, base_y = get_base_xy(idx)
    return ((y - base_y) // CELLSIZE, (x - base_x) // CELLSIZE)
  def human_screen_to_indices(x, y):
    return screen_to_indices(x, y, 0)
  def indices_to_screen(row, col, idx):
    base_x, base_y = get_base_xy(idx)
    return (base_x + col * CELLSIZE + H_CELLSIZE, base_y + row * CELLSIZE + H_CELLSIZE)
  def is_from_human(x, y):
    return get_index(x, y) == 0
  def draw_board(screen, board, idx):
    n = len(board[0])
    for row in range(n):
      for col in range(n):
        coords = indices_to_screen(row, col, idx)
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
            boards[0].swap((row, col), (previous_row, previous_col))
            # AI agents do the update.
            for i in range(1, NUM_BOARDS):
              agent_name = AGENTS[i]
              new_reward = boards[i].ai_swap(method=agent_name)
              print('New reward for %s is %d' % (agent_name, new_reward,))
            draw_histories = True
            previous_row, previous_col = -1, -1
          else:
            previous_row, previous_col = row, col
    screen.fill(pygame.color.Color('white'))
    # Draw the histories.
    human_histories = boards[0].get_histories()
    computer_histories = [boards[i].get_histories() for i in range(1, NUM_BOARDS)]
    if draw_histories:
      time_to_start_draw_histories = pygame.time.get_ticks()
      max_computer_histories = max(len(h) for h in computer_histories)
      time_to_stop_draw_histories = time_to_start_draw_histories + PAUSE * max(len(human_histories), max_computer_histories)
      draw_histories = False
    font = pygame.font.Font('freesansbold.ttf', CELLSIZE)
    human_text = font.render('Human score: %d' % (boards[0].get_score(),), True, (0, 0, 0))
    human_text_rect = human_text.get_rect()
    human_x, human_y = get_base_xy(0)
    human_text_rect.center = ((human_x + CELLSIZE * (SIZE // 2)), (human_y + CELLSIZE * (SIZE + 2)))
    screen.blit(human_text, human_text_rect)
    for i in range(1, NUM_BOARDS):
      computer_text = font.render('%s score: %d' % (AGENTS[i], boards[i].get_score(),), True, (0, 0, 0))
      computer_text_rect = computer_text.get_rect()
      computer_x, computer_y = get_base_xy(i)
      computer_text_rect.center = ((computer_x + CELLSIZE * (SIZE // 2)), (computer_y + CELLSIZE * (SIZE + 2)))
      screen.blit(computer_text, computer_text_rect)
    move_text = font.render('Number of moves: %d' % (boards[0].num_swaps(),), True, (0, 0, 0))
    move_text_rect = move_text.get_rect()
    move_text_rect.center = (CENTER_X, H_CELLSIZE)
    screen.blit(move_text, move_text_rect)
    if pygame.time.get_ticks() < time_to_stop_draw_histories:
      idx = (pygame.time.get_ticks() - time_to_start_draw_histories) // PAUSE
      draw_board(screen, human_histories[idx] if idx < len(human_histories) else boards[0].get_board(), 0)
      for i in range(1, NUM_BOARDS):
        draw_board(screen, computer_histories[i - 1][idx] if idx < len(computer_histories[i - 1]) else boards[i].get_board(), i)
      pygame.display.flip()
    else:
      # Draw the board.
      draw_board(screen, boards[0].get_board(), 0)
      for i in range(1, NUM_BOARDS):
        draw_board(screen, boards[i].get_board(), i)
      pygame.display.flip()
    clock.tick(60)


if __name__ == '__main__':
  main()


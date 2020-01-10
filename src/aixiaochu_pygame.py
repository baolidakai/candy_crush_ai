import os
import pygame
from pygame.locals import *
from pygame.compat import geterror
from aixiaochu import Aixiaochu

# Good reference: https://stackoverflow.com/questions/33163325/game-of-life-pygame
def main():
  pygame.init()
  screen = pygame.display.set_mode((1000, 1000))
  clock = pygame.time.Clock()
  # TODO(bowendeng): Make parameters adjustable some way.
  game = Aixiaochu(rows=20, cols=20, num_colors=5)
  pygame.time.set_timer(pygame.USEREVENT, 1500)
  CELLSIZE = 32
  H_CELLSIZE = CELLSIZE // 2
  def scr_to_board(pos):
    x, y = pos
    return (x // CELLSIZE, y // CELLSIZE)
  def board_to_scr(pos):
    x, y = pos
    return (x * CELLSIZE + H_CELLSIZE, y * CELLSIZE + H_CELLSIZE)
  while True:
    if pygame.event.get(pygame.QUIT):
      break
    for e in pygame.event.get():
      if e.type == pygame.MOUSEBUTTONDOWN:
        x, y = scr_to_board(pygame.mouse.get_pos())
        print('Clicked %d %d' % (x, y))
        game.click(y, x)
    screen.fill(pygame.color.Color('white'))
    # List of (row, col, color).
    cells = game.get_cells()
    for row, col, color in cells:
      pygame.draw.circle(screen, pygame.color.Color(color), board_to_scr((col, row)), H_CELLSIZE, 0)
    pygame.display.flip()
    clock.tick(60)


if __name__ == '__main__':
  main()


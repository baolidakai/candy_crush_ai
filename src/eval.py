"""Evaluation of different algorithms."""
import candy_crush_board
import os
import utils

eval_base = '../config/eval'
config_file = os.path.join(eval_base, 'config1.txt')
agents = utils.AI_AGENTS[:]
num_boards = len(agents)
boards = [candy_crush_board.CandyCrushBoard(config_file=config_file) for _ in range(num_boards)]
num_moves = 40

rewards = []
for i in range(num_boards):
  agent_name = agents[i]
  for b in range(num_moves):
    boards[i].ai_swap(method=agent_name)
  rewards.append(boards[i].get_score())

for i in range(num_boards):
  print('Reward for %s: %f' % (agents[i], rewards[i]))


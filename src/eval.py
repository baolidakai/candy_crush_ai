"""Evaluation of different algorithms."""
import candy_crush_board
import os
import utils

eval_base = '../config/eval'
agents = utils.AI_AGENTS[:]

num_boards = len(agents)
rewards = [0.0 for _ in range(num_boards)]

NUM_EVAL = 10
for num_eval in range(NUM_EVAL):
  config_file = os.path.join(eval_base, 'config%d.txt' % (num_eval + 1))
  boards = [candy_crush_board.CandyCrushBoard(config_file=config_file) for _ in range(num_boards)]
  num_moves = 40

  for i in range(num_boards):
    agent_name = agents[i]
    for b in range(num_moves):
      boards[i].ai_swap(method=agent_name)
    rewards[i] += boards[i].get_score()

  print('Evaluation round %d out of %d' % (num_eval, NUM_EVAL))
  for agent, reward in zip(agents, rewards):
    print('Reward for %s: %f' % (agent, reward / float(num_eval + 1)))

for i in range(num_boards):
  rewards[i] /= float(NUM_EVAL)

with open('eval.txt', 'w') as fout:
  for i in range(num_boards):
    fout.write('Reward for %s: %f' % (agents[i], rewards[i]))
    fout.write('\n')


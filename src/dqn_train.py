import candy_crush_board
import collections
import itertools
import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import sys
from PIL import Image
from dqn_base import DQN
import pdb
from dqn_viz import make_dot


device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
  def __init__(self, capacity):
    self._capacity = capacity
    self._memory = []
    self._position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self._memory) < self._capacity:
      self._memory.append(None)
    self._memory[self._position] = Transition(*args)
    self._position = (self._position + 1) % self._capacity

  def sample(self, batch_size):
    return random.sample(self._memory, batch_size)

  def __len__(self):
    return len(self._memory)


BATCH_SIZE = 5
GAMMA = 0.99
# 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

board = candy_crush_board.CandyCrushBoard(config_file='../config/train/config1.txt')

def get_state(board):
  raw_state = board.get_numpy_board()
  raw_state = np.ascontiguousarray(raw_state, dtype=np.float32)
  raw_state = torch.from_numpy(raw_state)
  return raw_state.unsqueeze(0).to(device)

init_screen = get_state(board)
_, _, screen_height, screen_width = init_screen.shape

actions = board.get_actions()
n_actions = len(actions)

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def select_action(state):
  global steps_done
  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
  steps_done += 1
  if sample > eps_threshold:
    with torch.no_grad():
      return policy_net(state).max(1)[1].view(1, 1)
  else:
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

visualized = False
loss_value = 0

def optimize_model():
  if len(memory) < BATCH_SIZE:
    return
  transitions = memory.sample(BATCH_SIZE)
  # Transpose the batch.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements.
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Visualize code.
  global visualized
  if not visualized:
    y = policy_net(state_batch)
    g = make_dot(y, policy_net.state_dict())
    g.view()
    visualized = True

  # Compute Q(s_t, a) - the model computes Q(s_t),
  # then we select columns of actions taken.
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t + 1}) for all next states.
  next_state_values = torch.zeros(BATCH_SIZE, device=device)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
  # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
  # print('loss:', loss.item())
  global loss_value
  loss_value = loss.item()

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  for param in policy_net.parameters():
    param.grad.data.clamp_(-1, 1)
  optimizer.step()


print('Example usage: python3 dqn_train.py 100 where 100 is number of episodes.')
num_episodes = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
monte_carlo_B = 10
USE_MONTE_CARLO = False
NUM_CONFIGS = 100
start_time = time.time()
average_losses = []
for i_episode in range(num_episodes):
  config = 1 + (i_episode % NUM_CONFIGS)
  board = candy_crush_board.CandyCrushBoard(config_file='../config/train/config%d.txt' % (config,))
  board.set_monte_carlo_B(monte_carlo_B)
  state = get_state(board)
  print('Episode %d' % (i_episode,))
  print('Elapsed time %d seconds' % (time.time() - start_time,))
  print('Average time %f seconds' % (float(time.time() - start_time) / float(i_episode) if i_episode else float('inf'),))
  total_loss = 0.0
  total_count = 0
  for t in itertools.count():
    action = select_action(state)
    action_index = action.item()
    # Simulated reward by Monte Carlo.
    monte_carlo_reward = board.get_monte_carlo_score(action_index)
    # Actual reward by taking the action.
    naive_reward = board.step(action_index)
    raw_reward = monte_carlo_reward if USE_MONTE_CARLO else float(naive_reward)
    reward = torch.tensor([raw_reward], device=device)
    done = board.is_done()

    # Observe new state
    next_state = get_state(board) if not done else None

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization
    optimize_model()
    total_loss += loss_value
    total_count += 1

    if done:
      episode_durations.append(t + 1)
      average_loss = total_loss / total_count if total_count else float('inf')
      average_losses.append(average_loss)
      print('average loss:', average_loss)
      break
  # Update the target network, copying all weights and biases in DQN
  if i_episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())

print('Complete')

target_net_path = 'monte_carlo_target_net.pt' if USE_MONTE_CARLO else 'naive_target_net.pt'
policy_net_path = 'monte_carlo_policy_net.pt' if USE_MONTE_CARLO else 'naive_policy_net.pt'
torch.save(target_net.state_dict(), device_type + '_' + target_net_path)
torch.save(policy_net.state_dict(), device_type + '_' + policy_net_path)

# Write average losses to file.
with open('average_losses.txt', 'w') as fout:
  for average_loss in average_losses:
    fout.write(str(average_loss) + '\n')


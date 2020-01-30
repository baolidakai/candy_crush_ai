# Inference for DQN.

import candy_crush_board
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resize = T.Compose([T.ToPILImage(),
  T.Resize(40, interpolation=Image.CUBIC),
  T.ToTensor()])

board = candy_crush_board.CandyCrushBoard()

def get_state(board):
  raw_state = board.get_numpy_board()
  raw_state = np.ascontiguousarray(raw_state, dtype=np.float32)
  raw_state = torch.from_numpy(raw_state)
  return resize(raw_state).unsqueeze(0).to(device)

class DQN(nn.Module):
  def __init__(self, h, w, outputs):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)

    def conv2d_size_out(size, kernel_size=5, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1
    
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
    linear_input_size = convw * convh * 32
    self.head = nn.Linear(linear_input_size, outputs)

  # Called with either one element to determine next action, or a batch during optimization.
  # Returns tensor([[left0exp, right0exp]...]).
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))

init_screen = get_state(board)
_, _, screen_height, screen_width = init_screen.shape
actions = board.get_actions()
n_actions = len(actions)

target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(torch.load('target_net'))
target_net.eval()

board.reset()
state = get_state(board)
print(target_net(state))



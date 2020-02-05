Code structures:
utils.py - Utility functions and constants shared by all other files.
candy\_crush\_game.py - Game controller.
config\_generator.py - Generator for a random configuration.
candy\_crush\_board.py - Main object of the game, represents a game state, and can be interacted with.

To generate a random configuration:
```
python config_generator.py 200 10 5 config1.txt
```

TODOs:
* [P1] Check the performance of Monte Carlo DQN
* [P2] other candy types

How to featurize: raw pixel vs my own representation
Which model use

consider good randomization technique

Some other models:
policy gradients (DDPG)

Either try different architectures and do one experiment
Or try existing architectures and more analysis


Evaluation:
Reward for naive dqn: 349
Reward for monte carlo: 702
Reward for monte carlo dqn: 349
Reward for brute force: 157


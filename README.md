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
* other candy types
* Side by side with brute force
* Implements Monte Carlo agent

How to featurize: raw pixel vs my own representation
Which model use

consider good randomization technique

Some other models:
policy gradients (DDPG)

Either try different architectures and do one experiment
Or try existing architectures and more analysis



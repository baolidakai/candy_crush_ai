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
* [P1] Have an automatic eval pipeline
* [P1] How to beat or match Monte Carlo
* [P2] other candy types

How to featurize: raw pixel vs my own representation
Which model use

consider good randomization technique

Some other models:
policy gradients (DDPG)

Either try different architectures and do one experiment
Or try existing architectures and more analysis



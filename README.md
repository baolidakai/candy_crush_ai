Code structures:
utils.py - Utility functions and constants shared by all other files.
candy\_crush\_game.py - Game controller.
config\_generator.py - Generator for a random configuration.
candy\_crush\_board.py - Main object of the game, represents a game state, and can be interacted with.

To generate a random configuration:
```
python config_generator.py 200 10 5 config1.txt
```

TODO:
* Concentrate on instant reward training.
* Adds a tensorboard: https://pytorch.org/tutorials/intermediate/tensorboard\_tutorial.html
* Add more layers.
* Consider permutation invariance: add a loss on swapping two channels.
* Plot average loss.
* Train a new naive model using GPU.
* [P1] Cache Monte Carlo eval.


Evaluation:
Reward for naive dqn: 349
Reward for monte carlo: 702
Reward for brute force: 157

Notes:
Consider randomized training.
Try to at least match Monte Carlo.
Investigate reason Monte Carlo DQN is worse.
Milestone does not need result to be good, but best to have an explanation.
Milestone: write a next step section, instead of leaving TODOs
fine to dig deeper without look at other models.
could consider learn from me.

Test

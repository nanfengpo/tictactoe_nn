# tictactoe_nn

The AI is implemented using Convolutional NN and is trained with 
the [Deep Q-Learning algorithm](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

## How to use

1. Add your own local configuration file at `playground/_train_config.py`
2. Extend the `tictactoe_nn.config.TrainConfig` class and define options you need.
See the `TrainConfig` for available options.
3. I recommend you to install the `fflib` - apart of other things it provides the tool
for quickly running scripts from the `playground` directory.  
`pip install https://github.com/faddey-w/fflib/archive/master.zip`
4. Run the training script. If you've installed the fflib, simply type `play run`, 
otherwise - `python3 -m playground.run`

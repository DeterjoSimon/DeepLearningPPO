# DeepLearningPPO
PPO implementation for a project in the course Deep Learning (02456). This implementation runs through the IMPALA framework and tries to learn to play the game "Starpilot" offered by the Procgen Benchmark in order to evaluate the generalization capability of an agent linked to Reinforcement Learning.

The Project resolves around an object-oriented vision where in train.py all the important classes used in the DRL algorithm are initialised one by one. These classes are respectively:

## The PPO agent (ppo.py)
This is where the PPO magic happens, in this class the agent is able to train and to evaluate itself on the respective triaining and evaluation environments it has been initialised with. The training method comprises of a predict and evaluation method. The latter uses the evaluation method to see its own performance once released in an evaluation environment. The training shares its data with a Logger to report relevant informations to be saved. At the end of the agent's training, a checkpoint is made where our model is saved for either further training or seperate video evaluation.

## The deep neural network model (model.py)
This is where the IMPALA architecture is built and lays. The overall structure of the encoder is provided in our report, explaining the architecture with all of its parts and layers. In our paper, the encoder consists of an input going into an ”IMPALA block”.   This  IMAPALA  block  consists  of  a  convolutional  network, this is the maxpooled, and followed with two residual blocks. A residual block consists of ReLU, a convolution network, another ReLU and another convolutional network. The IMPALA block is repeated three times, followed by a ReLU, a fully connected linear layer, another ReLU and lastly entering a gated-recurrent unit (GRU).

## A Storage (storage.py)
As provided in the first lab code by Nicklas Hansen (our TA), to ensure efficient data handling a storage class should be created. The storage class handles all the enqueued rollouts and processes them for optimisation (data gets sampled along the time-horizon if agent's policy is recurrent). 

## A Logger (logger.py)
In order to save our training and evaluation we use a logger class which collects summaries of the whole RL process. Thanks to SummaryWriter package from the tensorboard library it becomes quite easy to report our experiments and save them in a csv file during training. Everytime the program is ran, a log is saved in the current directory. Even if the program crashes information is still saved from the prior experiments.

## Categorical Policy (policy.py)
This is can be seen as the actor-critic interface where decisions are made. This is where the model is initialised through orthogonal initialisation. The probability distribution sampling is brought by the forward process combined with the hidden states and the value function estimate.

## Miscellaneous utility files (utils.py and misc_utils.py)
Definition of orthogonal initialisation plus the appropriate attari wrappers from procgen when defining environments. 

## Training the agent (train.py)
All our tools for applying PPO for RL is pipelined through this script which is boiled down as the initialisation of all the neural-networks hyperparameters, the environment intialisation finally the initialisation of the classes stated above.

Thank you to our supervisors help and our teacher for letting us work on this project. We have learnt so much about deep reinforcement learning and could not have imagined how much this subject still has much to offer!

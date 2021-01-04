from utils import *
from common.logger import Logger
from common.storage import Storage
from common.model import ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from agent.ppo import PPO
import os, time
import gym
from procgen import ProcgenEnv
import random
import torch

if __name__ == '__main__':
    exp_name = 'test'
    env_name = 'starpilot'
    start_level = int(0)
    num_levels = int(50)
    distribution_mode = 'easy'
    num_timesteps = int(20e6)
    seed = random.randint(0, 9999)
    log_level = int(40)
    num_checkpoints = int(1)
    eps = .2
    learning_rate = 0.0005
    grad_eps = .5
    value_coef = .5
    entropy_coef = .01
    mini_batch_per_epoch = 8
    batch_size = 2048
    gamma = 0.999
    lmdba = 0.95
    num_epochs = 3

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ############
    ## DEVICE ##
    ############
    device = torch.device('cuda')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = 256
    n_envs = 64
    torch.set_num_threads(1)
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode,
                     use_backgrounds=False,
                     restrict_themes=True
                     )
    normalize_rew = True
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames.
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    v_env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode,
                     use_backgrounds=False,
                     restrict_themes=True
                     )
    v_env = VecExtractDictObs(v_env, "rgb")
    if normalize_rew:
        v_env = VecNormalize(v_env, ob=False)
    v_env = TransposeFrame(v_env)
    v_env = ScaledFloatFrame(v_env)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = 'impala'
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    model = NatureModel(in_channels=in_channels)
    recurrent = True
    action_size = action_space.n
    policy = CategoricalPolicy(model, recurrent, action_size)
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    agent = PPO(env, v_env, policy, logger, storage, device, num_checkpoints, n_steps,
                n_envs, num_epochs, mini_batch_per_epoch=mini_batch_per_epoch, mini_batch_size=batch_size,
                gamma=gamma, lmbda=lmdba, learning_rate=learning_rate, grad_clip_norm=grad_eps,
                eps_clip=eps, value_coef=value_coef, entropy_coef=entropy_coef,
                normalize_adv=True, use_gae=True)

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)

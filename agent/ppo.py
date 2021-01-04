from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
import numpy as np


class PPO(object):
    def __init__(self,
                 env,
                 v_env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 use_gae=True,
                 **kwargs):

        self.env = env
        self.v_env = v_env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = n_checkpoints

        self.t = 0
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        """
        Actor predicting action he will take based on his policy.
        :param obs: environment observation as an array
        :param hidden_state: Information flow for the GRU layer
        :param done: array informing of the state of an agent in an environment
        :return: sampled action from the policy; log_prob from this policy; value estimate; hidden_states
        """
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def action(self, obs):
        """
        Function only used in evaluation.
        :param obs: Only requires observation from the environment to make predictions since no backward pass is done.
        :return: action sampled; log_prob distribution; value estimate; hidden_state (which can be ignored).
        """
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        self.policy.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu()

    def optimize(self):
        """
        Application of PPO optimisation as described in report, applying batch generation from IMPALA.
        :return: Summary of the surrogate objective loss
        """
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1 - done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip,
                                                                                              self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def eval(self):
        """
        Similar to train, but only collects relevant information to evaluate the reward of our agent in an evaluation
        environment.
        :return: Summary of reward for the Logger
        """
        info_batch = []
        episode_rewards = []
        eval_reward = torch.zeros(self.n_steps, self.n_envs)
        ep_done = torch.zeros(self.n_steps, self.n_envs)
        episode_reward_buffer = []
        eval_obs = self.v_env.reset()
        for _ in range(self.n_envs):
            episode_rewards.append([])
        self.policy.eval()
        for step in range(self.n_steps):
            eval_act, _, _, eval_next_hidden_state = self.action(eval_obs)
            eval_next_obs, eval_rew, eval_done, info = self.env.step(eval_act)
            ep_done[step] = torch.from_numpy(eval_done.copy())
            eval_reward[step] = torch.from_numpy(eval_rew.copy())
            info_batch.append(info)
            eval_obs = eval_next_obs

        if 'env_reward' in info_batch[0][0]:
            rew_batch = []
            for step in range(self.n_steps):
                infos = info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = eval_reward.numpy()
        if 'env_done' in info_batch[0][0]:
            done_batch = []
            for step in range(self.n_steps):
                infos = info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = ep_done.numpy()

        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T
        for i in range(self.n_envs):
            for j in range(steps):
                episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    episode_reward_buffer.append(np.sum(episode_rewards[i]))
                    episode_rewards[i] = []
        return episode_reward_buffer, np.max(episode_reward_buffer)

    def train(self, num_timesteps):
        """
        Train the agent by running the policy, computing advantage estimates from the value function,
        optimize the policy and value network, logging the training-procedure and finally evaluation the model.
        :param num_timesteps: Number of timesteps for running the training
        """
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & value
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            # Evaluate
            eval_batch, eval_max = self.eval()
            self.logger.feed(rew_batch, done_batch, eval_batch, eval_max)
            self.logger.write_summary(summary)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
                           '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()

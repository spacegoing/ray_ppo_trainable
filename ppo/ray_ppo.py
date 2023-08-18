from ray import tune

import os
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
# logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
# RL Related
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.env.venv_wrappers import VectorEnvNormObs
from ppo_policy import PPOPolicy
from mujoco_env import make_mujoco_env
from utils_rl import ConfigDict, load_config
from ray_trainer import OnpolicyTrainer


class Policy(tune.Trainable):

  def setup(self, config):
    # with reset_config() implemented, this sill only be called at the very start/first
    # execution of each trial_id/name/worker.
    # Even after done=True by step(), e.g., after max_epoch is reached, this will NOT be called.
    self.config = ConfigDict(config)
    self.setup_env_nn_optim_policy_buffer_collector(self.config)

    # MISC
    self.topk_suffix = 0

  def setup_env_nn_optim_policy_buffer_collector(self, config):
    # will be called after max_epoch is reached. aka, after step() return done=True.
    config = ConfigDict(config)
    self.config = config
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    #* Env; No inter epoch updates of training env nums
    self.env, self.train_envs, self.test_envs = make_mujoco_env(
        config.env, config.seed, config.num_envs_per_worker,
        config.eval_num_envs_per_worker, config.obs_norm)
    config.state_shape = self.env.observation_space.shape or self.env.observation_space.n
    config.action_shape = self.env.action_space.shape or self.env.action_space.n

    #* NN: No inter epoch updates of NN structure
    net_a = Net(
        config.state_shape,
        hidden_sizes=config.hidden_sizes,
        activation=nn.Tanh,
        device=config.device,
    )
    actor = ActorProb(
        net_a,
        config.action_shape,
        unbounded=True,
        device=config.device,
    ).to(config.device)
    net_c = Net(
        config.state_shape,
        hidden_sizes=config.hidden_sizes,
        activation=nn.Tanh,
        device=config.device,
    )
    critic = Critic(net_c, device=config.device).to(config.device)
    actor_critic = ActorCritic(actor, critic)

    # initialization
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
      if isinstance(m, torch.nn.Linear):
        # orthogonal initialization
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
      if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.weight.data.copy_(0.01 * m.weight.data)

    #* optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=config.lr)

    # todo: what to do with scheduler?
    lr_scheduler = None
    if config.lr_decay:
      # decay learning rate to 0 linearly
      max_update_num = np.ceil(
          config.step_per_epoch / config.step_per_collect) * config.max_epoch
      lr_scheduler = LambdaLR(
          optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    #* Policy
    def dist(*logits):
      return Independent(Normal(*logits), 1)

    self.policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=config.gamma,
        gae_lambda=config.gae_lambda,
        max_grad_norm=config.max_grad_norm,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        reward_normalization=config.rew_norm,
        action_scaling=config.action_scaling,
        action_bound_method=config.bound_action_method,
        lr_scheduler=lr_scheduler, # not from config
        action_space=self.env.action_space, # not from config
        eps_clip=config.eps_clip,
        value_clip=config.value_clip,
        dual_clip=config.dual_clip,
        advantage_normalization=config.norm_adv,
        recompute_advantage=config.recompute_adv,
    )

    #* collector
    buffer = VectorReplayBuffer(config.buffer_size, len(self.train_envs))
    self.train_collector = Collector(
        self.policy, self.train_envs, buffer, exploration_noise=True)
    self.test_collector = Collector(self.policy, self.test_envs)

    #* Top K Saving
    # Define size of top_k array; Initialize with negative infinity
    self.top_k = np.zeros(self.config.saving_top_k) - float("inf")

    #* Logger
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    self.config.algo_name = "ppo"
    log_name = os.path.join(self.config.env, self.config.algo_name,
                            'seed' + str(self.config.seed), now)
    self.log_path = os.path.join(self.config.logdir, log_name)

    # logger
    writer = SummaryWriter(self.log_path)
    writer.add_text("config", str(self.config))
    self.logger = TensorboardLogger(writer)

  def step(self):
    trainer = OnpolicyTrainer(
        self.policy,
        self.train_collector,
        test_collector=None,
        step_per_collect=self.config.step_per_collect,
        step_per_epoch=self.config.step_per_epoch,
        max_epoch=1,
        repeat_per_collect=self.config.repeat_per_collect,
        batch_size=self.config.minibatch_size,
        episode_per_test=self.config.eval_num_envs_per_worker,
        logger=self.logger,
        test_in_train=False,
        show_progress=False)

    trainer.run()
    train_rew_mean = trainer.this_epoch_eprews_mean
    train_rew_std = trainer.this_epoch_eprews_std
    self.rew_mean = train_rew_mean
    self.rew_std = train_rew_std

    #* Test
    if self.training_iteration > self.config.test_burn_in_time_attr:
      self.policy.eval()
      self.test_envs.seed(self.config.seed)
      self.test_collector.reset()
      test_result = self.test_collector.collect(
          n_episode=self.config.eval_num_envs_per_worker,
          render=self.config.render)
      self.rew_mean = test_result['rew']
      self.rew_std = test_result['rew_std']

      self.policy.train()

    #* Finish With Flags: checkpoint / done

    # Check if current metric is among top K metrics
    should_checkpoint = False
    min_top_k = np.min(self.top_k)
    if self.rew_mean > min_top_k:
      # Replace the smallest top_k value with the new metric
      self.top_k[np.argmin(self.top_k)] = self.rew_mean
      should_checkpoint = True

    checkpoint_score = self.rew_mean
    done = False if self.training_iteration < self.config.max_epoch - 1 else True
    return {
        'reward_mean': self.rew_mean,
        'reward_std': self.rew_std,
        'done': done,
        'should_checkpoint': should_checkpoint,
        'checkpoint_score': checkpoint_score,
        # 'policy': self.policy,
        # 'test_envs': self.test_envs
    }

  def reset_config(self, new_config):
    new_config = ConfigDict(new_config)
    # update old self.config to new config
    # for pbt, this takes no effect since pbt interally set_config in _exploit()
    self.config = new_config

    # with TuneConfig{reuse_actor==True}, tune will reuse everything in setup
    # even after step() return done=True. Therefore, has to mannually recreate
    # everything after each max_epoch is reached.
    if self.training_iteration == self.config['max_epoch']:
      self.setup_env_nn_optim_policy_buffer_collector(self.config)
      # no need for changing configs below, return directly
      return True

    #* from ppo_policy.__init__:
    self.policy._lambda = new_config.gae_lambda
    self.policy._weight_vf = new_config.vf_coef
    self.policy._weight_ent = new_config.ent_coef
    self.policy._grad_norm = new_config.max_grad_norm

    self.policy._norm_adv = new_config.norm_adv
    # self.policy._eps_clip = new_config.eps_clip
    # self.policy._dual_clip = new_config.dual_clip
    # self.policy._value_clip = new_config.value_clip
    # self.policy._recompute_adv = new_config.recompute_advantage

    # PGPolicy
    self.policy._gamma = new_config.gamma
    self.policy._rew_norm = new_config.rew_norm
    # self.policy.optim = new_config.optim
    # self.policy.dist_fn = new_config.dist_fn
    # self.policy._deterministic_eval = new_config.deterministic_eval

    # BasePolicy
    # self.observation_space = new_config.observation_space
    # self.action_scaling = new_config.action_scaling
    # can be one of ("clip", "tanh", ""), empty string means no bounding
    # self.action_bound_method = new_config.action_bound_method
    # self.lr_scheduler = new_config.lr_scheduler

    return True

  def save_checkpoint(self, checkpoint_dir):
    '''
    Without further config, checkpoint_dir is ~/ray_results/experiments/checkpoint_0000011
    Either return the ~path~ or a ~dict~ containing parameters
    '''
    save_path = os.path.join(
        checkpoint_dir, f"_rew_{int(self.rew_mean)}_std_{int(self.rew_std)}",
        "policy.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {"model": self.policy.state_dict()}
    if isinstance(self.train_envs, VectorEnvNormObs):
      state.update({"obs_rms": self.train_envs.get_obs_rms()})
    torch.save(state, save_path)

    return save_path

  def load_checkpoint(self, checkpoint):
    '''
    checkpoint is different from save_checkpoint's return path's prefix path.
    this is fine. ray use tmp path for store intermediate checkpoints
    '''
    ckpt = torch.load(checkpoint)
    self.policy.load_state_dict(ckpt["model"])
    if "obs_rms" in ckpt:
      self.train_envs.set_obs_rms(ckpt["obs_rms"])
      self.test_envs.set_obs_rms(ckpt["obs_rms"])

if __name__ == '__main__':
  config = load_config('../ts_ppo_config.yaml')
  config = ConfigDict(config)

  doe = Policy(config)
  for i in range(config.max_epoch):
    result = doe.step()
    print(
        f"{i}th Epoch: reward={result['reward_mean']}, std={result['reward_std']}"
    )
    # if result['should_checkpoint']:
    #   doe.save('./')

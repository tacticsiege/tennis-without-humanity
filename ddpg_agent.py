import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic
from experience_buffers import ReplayBuffer
from ou_noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

# default parameter values
DEFAULT_NAME = 'default_ddpg'
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

LAYERS_ACTOR = [128, 128]
LAYERS_CRITIC = [128, 128]
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0.0001   # L2 weight decay

LEARN_EVERY = 100       # timesteps between learning
LEARN_PASSES = 1        # number of passes each learning update

default_params = {
    'name':DEFAULT_NAME,
    'buffer_size':BUFFER_SIZE,
    'batch_size':BATCH_SIZE,
    'layers_actor':LAYERS_ACTOR,
    'layers_critic':LAYERS_CRITIC,
    'lr_actor':LR_ACTOR,
    'lr_critic':LR_CRITIC,
    'gamma':GAMMA,
    'tau':TAU,
    'weight_decay':WEIGHT_DECAY,
    'learn_every':LEARN_EVERY,
    'learn_passes':LEARN_PASSES
}

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, brain_name, seed, 
                params=default_params,
                device=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # save params
        params = self._fill_params(params)
        self.params = params

        # implementation and identity
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = params['name']
        self.brain_name = brain_name

        # set environment information
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # create actor and critic local and target networks
        # Actor (policy) Network w/ Target
        self.actor_local = Actor(
            state_size, action_size, seed,
            params['layers_actor'], batch_norm=params['batch_norm']
        ).to(self.device)

        self.actor_target = Actor(
            state_size, action_size, seed,
            params['layers_actor'], batch_norm=params['batch_norm']
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=params['lr_actor'])

        # Critic (action-value) Network w/ Target
        self.critic_local = Critic(
            state_size, action_size, seed,
            params['layers_critic'], batch_norm=params['batch_norm']
        ).to(self.device)

        self.critic_target = Critic(
            state_size, action_size, seed,
            params['layers_critic'], batch_norm=params['batch_norm']
        ).to(self.device)

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=params['lr_critic'],
            weight_decay=params['weight_decay'])

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(params['buffer_size'], params['batch_size'], 
                                    seed, actions_continuous=True, device=self.device)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory."""
        self.memory.add(state, action, reward, next_state, done)

    def start_learn(self, t):
        if not t % self.params['learn_every'] == 0:
            return
        if len(self.memory) <= self.params['batch_size']:
            return
        
        # do learning passes
        for _ in range(self.params['learn_passes']):
            experiences = self.memory.sample()
            self.learn(experiences, self.params['gamma'])

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute and minimize critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.params['tau'])
        self.soft_update(self.actor_local, self.actor_target, self.params['tau'])                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _fill_params(self, src_params):
        params = {
            'name': self._get_param_or_default('name', src_params, default_params),            
            'buffer_size':self._get_param_or_default('buffer_size', src_params, default_params),
            'batch_size':self._get_param_or_default('batch_size', src_params, default_params),
            'layers_actor':self._get_param_or_default('layers_actor', src_params, default_params),
            'layers_critic':self._get_param_or_default('layers_critic', src_params, default_params),
            'lr_actor':self._get_param_or_default('lr_actor', src_params, default_params),
            'lr_critic':self._get_param_or_default('lr_critic', src_params, default_params),
            'gamma':self._get_param_or_default('gamma', src_params, default_params),
            'tau':self._get_param_or_default('tau', src_params, default_params),
            'weight_decay':self._get_param_or_default('weight_decay', src_params, default_params),
            'learn_every':self._get_param_or_default('learn_every', src_params, default_params),
            'learn_passes':self._get_param_or_default('learn_passes', src_params, default_params),
            'batch_norm':self._get_param_or_default('batch_norm', src_params, default_params),
        }
        return params

    def display_params(self, force_print=False):
        if force_print:
            print(self.params)
        return self.params
    
    def _get_param_or_default(self, key, src_params, default_params):
        if key in src_params:
            return src_params[key]
        else:
            return default_params[key]

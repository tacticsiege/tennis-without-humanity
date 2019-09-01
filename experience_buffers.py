import numpy as np
import random
from collections import namedtuple, deque

import torch


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed,
                actions_continuous=False, device=None):
        """ Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            actions_continuous: type(actions) == float if True, long if False
        """
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.actions_continuous = actions_continuous
    
    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """ Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return self._encode_exp(experiences)

    def sample_index(self, idxes):
        """ Samples the experiences for the given indicies """
        experiences = []
        for i in idxes:
            experiences.append(self.memory[i])
        
        return self._encode_exp(experiences)

    def make_index(self, batch_size):
        """ Create a random index into the memory for sampling. """
        return [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [len(self.memory - 1 - i) for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def _encode_exp(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        if self.actions_continuous:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        else:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Return the current size of internal memory."""
        return len(self.memory)
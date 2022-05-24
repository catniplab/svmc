"""
Abstract classes
"""
from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module


class Filter(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, *args):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Dynamics(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self):
        """Sample a particle passing through the dynamical model"""
        pass

    @abstractmethod
    def log_weight(self, prev, curr):
        """
        :param prev: previous state
        :param curr: current state
        :return:
        """
        pass

    def forward(self, prev, curr):
        return self.log_weight(prev, curr)


class Proposal(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

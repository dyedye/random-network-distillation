from abc import ABCMeta, abstractmethod
from tensorboardX import SummaryWriter


class BaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(reward: int, global_step: int, step_in_episode: int):
        pass


class DummyLogger(BaseLogger):
    def __init__(self):
        pass


class TFLogger(BaseLogger):
    def __init__(self):
        self.writer = SummaryWriter()

from abc import ABCMeta, abstractmethod


class BaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(reward: int, global_step: int, step_in_episode: int):
        pass


class DummyLogger(BaseLogger):
    def __init__(self):
        pass

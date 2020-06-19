from ensm.agents import AgentSubPopulation
from ensm.games import GamesNetwork
from typing import List


class Norm(object):
    NORM_COUNT = 0

    def __init__(self, context, action, sanction):
        self.__context = context
        self.__action = action
        self.__sanction = sanction

        self.NORM_COUNT += 1
        self.__id = self.NORM_COUNT

    @property
    def context(self):
        return self.__context

    @property
    def action(self):
        return self.__action

    @property
    def sanction(self):
        return self.__sanction

    def __str__(self):
        return '{}: ({}) -> {} / {}'.format(self.__id, self.__context, self.__action, self.__sanction)

    def __eq__(self, other):
        return self.__context == other.context and self.action == other.action

    def __hash__(self):
        return hash(repr(self.context + self.action))


class NormReplicator(object):
    def __init__(self):
        pass

    def replicate(self, sub_population: List[AgentSubPopulation], games_net: GamesNetwork):
        pass

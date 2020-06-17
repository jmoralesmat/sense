from ensm.agents import AgentSubPopulation
from ensm.games import GamesNetwork
from typing import List


class MAS(object):
    def __init__(self, games_net: GamesNetwork, population: List[AgentSubPopulation]):
        """

        :param games_net: list of games to be played by the agents of the population
        :param population: list of AgentSubPopulation, each with a frequency
        """
        self.__population = population
        self.__games_net = games_net

    @property
    def games_net(self):
        return self.__games_net

    @property
    def population(self):
        return self.__population

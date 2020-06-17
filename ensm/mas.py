from ensm.agents import AgentSubPopulation
from ensm.games import Game
from typing import List


class MAS(object):
    def __init__(self, games: List[Game], population: List[AgentSubPopulation]):
        """

        :param games: list of games to be played by the agents of the population
        :param population: list of AgentSubPopulation, each with a frequency
        """
        self.__population = population
        self.__games = games

    @property
    def games(self):
        return self.__games

    @property
    def population(self):
        return self.__population

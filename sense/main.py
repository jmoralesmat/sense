from ensm.agents import AgentSubPopulation
from ensm.games import Game
from ensm.ensm import ENSM
from ensm.mas import MAS

from collections import defaultdict
import logging

MAX_GENERATIONS = 100


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    games = __create_games()
    population = __create_population(games)

    mas = MAS(games=games, population=population)
    ensm = ENSM(mas=mas, max_generations=MAX_GENERATIONS)

    while not ensm.converged:
        fitnesses = ensm.evolve()
        logger.info(fitnesses)


def __create_games():

    # Prisoner's Dilemma
    contexts = ['player(one)', 'player(two)']
    utilities = {
        ('C', 'C'): 1,
        ('C', 'D'): 0.5,
        ('D', 'C'): 0.5,
        ('D', 'D'): 0,
    }

    pd = Game(name='Prisoner Dilemma', contexts=contexts, utilities=utilities)
    return [pd]


def __create_population(games):

    pd = games[0]

    coop_payoffs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    coop_payoffs[pd][('C', 'C')][0] = 3
    coop_payoffs[pd][('C', 'C')][1] = 3
    coop_payoffs[pd][('C', 'D')][0] = 3
    coop_payoffs[pd][('C', 'D')][1] = 1
    coop_payoffs[pd][('D', 'C')][0] = 1
    coop_payoffs[pd][('D', 'C')][1] = 3
    coop_payoffs[pd][('D', 'D')][0] = 1
    coop_payoffs[pd][('D', 'D')][1] = 1

    def_payoffs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    def_payoffs[pd][('C', 'C')][0] = 3
    def_payoffs[pd][('C', 'C')][1] = 3
    def_payoffs[pd][('C', 'D')][0] = 1
    def_payoffs[pd][('C', 'D')][1] = 1
    def_payoffs[pd][('D', 'C')][0] = 5
    def_payoffs[pd][('D', 'C')][1] = 2
    def_payoffs[pd][('D', 'D')][0] = 2
    def_payoffs[pd][('D', 'D')][1] = 1

    population = [
        AgentSubPopulation(frequency=0.8, payoffs=coop_payoffs),
        AgentSubPopulation(frequency=0.2, payoffs=def_payoffs)
    ]
    return population


if __name__ == '__main__':
    main()

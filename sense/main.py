from ensm.agents import AgentSubPopulation
from ensm.games import Game, GamesNetwork
from ensm.ensm import ENSM
from ensm.mas import MAS

from collections import defaultdict
from ast import literal_eval
import ruamel.yaml as ruamel
import argparse
import logging


MAX_GENERATIONS = 100


def main(config):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    games_net = __create_games(config)
    population = __create_population(games_net, config)
    
    mas = MAS(games_net=games_net, population=population)
    ensm = ENSM(mas=mas, max_generations=MAX_GENERATIONS)

    while not ensm.converged:
        fitnesses = ensm.evolve()
        logger.info(fitnesses)


def __create_games(config) -> GamesNetwork:
    """
    Creates a games network adding the games defined in a configuration file
    :param config: configuration file
    :return: a GamesNetwork containing the games to be played in the MAS
    """
    games = {}
    dependencies = []

    for game_cfg in config['games']:
        name = game_cfg['name']
        games[name] = (Game(name=name,
                            contexts=game_cfg['contexts'],
                            utilities={literal_eval(ac_comb): u for ac_comb, u in game_cfg['utilities'].items()}))

    for game_role_a, game_role_b in config['gameDependencies'].items():
        name_a, role_a = literal_eval(game_role_a)
        name_b, role_b = literal_eval(game_role_b)
        dependencies.append(((games[name_a], role_a), (games[name_b], role_b)))

    return GamesNetwork(games=games, dependencies=dependencies)


def __create_population(games_net: GamesNetwork, config: dict):
    population = []

    for sub_population in config['population']:
        assert 'name' in sub_population, 'Missing \'name\' in sub-population'
        assert 'frequency' in sub_population, f'Missing \'frequency\' in sub-population {sub_population["name"]}'

        for game_payoffs in sub_population['gamePayoffs']:
            assert 'gameName' in game_payoffs, f'Missing \'gameName\' in sub-population {sub_population["name"]}'
            assert 'payoffs' in game_payoffs, f'Missing \'payoffs\' in sub-population {sub_population["name"]}'

            all_payoffs = defaultdict(lambda: defaultdict(tuple))
            game = games_net.games[game_payoffs['gameName']]

            for ac_combination, payoffs in game_payoffs['payoffs'].items():
                all_payoffs[game][literal_eval(ac_combination)] = payoffs

            population.append(AgentSubPopulation(frequency=sub_population['frequency'], payoffs=all_payoffs))

    return population


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, help='Configuration file', required=True)
    parser.add_argument('-d', '--data-path', type=str, help='Local path to save data to', required=True)

    args = parser.parse_args()

    yaml = ruamel.YAML()
    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)

    main(cfg)

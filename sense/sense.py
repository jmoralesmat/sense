from ensm.agents import AgentSubPopulation
from ensm.games import Game, GamesNetwork
from ensm.ensm import ENSM
from ensm.mas import MAS
from ensm.norms import Norm

from collections import defaultdict
from ast import literal_eval

from pprint import pprint
import ruamel.yaml as ruamel
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(config):

    # Create the games network, get the action spaces and norm spaces of each possible coordination context
    # that the agents can play in the games of the MAS, and create an agent population as a set of homogeneous
    # sub-populations, each with a given proportion in the population
    games_net = _create_games(config=config)
    action_spaces, norm_spaces = _create_action_spaces_and_norms(
        games_net=games_net, regulate=config["regulate"]
    )
    population = _create_population(
        games_net=games_net,
        action_spaces=action_spaces,
        norm_spaces=norm_spaces,
        config=config,
    )

    # Create the MAS, the Evolutionary Norm Synthesis Machine, and run evolution until convergence
    mas = MAS(games_net=games_net, population=population)
    ensm = ENSM(
        mas=mas,
        games_net=games_net,
        action_spaces=action_spaces,
        norm_spaces=norm_spaces,
        max_generations=config["maxGenerations"],
        stability_margin=config["stabilityMargin"],
        min_num_stable_generations=config["minNumStableGenerations"],
    )

    while not ensm.converged and not ensm.timed_out:
        action_freqs = ensm.evolve()
        logger.info(action_freqs)
        # pprint(action_freqs)

    pprint(
        f"Evolutionary process converged in {ensm.num_generations - config['minNumStableGenerations']} generations."
    )
    pprint(action_freqs)


def _create_games(config) -> GamesNetwork:
    """
    Creates a games network adding the games defined in a configuration file
    :param config: configuration file
    :return: a GamesNetwork containing the games to be played in the MAS
    """
    games = {}
    dependencies = []

    for game_cfg in config["games"]:
        name = game_cfg["name"]
        games[name] = Game(
            name=name,
            contexts=game_cfg["contexts"],
            utilities={
                literal_eval(ac_comb): u for ac_comb, u in game_cfg["utilities"].items()
            },
        )

    if "gameDependencies" in config:
        for game_role_a, game_role_b in config["gameDependencies"].items():
            name_a, role_a = literal_eval(game_role_a)
            name_b, role_b = literal_eval(game_role_b)
            dependencies.append(((games[name_a], role_a), (games[name_b], role_b)))

    return GamesNetwork(games=games, dependencies=dependencies)


def _create_action_spaces_and_norms(games_net, regulate):
    action_spaces = defaultdict(list)
    norm_spaces = defaultdict(list)

    # Get all possible pairs of (game, role) that the agents can play
    game_roles = [
        (game, role)
        for game in games_net.games.values()
        for role in range(game.num_roles)
    ]
    for game, role in game_roles:
        sanctions = [None] if game.sanctions is None else game.sanctions

        # Get all possible pairs (context, action) of actions that the agents can perform in each context
        context_actions = [
            (context, action)
            for context in games_net.contexts_playing(game, role)
            for action in game.action_space(role)
        ]

        # Create the action space of each context and a norm for each possible action in the action space
        for context, action in context_actions:
            if action not in action_spaces[context]:
                action_spaces[context].append(action)

                if regulate:
                    for sanction in sanctions:
                        norm_spaces[context].append(Norm(context, action, sanction))

            if not regulate and context not in norm_spaces:
                norm_spaces[context].append(None)

    return action_spaces, norm_spaces


def _create_population(
    games_net: GamesNetwork, action_spaces: dict, norm_spaces: dict, config: dict
):
    population = []

    for sub_population in config["population"]:
        assert "name" in sub_population, "Missing 'name' in sub-population"
        assert (
            "proportion" in sub_population
        ), f'Missing \'frequency\' in sub-population {sub_population["name"]}'

        all_payoffs = defaultdict(lambda: defaultdict(tuple))
        for game_payoffs in sub_population["gamePayoffs"]:
            assert (
                "gameName" in game_payoffs
            ), f'Missing \'gameName\' in sub-population {sub_population["name"]}'
            assert (
                "payoffs" in game_payoffs
            ), f'Missing \'payoffs\' in sub-population {sub_population["name"]}'

            game = games_net.games[game_payoffs["gameName"]]

            for ac_combination, payoffs in game_payoffs["payoffs"].items():
                all_payoffs[game][literal_eval(ac_combination)] = payoffs

        population.append(
            AgentSubPopulation(
                name=sub_population["name"],
                proportion=sub_population["proportion"],
                payoffs=all_payoffs,
                action_spaces=action_spaces,
                norm_spaces=norm_spaces,
            )
        )

    return population


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", type=str, help="Configuration file", required=True
    )
    parser.add_argument(
        "-d", "--data-path", type=str, help="Local path to save data to", required=True
    )

    args = parser.parse_args()

    yaml = ruamel.YAML()
    with open(args.config_file, "r") as f:
        cfg = yaml.load(f)

    main(cfg)

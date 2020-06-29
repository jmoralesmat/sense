from ensm.agents import AgentSubPopulation
from ensm.games import GamesNetwork, Game
from ensm.norms import Norm
import numpy as np
import itertools


class StrategyReplicator(object):

    @staticmethod
    def update_fitness(sub_population: AgentSubPopulation, games_net: GamesNetwork,
                       action_spaces: dict, norm_spaces: dict, mean_action_freqs_by_game: dict,
                       fitness_aggregation):
        """

        :param sub_population:
        :param games_net:
        :param action_spaces:
        :param norm_spaces:
        :param mean_action_freqs_by_game:
        :param fitness_aggregation:
        :return:
        """

        for context in games_net.contexts:
            for action in action_spaces[context]:
                for norm in norm_spaces[context]:
                    all_fitnesses = []

                    # Compute the fitness values of the sub-population in each co-dependent game that
                    # they play when they perceive the context
                    all_played_roles = games_net.played_roles(context).items()
                    for game, role in [(game, role) for game, roles in all_played_roles for role in roles]:
                        fitness_in_game = StrategyReplicator.__compute_fitness_in_game(
                            game=game, role=role, action=action, sub_population=sub_population,
                            mean_action_freqs_by_game=mean_action_freqs_by_game
                        )
                        all_fitnesses.append(fitness_in_game)

                    # Aggregate all fitness using a pre-defined fitness aggregation function
                    sub_population.fitness[context][norm][action] = fitness_aggregation(all_fitnesses)

    @staticmethod
    def __compute_fitness_in_game(game: Game, role: int, action: str, sub_population: AgentSubPopulation,
                                  mean_action_freqs_by_game: dict):
        """

        :param game:
        :param role:
        :param action:
        :param sub_population:
        :param mean_action_freqs_by_game:
        :return:
        """
        fitness = np.float64(0)

        # Get all action combinations that can be played in the game and in which the agent
        # playing the given role is performing the specified action
        action_spaces = [game.action_space(role) for role in range(game.num_roles)]
        action_combinations = [ac for ac in itertools.product(*action_spaces) if ac[role] == action]

        # Get the payoff of the player with the given role once all the players of the game
        # perform each combination of actions. Accumulate the payoff in the fitness computation
        # by weighting it with the frequency with which the action combination is played in the game,
        # computed as the joint mean frequency of each action in the action combination
        for action_combination in action_combinations:
            payoff_in_game_role = sub_population.payoff[game][action_combination][role]

            joint_action_freq = StrategyReplicator.__get_action_combination_frequency(
                game=game, action_combination=action_combination, reference_role=role,
                mean_action_freqs_by_game=mean_action_freqs_by_game
            )
            fitness += payoff_in_game_role * joint_action_freq

        return fitness

    @staticmethod
    def __get_action_combination_frequency(game: Game, action_combination: tuple, reference_role: int,
                                           mean_action_freqs_by_game: dict):

        """
        Returns the joint probability that the agents of a MAS will play the game by adopting
        a given combination of actions when playing each role
        :return:
        """
        # Get roles not played by the reference agent
        roles = [r for r in range(len(action_combination)) if r != reference_role]

        # Get the global (mean) frequency of the action played by each role and mutiply them all
        # to compute the frequency of the action combination
        action_freqs = [mean_action_freqs_by_game[game][role][action_combination[role]] for role in roles]
        action_comb_freq = np.prod(action_freqs)

        return action_comb_freq

    @staticmethod
    def replicate(sub_population: AgentSubPopulation, games_net: GamesNetwork, action_spaces: dict, norm_spaces: dict):
        """

        :param sub_population:
        :param games_net:
        :param action_spaces:
        :param norm_spaces:
        :return:
        """
        for context in games_net.contexts:
            action_space = action_spaces[context]

            for norm in norm_spaces[context]:

                # Compute mean sub-population fitness for any possible action that they can perform
                # once they are given the norm
                action_fitnesses = sub_population.fitness[context][norm]
                action_freqs = sub_population.action_freqs[context][norm]
                mean_fitness = np.sum([action_fitnesses[action] * action_freqs[action]
                                       for action in action_space])

                # Update frequency of each action using the Replicator Equation. Clip low action frequencies
                # to 1e-5 in order to ensure that they never go to zero and hence can be resurrected
                for action in action_space:
                    action_freqs[action] = max(action_freqs[action] * (action_fitnesses[action] / mean_fitness), 1e-5)

                # Normalise so that all action frequencies sum up to 1 (just in case due to float point precision)
                total_freq = np.sum([action_freqs[a] for a in action_space])
                for action in action_space:
                    action_freqs[action] /= total_freq

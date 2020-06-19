from ensm.strategies import StrategyReplicator
from ensm.games import GamesNetwork
from ensm.norms import Norm
from ensm.mas import MAS

from collections import defaultdict
import numpy as np


class ENSM(object):
    def __init__(self, mas: MAS, games_net: GamesNetwork, action_spaces: dict, norm_spaces: dict, max_generations: int):
        self.__max_generations = max_generations
        self.__action_spaces = action_spaces
        self.__norm_spaces = norm_spaces
        self.__games_net = games_net
        self.__mas = mas

        self.__must_evolve_norms = True
        self.__num_generations = 0
        self.__new_norms = []

        self.__converged = False
        self.__timeout = False

        # Dictionary of context -> norm -> frequency that stores the frequencies of each norm in the norm space
        # of each context, no matter the profile of the agents that have the norm. Each norm is provided to the
        # same proportion of each agent sub-population. For example, a norm with frequency 0.5 will be provided
        # to 50% of the agents in each sub-population. Note that the frequencies of the norms
        # of a context should sum up to 1)
        self.__norm_freqs = {c: {n: np.float64(1 / len(norm_spaces[c])) for n in norm_spaces[c]}
                             for c in norm_spaces.keys()}

        # Dictionary of context -> norm -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population with a given norm perform the action in a context, no matter their profile
        # (that is, it is averaged across all sub-populations)
        self.__mean_action_freqs_by_norm = {c: {n: {a: np.float64(1 / np.float64(len(action_spaces[c])))
                                                    for a in action_spaces[c]} for n in norm_spaces[c]}
                                            for c in norm_spaces}

        # Dictionary of context -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population perform a given action when perceiving a given context,
        # no matter their profile (that is, it is averaged across all sub-populations)
        self.__mean_action_freqs_by_context = {c: {n: np.float64(1 / np.float64(len(action_spaces[c])))
                                                   for n in list(action_spaces[c])} for c in action_spaces}

        # Dictionary of game -> role -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population will perform a given action when playing a role of a game,
        # no matter their profile (that is, it is averaged across all sub-populations)
        self.__mean_action_freqs_by_game = {g: {r: {a: np.float64(1 / len(g.action_space(r)))
                                                    for a in g.action_space(r)}
                                                for r in range(g.num_roles)}
                                            for g in games_net.games.values()}

        # Set up action frequencies
        self.__update_action_frequencies()

    def evolve(self):
        """

        :return:
        """
        self.__num_generations += 1
        self.__new_norms = []

        # Update the strategy probabilities of each agent profile based on the
        # frequencies of the norms that they are provided with
        self.__evolve_strategies()
        self.__update_action_frequencies()

        # Evaluate norms in terms of their utility to achieve the MAS goals. Replicate norms based on their utility
        if self.__must_evolve_norms:
            self.__evolve_norms()

        # Adjust norm frequencies (because of norm reproduction)
        self.__adjust_norm_frequencies()

        # Get population fitnesses organised by context
        context_fitness = defaultdict(list)
        for context in self.__games_net.contexts:
            for sub_population in self.mas.population:
                context_fitness[context].append(sub_population.fitness[context])

        self.__converged = True  # TODO Change this
        self.__timeout = self.__num_generations > self.__max_generations

        return context_fitness

    def __evolve_strategies(self):
        for sub_population in self.mas.population:
            StrategyReplicator.update_fitness(sub_population, self.__games_net)
            StrategyReplicator.replicate(sub_population, self.__games_net)

    def __update_action_frequencies(self):
        """
        Computes the probabilities that the agents will perform each action combination in each game
        given their current configuration (in terms of strategy/norm frequencies)
        :return:
        """
        for context in self.games_net.contexts:
            for action in self.action_spaces[context]:
                mean_action_freq = np.float64(0)

                for norm in self.norm_spaces[context]:
                    mean_action_freq_by_norm = np.float64(0)

                    # Get the proportion of the agent sub-population that have the norm and perform
                    # the given action when perceiving the context. Accumulate the action frequency for the
                    # agents that have the norm, averaged across all sub-populations
                    for sub_population in self.mas.population:
                        action_freq_per_norm = sub_population.action_freqs[context][norm][action]
                        mean_action_freq_by_norm += action_freq_per_norm * sub_population.frequency

                    # Save the overall action frequency in the context for the agents that
                    # have the norm, averaged across all sub-populations
                    self.__mean_action_freqs_by_norm[context][norm][action] = mean_action_freq_by_norm

                    # Accumulate the action frequency averaged across all norms
                    mean_action_freq += mean_action_freq_by_norm * self.__norm_freqs[context][norm]

                # Save the overall action frequency in the context, no matter the norms they have
                # or their profile (averaged across all norms and sub-populations)
                self.__mean_action_freqs_by_context[context][action] = mean_action_freq

        # Compute the global action frequencies per game and role, averaged across all norms and sub-populations
        for game, role in [(game, role) for game in self.games_net.games for role in range(game.num_roles)]:
            for action in game.action_space(role):
                contexts_playing = self.games_net.contexts_playing(game, role)

                action_freq = np.float64(sum(self.__mean_action_freqs_by_context[context][action]
                                             for context in contexts_playing) / np.float64(len(contexts_playing)))

                self.__mean_action_freqs_by_game[game][role][action] = action_freq

    def __evolve_norms(self):
        for sub_population in self.mas.population:
            StrategyReplicator.update_fitness(sub_population, self.__games_net)
            StrategyReplicator.replicate(sub_population, self.__games_net)

    def __adjust_norm_frequencies(self):
        pass

    @property
    def mas(self):
        return self.__mas

    @property
    def num_generations(self):
        return self.__num_generations

    @property
    def converged(self):
        return self.__converged

    @property
    def games_net(self):
        return self.__games_net

    @property
    def action_spaces(self):
        return self.__action_spaces

    @property
    def norm_spaces(self):
        return self.__norm_spaces

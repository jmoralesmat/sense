from ensm.strategies import StrategyReplicator
from ensm.agents import AgentSubPopulation
from ensm.norms import NormReplicator
from ensm.games import GamesNetwork
from ensm.mas import MAS

from collections import defaultdict
from copy import deepcopy
import numpy as np


class ENSM(object):
    def __init__(self, mas: MAS, games_net: GamesNetwork, action_spaces: dict, norm_spaces: dict,
                 max_generations: int, stability_margin: float, min_num_stable_generations: int):
        """

        :param mas:
        :param games_net:
        :param action_spaces:
        :param norm_spaces:
        :param max_generations:
        :param stability_margin:
        """
        self._min_num_stable_generations = min_num_stable_generations
        self._stability_margin = stability_margin
        self._max_generations = max_generations
        self._action_spaces = action_spaces
        self._norm_spaces = norm_spaces
        self._games_net = games_net
        self._mas = mas

        self._must_evolve_norms = False
        self._num_generations = 0
        self._num_stable_generations = 0
        self._new_norms = []

        self._converged = False
        self._timeout = False

        # To save the action frequencies for each context/norm before replication
        self._old_action_freqs = defaultdict(dict)

        # Dictionary of context -> norm -> frequency that stores the frequencies of each norm in the norm space
        # of each context, no matter the profile of the agents that have the norm. Each norm is provided to the
        # same proportion of each agent sub-population. For example, a norm with frequency 0.5 will be provided
        # to 50% of the agents in each sub-population. Note that the frequencies of the norms
        # of a context should sum up to 1)
        self._norm_freqs = {c: {n: np.float64(1 / len(norm_spaces[c])) for n in norm_spaces[c]}
                            for c in norm_spaces.keys()}

        # Dictionary of context -> norm -> utility that stores the utilities of each norm
        # in the norm space of each context that the agents can perceive in the MAS
        self._norm_utilities = {c: {n: np.float64(0) for n in norm_spaces[c]} for c in norm_spaces.keys()}

        # Dictionary of context -> norm -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population with a given norm perform the action in a context, no matter their profile
        # (that is, it is averaged across all sub-populations)
        self._mean_action_freqs_by_norm = {c: {n: {a: np.float64(1 / np.float64(len(action_spaces[c])))
                                                   for a in action_spaces[c]} for n in norm_spaces[c]}
                                           for c in norm_spaces}

        # Dictionary of context -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population perform a given action when perceiving a given context,
        # no matter their profile (that is, it is averaged across all sub-populations)
        self._mean_action_freqs_by_context = {c: {n: np.float64(1 / np.float64(len(action_spaces[c])))
                                                  for n in list(action_spaces[c])} for c in action_spaces}

        # Dictionary of game -> role -> action -> frequency that stores the overall frequency with which
        # the agents in the MAS population will perform a given action when playing a role of a game,
        # no matter their profile (that is, it is averaged across all sub-populations)
        self._mean_action_freqs_by_game = {g: {r: {a: np.float64(1 / len(g.action_space(r)))
                                                   for a in g.action_space(r)}
                                               for r in range(g.num_roles)}
                                           for g in games_net.games.values()}

        # Set up action frequencies
        self._update_action_frequencies()

    def evolve(self):
        """

        :return:
        """
        self._num_generations += 1
        self._new_norms = []

        # Update the strategy probabilities of each agent profile based on the
        # frequencies of the norms that they are provided with
        self._evolve_strategies()
        self._update_action_frequencies()

        # Evaluate norms in terms of their utility to achieve the MAS goals. Replicate norms based on their utility
        if self._must_evolve_norms:
            self._evolve_norms()

        # Get population fitnesses organised by context
        context_fitness = defaultdict(dict)
        for context in self._games_net.contexts:
            for sub_population in self.mas.population:
                context_fitness[context][sub_population] = sub_population.fitness[context]

        self._converged = self._check_convergence()
        self._timeout = self._num_generations > self._max_generations

        return self._old_action_freqs

    def _evolve_strategies(self):
        """ Evolve strategies """
        for sub_population in self.mas.population:

            # Backup sub-population action frequencies
            self._old_action_freqs[sub_population] = deepcopy(sub_population.action_freqs)

            # Update sub-population fitness and replicate
            StrategyReplicator.update_fitness(sub_population=sub_population,
                                              games_net=self._games_net,
                                              action_spaces=self.action_spaces,
                                              norm_spaces=self.norm_spaces,
                                              mean_action_freqs_by_game=self._mean_action_freqs_by_game,
                                              fitness_aggregation=min)
            StrategyReplicator.replicate(sub_population=sub_population,
                                         games_net=self._games_net,
                                         action_spaces=self.action_spaces,
                                         norm_spaces=self.norm_spaces)

    def _evolve_norms(self):
        """ Evolve norms """
        for context in self.games_net.contexts:
            NormReplicator.update_utilities(context=context,
                                            games_net=self.games_net,
                                            norm_space=self.norm_spaces[context],
                                            action_freqs=self._mean_action_freqs_by_context,
                                            action_freqs_by_norm=self._mean_action_freqs_by_norm)
            NormReplicator.replicate(context=context,
                                     norm_freqs=self._norm_freqs,
                                     norm_utilities=self._norm_utilities)

    def _update_action_frequencies(self):
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
                        mean_action_freq_by_norm += action_freq_per_norm * sub_population.proportion

                    # Save the overall action frequency in the context for the agents that
                    # have the norm, averaged across all sub-populations
                    self._mean_action_freqs_by_norm[context][norm][action] = mean_action_freq_by_norm

                    # Accumulate the action frequency averaged across all norms
                    mean_action_freq += mean_action_freq_by_norm * self._norm_freqs[context][norm]

                # Save the overall action frequency in the context, no matter the norms they have
                # or their profile (averaged across all norms and sub-populations)
                self._mean_action_freqs_by_context[context][action] = mean_action_freq

        # Compute the global action frequencies per game and role, averaged across all norms and sub-populations
        for game, role in [(game, role) for game in self.games_net.games.values() for role in range(game.num_roles)]:
            for action in game.action_space(role):
                contexts_playing = self.games_net.contexts_playing(game, role)

                action_freq = np.float64(sum(self._mean_action_freqs_by_context[context][action]
                                             for context in contexts_playing) / np.float64(len(contexts_playing)))

                self._mean_action_freqs_by_game[game][role][action] = action_freq

    def _check_convergence(self):
        """

        :return:
        """
        stable = True

        for sub_population, context, norm, action in [(p, c, n, a) for p in self.mas.population
                                                      for c in self.games_net.contexts for n in self.norm_spaces[c]
                                                      for a in self._action_spaces[c]]:

            old_action_freq = self._old_action_freqs[sub_population][context][norm][action]
            curr_action_freq = sub_population.action_freqs[context][norm][action]

            if abs(old_action_freq - curr_action_freq) > self._stability_margin:
                stable = False
                break

        self._num_stable_generations = self._num_stable_generations + 1 if stable else 0

        converged = False
        if self._num_stable_generations >= self._min_num_stable_generations:
            converged = True

        return converged

    @property
    def mas(self):
        return self._mas

    @property
    def num_generations(self):
        return self._num_generations

    @property
    def converged(self):
        return self._converged

    @property
    def timed_out(self):
        return self._timeout

    @property
    def games_net(self):
        return self._games_net

    @property
    def action_spaces(self):
        return self._action_spaces

    @property
    def norm_spaces(self):
        return self._norm_spaces

from collections import defaultdict
from typing import Dict, Set
import numpy as np
import random


class AgentSubPopulation(object):
    """ A homogeneous sub-population of agents with the same profile. The agents within a sub-population
    will all have the same payoffs in each role of each possible game they can play in a MAS """

    def __init__(self, name: str, proportion: float, payoffs: dict,
                 action_spaces: Dict[str, Set], norm_spaces: Dict[str, Set]):
        """
        Initialises an agent sub-population with a given frequency, a dictionary of payoffs for each triplet
        (role, action_combination) that they can play in each possible game, and the frequencies of each norm
        within the sub-population (which virtually splits the sub-population in as many sub-sub-population as norms)

        :param name: name of the agent profile
        :param proportion: proportion of agents with the profile in a MAS
        :param payoffs: base payoff that an agent expects to obtain when playing a particular role of a game and
        a given combination of actions is played by all the players of the game (including the agent itself).
        This variable should be a dictionary of (game, role, action_combination) -> player payoff
        :param action_spaces: dictionary of agent contexts to the sets of actions that can be performed in them
        :param norm_spaces: dictionary of agent contexts to their applicable norms
        """
        self._name = name
        self._proportion = proportion
        self._payoffs = payoffs

        # Frequencies of each norm in each possible context that the sub-population may encounter in all games.
        # This data structure is of the form: context -> norm -> frequency
        self._norm_freqs = defaultdict(lambda: defaultdict(np.float64))
        for context in norm_spaces:
            norms_list = list(norm_spaces[context])
            norm_freqs = [random.randint(70, 100) for _ in norms_list]
            for i in range(len(norms_list)):
                self._norm_freqs[context][norms_list[i]] = np.float64(norm_freqs[i] / sum(norm_freqs))

        # Split of the sub-population by norms, where each sub-sub-population has a different norm for
        # each context they may encounter in the MAS, and hence, may have different frequencies for each
        # action in the action space of each context. This data structure is a dictionary of the form
        # context -> norm -> action -> frequency
        self._action_freqs = defaultdict(lambda: defaultdict(lambda: defaultdict(np.float64)))
        for context in norm_spaces:
            for norm in norm_spaces[context]:

                actions_list = list(action_spaces[context])
                action_freqs = [random.randint(70, 100) for _ in actions_list]
                for i in range(len(actions_list)):
                    self._action_freqs[context][norm][actions_list[i]] = np.float64(action_freqs[i] / sum(action_freqs))

        # Fitness (expected average payoff) of the sub-population with a given norm when it repeatedly
        # interacts in a coordination context with other agents from its same sub-population and
        # other sub-populations, each having different frequencies. Dictionary of the form
        # context -> norm -> action -> fitness
        self._fitness = {c: {n: {a: np.float64(0) for a in action_spaces[c]} for n in norm_spaces[c]}
                         for c in norm_spaces}

    @property
    def proportion(self):
        return self._proportion

    @property
    def payoff(self):
        return self._payoffs

    @property
    def norm_freqs(self):
        """ Returns dictionary of the form context -> norm -> frequency """
        return self._norm_freqs

    @property
    def action_freqs(self):
        """ Returns dictionary of the form context -> norm -> action -> frequency """
        return self._action_freqs

    @property
    def fitness(self):
        return self._fitness

    def __str__(self):
        return self._name

    def __repr__(self):
        return self.__str__()


class AgentContext(object):
    """ An agent's context in a game, that is, the agent's individual perspective when playing the game """

    def __init__(self, ctxt_desc: str):
        """ Initialises the agent context from a string of the form:

                'pred_1(term_1) & pred_2(term2) & ... & pred_n(term_n)'

            where each pair (pred_i, term_i) stands for a predicate and a term defining one perception
            block of the world from an agent's perspective. Example (in a car traffic domain):

                'front-right(car-to-left) & front(car-to-left)'

            where 'front-right(car-to-left)' describes a car to the right position of a reference car which is
            heading towards the left from its perspective, and 'front(car-to-left)' describes a car in front of
            the reference car which is heading towards the left
        """
        pred_terms = sorted([p.strip() for p in ctxt_desc.split('&')])

        assert len(pred_terms) > 0, 'The number of predicates must be at least 1'

        self._context = {}

        for pred_term in pred_terms:
            pred, term = pred_term.split('(')
            pred = pred.replace('"', '')
            term = term.replace('"', '').replace(')', '')

            assert len(pred) > 0, 'One of the predicates was wrongly defined and its length is zero'
            assert len(term) > 0, 'One of the terms was wrongly defined and its length is zero'

            self._context[pred.strip()] = term.strip()

    def __str__(self):
        """ Describes an agent context """
        return ' & '.join([f'{pred}({term})' for pred, term in self._context.items()])

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    @property
    def predicates(self):
        """ Returns the predicates of a context """
        return list(self._context.keys())

    def term(self, pred):
        """ Returns the term of a predicate """
        return self._context[pred]

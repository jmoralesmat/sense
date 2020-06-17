from collections import defaultdict
from typing import List, Tuple

from ensm.games import Game
from ensm.norms import Norm
import numpy as np


class AgentSubPopulation(object):
    """ A homogeneous sub-population of agents with the same profile. The agents within a sub-population
    will all have the same payoffs in each role of each possible game they can play in a MAS """

    def __init__(self, frequency: float, payoffs: dict, norms: List[Norm]=None):
        """
        Initialises an agent sub-population with a given frequency, a dictionary of payoffs for each triplet
        (role, action_combination) that they can play in each possible game, and the frequencies of each norm
        within the sub-population (which virtually splits the sub-population in as many sub-sub-population as norms)

        :param frequency: frequency of the agent sub-population
        :param payoffs: dictionary of (game, action_combination, role) -> player payoff
        :param norms: set of possible norms to be applied to each player of the game
        """
        self.__frequency = frequency
        self.__payoffs = payoffs
        self.__norms = norms if norms is not None else [None]

        # Frequencies of each norm in the sub-population
        self.__norm_freqs = {n: 1/len(self.__norms) for n in self.__norms}
        self.__action_freqs = defaultdict(lambda: defaultdict(lambda: defaultdict(np.float64)))

    @property
    def frequency(self):
        return self.__frequency

    def payoff(self, action_combination: Tuple[str], role: int, game: Game):
        """
        Returns the base payoff that an agent expects to obtain when playing a particular role of a game
        and a given combination of actions is played by all the players of the game (including the agent itself)

        :param action_combination: a combination of actions carried out by the players of a game
        :param role: the role of the agent to get the payoff from
        :param game: a game to be played by a group of agents, each with some role
        :return: a scalar defining the agent's payoff
        """
        return self.__payoffs[game][action_combination][role]

    @property
    def action_freqs(self):
        return self.__action_freqs


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
        pred_terms = [p.strip() for p in ctxt_desc.split('&')]
        assert len(pred_terms) > 0, 'The number of predicates must be at least 1'

        self.__context = {}

        for pred_term in pred_terms:
            pred, term = pred_term.split('(')
            pred = pred.replace('"', '')
            term = term.replace('"', '').replace(')', '')

            assert len(pred) > 0, 'One of the predicates was wrongly defined and its length is zero'
            assert len(term) > 0, 'One of the terms was wrongly defined and its length is zero'

            self.__context[pred.strip()] = term.strip()

    def __str__(self):
        """ Describes an agent context """
        return ' & '.join([f'{pred}({term})' for pred, term in self.__context.items()])

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    @property
    def predicates(self):
        """ Returns the predicates of a context """
        return list(self.__context.keys())

    def term(self, pred):
        """ Returns the term of a predicate """
        return self.__context[pred]

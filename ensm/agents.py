from typing import Dict, Set
import numpy as np


class AgentSubPopulation(object):
    """ A homogeneous sub-population of agents with the same profile. The agents within a sub-population
    will all have the same payoffs in each role of each possible game they can play in a MAS """

    def __init__(self, frequency: float, payoffs: dict, action_spaces: Dict[str, Set], norm_spaces: Dict[str, Set]):
        """
        Initialises an agent sub-population with a given frequency, a dictionary of payoffs for each triplet
        (role, action_combination) that they can play in each possible game, and the frequencies of each norm
        within the sub-population (which virtually splits the sub-population in as many sub-sub-population as norms)

        :param frequency: frequency of the agent sub-population in the whole MAS
        :param payoffs: base payoff that an agent expects to obtain when playing a particular role of a game and
        a given combination of actions is played by all the players of the game (including the agent itself).
        This variable should be a dictionary of (game, role, action_combination) -> player payoff
        :param norm_spaces: dictionary of agent contexts to the sets of norms applicable in them
        :param norm_spaces: dictionary of agent contexts to the sets of actions that can be performed in them
        """
        self.__frequency = frequency
        self.__payoffs = payoffs

        # Frequencies of each norm in each possible context that the sub-population may encounter in all games.
        # This data structure is of the form: context -> norm -> norm_frequency
        self.__norm_freqs = {ctxt: {n: np.float64(1 / np.float64(len(norm_spaces[ctxt])))
                                    for n in list(norm_spaces[ctxt])} for ctxt in norm_spaces}

        # Split of the sub-population by norms, where each sub-sub-population will have a different norm for
        # each context they may encounter in the MAS, and hence, may have different frequencies for each
        # action in the action space of each context. This data structure is a dictionary of the form
        # context -> norm -> action -> action_frequency
        self.__action_freqs = {ctxt: {n: {a: np.float64(1 / np.float64(len(action_spaces[ctxt])))
                                          for a in action_spaces[ctxt]} for n in norm_spaces[ctxt]}
                               for ctxt in norm_spaces}

        # Fitness (expected average payoff) of the sub-population with a given norm when it repeatedly
        # interacts in a coordination context with other agents from its same sub-population and
        # other sub-populations, each having different frequencies
        self.__fitness = {ctxt: {n: {a: np.float64(0) for a in action_spaces[ctxt]} for n in norm_spaces[ctxt]}
                          for ctxt in norm_spaces}

    @property
    def frequency(self):
        return self.__frequency

    @property
    def payoff(self):
        return self.__payoffs

    @property
    def norm_freqs(self):
        return self.__norm_freqs

    @property
    def action_freqs(self):
        return self.__action_freqs

    @property
    def fitness(self):
        return self.__fitness


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

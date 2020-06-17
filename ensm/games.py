from collections import defaultdict
from typing import List, Dict


class Game(object):
    """ A strategic situation played between two or more players that interact, each playing one role of the game """

    def __init__(self, name, contexts, utilities):
        """
        Creates a game
        :param name: descriptive name of the game
        :param contexts: individual contexts of each player of the aame from their own perspective
        :param utilities: dictionary of action lists (action combinations) to their payoffs
        """
        self.__player_contexts = contexts
        self.__utilities = utilities
        self.__name = name

    def utility(self, action_combination: tuple):
        return self.__utilities[action_combination]

    @property
    def num_roles(self):
        return len(self.__player_contexts)

    @property
    def contexts(self):
        return self.__player_contexts

    @property
    def name(self):
        return self.__name

    @property
    def __str__(self):
        return self.name


class GamesNetwork(object):
    """ A network of games and the dependencies between their roles """

    def __init__(self, games: Dict[str, Game], dependencies: List[tuple]):
        self.__dependencies = defaultdict(lambda: defaultdict(set))
        self.__generalisations = defaultdict(set)
        self.__games = games

        # Add each agent context from each game as a new coordination context to regulate
        self.__coord_contexts = [ctxt for game in games.values() for ctxt in game.contexts]

        for (game_role_a, game_role_b) in dependencies:
            self.add_dependency(game_role_a, game_role_b)

    def add_dependency(self, game_role_a: tuple, game_role_b: set):
        """
        Adds a dependency between two roles of two different games
        :param game_role_a: tuple of the form (game, role)
        :param game_role_b: tuple of the form (game, role)
        """
        game_a, role_a = game_role_a
        game_b, role_b = game_role_b

        self.__dependencies[game_a][role_a].add(game_role_b)
        self.__dependencies[game_b][role_b].add(game_role_a)

        # Generate and add the joint contexts resulting from the interdependency from the two games
        context_a = game_a.contexts[role_a]
        context_b = game_b.contexts[role_b]

        if context_a != context_b:
            joint_context = ' & '.join([context_a, context_b])
            self.__coord_contexts.append(joint_context)

            # Add the joint contexts as parent of the two joined contexts
            self.__generalisations[context_a].add(joint_context)
            self.__generalisations[context_b].add(joint_context)

    def dependencies(self, game, role):
        """ Returns the dependencies of game's role with the roles of other games """
        return self.__dependencies[game][role]

    @property
    def games(self):
        """ Returns a dictionary of games with their names being the keys of the dictionary """
        return self.__games

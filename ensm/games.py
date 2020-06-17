from networkx.algorithms.dag import ancestors, descendants
from collections import defaultdict
from typing import List, Dict
import networkx as nx


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
        self.__contexts_graph = nx.DiGraph()
        self.__games = games

        # Add each agent context from each game as a new coordination context to regulate
        self.__coord_contexts = defaultdict(lambda: defaultdict(set))
        self.__played_roles = defaultdict(lambda: defaultdict(set))

        for game in games.values():
            for role, ctxt in enumerate(game.contexts):
                self.__coord_contexts[game][role].add(ctxt)
                self.__played_roles[ctxt][game].add(role)

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

            # Generate joint context and keep track of the game/roles in which it applies in both directions
            # (from a game-role to the context and from the context to the game-roles it plays)
            joint_context = ' & '.join([context_a, context_b])
            self.__coord_contexts[game_a][role_a].add(joint_context)
            self.__coord_contexts[game_b][role_b].add(joint_context)
            self.__played_roles[joint_context][game_a].add(role_a)
            self.__played_roles[joint_context][game_b].add(role_b)

            # Add the joint context as parent of the two joined contexts
            self.__contexts_graph.add_edges_from([(joint_context, context_a), (joint_context, context_b)])

    def dependencies(self, game, role):
        """
        Returns the dependencies of game's role with the roles of other games
        :param game: a game
        :param role: a game's role
        :return: set of dependencies of a game
        """
        return self.__dependencies[game][role]

    def contexts_playing(self, game, role):
        """
        Returns the set of contexts that apply to (can play) a given role of a game
        :param game: a game
        :param role: the role of a game
        :return: set of contexts applicable to the game's role
        """
        contexts = self.__coord_contexts[game][role]
        all_contexts = set(contexts)
        for context in contexts:
            ancest = ancestors(self.__contexts_graph, context)
            all_contexts = all_contexts.union(ancest)

        return all_contexts

    def played_roles(self, context):
        """
        Returns a dictionary of games roles to which a context is applicable
        :param context: an agent's context
        :return: a dictionary of game roles where the context is applicable
        """
        return self.__played_roles[context]

    @property
    def games(self):
        """ Returns a dictionary of games with their names being the keys of the dictionary """
        return self.__games

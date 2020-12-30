from collections import defaultdict
from typing import List, Dict


class Game(object):
    """ A strategic situation played between two or more players that interact, each playing one role of the game """

    def __init__(self, name, contexts, utilities, sanctions=None):
        """
        Creates a game
        :param name: descriptive name of the game
        :param contexts: individual contexts of each player of the aame from their own perspective
        :param utilities: dictionary of action lists (action combinations) to their payoffs
        """
        self._player_contexts = contexts
        self._utilities = utilities
        self._name = name
        self._sanctions = sanctions

        # Create action spaces of each role of the game
        self._action_spaces = defaultdict(list)
        for role in range(self.num_roles):
            for ac_comb in self._utilities:

                action = ac_comb[role]
                if action not in self._action_spaces[role]:
                    self._action_spaces[role].append(ac_comb[role])

    def utility(self, action_combination: tuple):
        return self._utilities[action_combination]

    def action_space(self, role):
        return self._action_spaces[role]

    @property
    def num_roles(self):
        return len(self._player_contexts)

    @property
    def contexts(self):
        return self._player_contexts

    @property
    def name(self):
        return self._name

    @property
    def sanctions(self):
        return self._sanctions

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class GamesNetwork(object):
    """ A network of games and the dependencies between their roles """

    def __init__(self, games: Dict[str, Game], dependencies: List[tuple]):
        self._dependencies = defaultdict(lambda: defaultdict(set))
        # self._contexts_graph = nx.DiGraph()
        self._games = games

        # Add each agent context from each game as a new coordination context to regulate
        self._contexts_per_role = defaultdict(lambda: defaultdict(set))
        self._roles_per_context = defaultdict(lambda: defaultdict(set))

        for game in games.values():
            for role, ctxt in enumerate(game.contexts):
                self._contexts_per_role[game][role].add(ctxt)
                self._roles_per_context[ctxt][game].add(role)

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
        self._dependencies[game_a][role_a].add(game_role_b)
        self._dependencies[game_b][role_b].add(game_role_a)

        # Generate joint context and keep track of the game/roles in which it applies in both directions
        # (from a game-role to the context and from the context to the game-roles it plays)
        # TODO Might need to sort contexts for consistence? Check this when testing
        if game_a.contexts[role_a] != game_b.contexts[role_b]:
            joint_context = " & ".join(
                [game_a.contexts[role_a], game_b.contexts[role_b]]
            )
            self._contexts_per_role[game_a][role_a].add(joint_context)
            self._contexts_per_role[game_b][role_b].add(joint_context)
            self._roles_per_context[joint_context][game_a].add(role_a)
            self._roles_per_context[joint_context][game_b].add(role_b)

            # # Add the joint context as parent of the two joined contexts
            # self._contexts_graph.add_edges_from([(joint_context, context_a), (joint_context, context_b)])

    def dependencies(self, game, role):
        """
        Returns the dependencies of game's role with the roles of other games
        :param game: a game
        :param role: a game's role
        :return: set of dependencies of a game
        """
        return self._dependencies[game][role]

    def contexts_playing(self, game, role):
        """
        Returns the set of contexts that apply to (can play) a given role of a game
        :param game: a game
        :param role: the role of a game
        :return: set of contexts applicable to the game's role
        """
        return self._contexts_per_role[game][role]

    def played_roles(self, context):
        """
        Returns a dictionary of games roles to which a context is applicable
        :param context: an agent's context
        :return: a dictionary of game roles where the context is applicable
        """
        return self._roles_per_context[context]

    @property
    def games(self):
        """ Returns a dictionary of games with their names being the keys of the dictionary """
        return self._games

    @property
    def contexts(self):
        return list(self._roles_per_context.keys())

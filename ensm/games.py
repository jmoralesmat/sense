class Game(object):
    def __init__(self, name, contexts, utilities):
        """

        "param name:
        :param contexts:
        :param utilities: dictionary of action lists (action combinations) to their payoffs
        """
        self.__player_contexts = contexts
        self.__utilities = utilities
        self.__name = name

        self.__num_roles = len(self.__player_contexts)

    def utility(self, action_combination: tuple):
        return self.__utilities[action_combination]

    @property
    def num_roles(self):
        return self.__num_roles

    @property
    def player_contexts(self):
        return self.__player_contexts

    @property
    def name(self):
        return self.__name

    @property
    def __str__(self):
        return self.name

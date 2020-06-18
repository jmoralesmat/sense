class Norm(object):
    NORM_COUNT = 0

    def __init__(self, context, action):
        self.__context = context
        self.__action = action

        self.NORM_COUNT += 1
        self.__id = self.NORM_COUNT

    @property
    def context(self):
        return self.__context

    @property
    def action(self):
        return self.__action

    def __str__(self):
        return '{}: ({}) -> {}'.format(self.__id, self.__context, self.__action)

    def __eq__(self, other):
        return self.__context == other.context and self.action == other.action

    def __hash__(self):
        return hash(repr(self.context + self.action))


class NormReplicator(object):
    pass

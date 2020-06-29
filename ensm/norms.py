from ensm.games import GamesNetwork
import itertools
import numpy as np


class Norm(object):
    NORM_COUNT = 0

    def __init__(self, context, action, sanction):
        self.__context = context
        self.__action = action
        self.__sanction = sanction

        self.NORM_COUNT += 1
        self.__id = self.NORM_COUNT

    @property
    def context(self):
        return self.__context

    @property
    def action(self):
        return self.__action

    @property
    def sanction(self):
        return self.__sanction

    def __str__(self):
        return '{}: ({}) -> {} / {}'.format(self.__id, self.__context, self.__action, self.__sanction)

    def __eq__(self, other):
        return self.__context == other.context and self.action == other.action

    def __hash__(self):
        return hash(repr(self.context + self.action))


class NormReplicator(object):
    def __init__(self):
        pass

    @staticmethod
    def update_utilities(context: str, games_net: GamesNetwork, norm_space: dict, action_freqs: dict,
                         action_freqs_by_norm: dict):
        """

        :param context:
        :param games_net:
        :param norm_space:
        :param action_freqs:
        :param action_freqs_by_norm:
        :return:
        """

        return

        for norm in norm_space:
            mean_utility = np.float64(0)

            for game, role in [(game, role) for game, roles in games_net.played_roles(context) for role in roles]:

                # Get all action combinations that can be played in the game and in which the agent
                # playing the given role is performing the specified action
                action_spaces = [game.action_space(role) for role in range(game.num_roles)]
                action_combinations = [ac for ac in itertools.product(*action_spaces)]

                utility_in_game = game.utility(action_combination)
                action_freqs  # TODO to compute action combination frequencies in game



        """
        AverageActionPayoffs avgAcUtilities = new AverageActionPayoffs();
    
        /* Get the set of games that an agent plays when perceiving the context */
        Map<Game,List<Integer>> playedRoles = cNet.getPlayedRoles(ctxt);
        for(Game game : playedRoles.keySet()) {
            for(int role=0; role<game.getNumRoles(); role++) {
                for(String action : game.getActionSpace(role)) {
    
                    /* Compute the fitness of the strategy and keep track of it 
                     * (limit to 100 decimals) */
                    BigDecimal avgAcUtility = computeAvgActionUtility(role, 
                            action, game,	avgAcProbs);
    
                    avgAcUtility = avgAcUtility.setScale(100, BigDecimal.ROUND_HALF_UP);
    
                    /* Keep track of the average utility of the pair <role,action> */
                    avgAcUtilities.set(role, action, game, avgAcUtility);
                }
            }
        }
        return avgAcUtilities;
        """

    @staticmethod
    def replicate(context, norm_freqs, norm_utilities):
        pass

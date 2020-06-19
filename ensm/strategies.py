from ensm.agents import AgentSubPopulation
from ensm.games import GamesNetwork, Game
from ensm.norms import Norm

from typing import List
import numpy as np
import itertools


class StrategyReplicator(object):

    @staticmethod
    def update_fitness(sub_population: AgentSubPopulation, games_net: GamesNetwork,
                       action_spaces: dict, norm_spaces: dict, mean_action_freqs_by_game: dict,
                       fitness_aggregation):
        """

        :param sub_population:
        :param games_net:
        :param action_spaces:
        :param norm_spaces:
        :param mean_action_freqs_by_game:
        :param fitness_aggregation:
        :return:
        """

        for context in games_net.contexts:
            for action in action_spaces[context]:
                for norm in norm_spaces[context]:
                    all_fitnesses = []

                    # Compute the fitness values of the sub-population in each co-dependent game that
                    # they play when they perceive the context
                    all_played_roles = games_net.played_roles(context).items()
                    for game, role in [(game, role) for game, roles in all_played_roles for role in roles]:
                        fitness_in_game = StrategyReplicator.__compute_fitness_in_game(
                            game=game, role=role, action=action, norm=norm, sub_population=sub_population,
                            mean_action_freqs_by_game=mean_action_freqs_by_game
                        )
                        all_fitnesses.append(fitness_in_game)

                    # Aggregate all fitness using a pre-defined fitness aggregation function
                    sub_population.fitness[context][norm][action] = fitness_aggregation(all_fitnesses)

    @staticmethod
    def __compute_fitness_in_game(game: Game, role: int, action: str, norm: Norm,
                                  sub_population: AgentSubPopulation, mean_action_freqs_by_game: dict):
        """

        :param game:
        :param role:
        :param action:
        :param norm:
        :param sub_population:
        :param mean_action_freqs_by_game:
        :return:
        """
        fitness = np.float64(0)

        # Get all action combinations that can be played in the game and in which the agent
        # playing the given role is performing the specified action
        action_spaces = [game.action_space(role) for role in range(game.num_roles)]
        action_combinations = [ac for ac in itertools.product(*action_spaces) if ac[role] == action]

        # Get the payoff of the player with the given role once all the players of the game
        # perform each combination of actions. Accumulate the payoff in the fitness computation
        # by weighting it with the frequency with which the action combination is played in the game,
        # computed as the joint mean frequency of each action in the action combination
        for action_combination in action_combinations:
            payoff_in_game_role = sub_population.payoff[game][action_combination][role]

            joint_action_freq = StrategyReplicator.__get_action_combination_frequency(
                game=game, action_combination=action_combination, reference_role=role,
                mean_action_freqs_by_game=mean_action_freqs_by_game
            )
            fitness += payoff_in_game_role * joint_action_freq

        return fitness

    """
    BigDecimal fitness = BigDecimal.ZERO;
    int numRoles = game.getNumRoles();

    /* Retrieve all the action combinations in which 
     * the agent playing role i performs the action */
    List<Combination<String>> allAcCombns = 
            GameUtilities.getActionCombinations(game, numRoles);

    /* Retrieve those strategy combinations in which player i has 
     * adopted the strategy of which we are computing the fitness */
    List<Combination<String>> acCombns = 
            new ArrayList<Combination<String>>();

    for(Combination<String> ac : allAcCombns) {
        if(ac.get(role).equals(action)) {
            acCombns.add(ac);
        }
    }

    /* For each possible strategy combination, get the payoff to player i 
     * once the players of the game perform the combination of actions 
     * dictated by their respective strategies in the strategy combination.
     * Once the payoff is retrieved, add it to the fitness computation
     * by weighting it with the probability that the agents play 
     * the game with that combination of strategies */
    for(Combination<String> ac : acCombns) {

        /* Compute reduced payoff of the triple <role,action,norm>*/
        BigDecimal redPayoff = this.getReducedPayoff(ac, role, 
                action, norm, game, pFunc);

        /* Get the joint probability that the agents play the game 
         * by adopting that combination of strategies */
        BigDecimal intrcProb = GameUtilities.getJointProbability(game, ac, 
                role, avgAcProbs);

        /* Computed weighted payoff */
        BigDecimal weightedPayoff = redPayoff.multiply(intrcProb);
        fitness = fitness.add(weightedPayoff);
    }
    return fitness;
    """

    @staticmethod
    def __get_action_combination_frequency(game: Game, action_combination: tuple, reference_role: int,
                                           mean_action_freqs_by_game: dict):

        """
        Returns the joint probability that the agents of a MAS will play the game by adopting
        a given combination of actions when playing each role
        :return:
        """
        # Get roles not played by the reference agent
        roles = [r for r in range(len(action_combination)) if r != reference_role]

        # Get the global (mean) frequency of the action played by each role and mutiply them all
        # to compute the frequency of the action combination
        action_freqs = [mean_action_freqs_by_game[game][role][action_combination[role]] for role in roles]
        action_comb_freq = np.prod(action_freqs)

        return action_comb_freq

    """
    public static BigDecimal getJointProbability(Game game, 
        Combination<String> acCombn, int refRole, 
        AverageActionProbabilities avgAcProbs) {

    int numRoles = acCombn.size();
    BigDecimal jointProb = BigDecimal.ONE;

    /* Compute the probability that a group of agents interact while 
     * having the strategy combination (computed as the joint frequency 
     * of each one of the strategy in the strategy combination */
    for(int role=0; role<numRoles; role++) {

        /* We do not consider the frequency of the strategy applicable to 
         * the role that is being evaluated, only the frequency of 
         * the strategies applicable to the other roles */
        if(role == refRole) {
            continue;
        }
        String playerAction = acCombn.get(role);
        BigDecimal avgAcProb = avgAcProbs.get(role, playerAction, game); // TODO WRONG!!!
        jointProb = jointProb.multiply(avgAcProb);
    }
    return jointProb;
}
    """
    @staticmethod
    def replicate(sub_population: List[AgentSubPopulation], games_net: GamesNetwork):
        pass

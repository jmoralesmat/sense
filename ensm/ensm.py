import numpy as np


class ENSM(object):
    def __init__(self, mas, max_generations):
        self.__max_generations = max_generations
        self.__mas = mas

        self.__must_evolve_norms = True
        self.__num_generations = 0
        self.__new_norms = []

        self.__converged = False
        self.__timeout = False

    def evolve(self):
        """

        :return:
        """
        self.__num_generations += 1
        self.__new_norms = []

        # Update the strategy probabilities of each agent profile based on the
        # frequencies of the norms that they are provided with
        self.__evolve_strategies()
        self.__update_action_probas()

        # Evaluate norms in terms of their utility to achieve the MAS goals. Replicate norms based on their utility
        if self.__must_evolve_norms:
            self.__evolve_norms()

        # Adjust norm frequencies (because of norm reproduction)
        self.__adjust_norm_frequencies()

        fitnesses = [self.__get_fitness(sub_population) for sub_population in self.mas.population]

        self.__converged = True  # TODO Change this
        self.__timeout = self.__num_generations > self.__max_generations

        return fitnesses

    def __evolve_strategies(self):
        pass

    def __update_action_probas(self):
        pass

    def __evolve_norms(self):
        pass

    def adjust_norm_frequencies(self):
        pass

    def __adjust_norm_frequencies(self):
        pass

    def __get_fitness(self, profile):
        return np.float64(0)

    @property
    def mas(self):
        return self.__mas

    @property
    def num_generations(self):
        return self.__num_generations

    @property
    def converged(self):
        return self.__converged

# -*- coding:utf-8 -*-

from operator import itemgetter
import copy
import math
from VariablesFunctions import *
from lstm_ga import Genetic
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from keras import backend as BK


def evolutionary_NN(nb_gens):
    # set GPU memory
    if 'tensorflow' == BK.backend():
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    genome = Genetic()
    population = genome.create_population(strength=GEN_LENGTH)
    with open(lstm_ga_log_file, 'w') as lf:
        lf.write('population num:\t' + str(GEN_LENGTH) + '\n')
        lf.write('generation num:\t' + str(NB_GENS) + '\n\n')
        lf.close()

    max_score = []
    min_score = []
    avg_score = []
    fittest_member_list = []
    for _gen_nb in range(nb_gens):
        with open(lstm_bpnn_ga_log_file, 'a') as lf:
            lf.write('****** current gen:\t' + str(_gen_nb) + ' ******\n')
            lf.close()
        # mutation_rate = INIT_MUTATION * (1.0 - np.random.rand() ** (1.0 - float(_gen_nb) / float(nb_gens)))
        fitness_scores, last_fittest_members, new_population = \
            genome.evolve(population)
        # visualize scores of each generation
        max_score.append(fitness_scores[0])
        min_score.append(fitness_scores[-1])
        avg_score.append(np.mean(fitness_scores))
        population = new_population
        fittest_member_list.append(copy.deepcopy(last_fittest_members[0]))
    # the final population is (probably) the fittest population
    trained_pop = list()
    for member in population:
        fitness_score = genome.evaluate_fitness(member)
        if math.isnan(fitness_score):
            continue
        else:
            trained_pop.append((fitness_score, member))
    # arrange members according to their fitness scores
    sorted_scores = sorted(trained_pop, key=itemgetter(0), reverse=True)
    fitness_scores = [pair[0] for pair in sorted_scores]
    members = [pair[1] for pair in sorted_scores]
    # complete the scores for all generations
    max_score.append(fitness_scores[0])
    min_score.append(fitness_scores[-1])
    avg_score.append(np.mean(fitness_scores))
    # select the best member in the last generation
    fitness_score = fitness_scores[0]
    fittest_member = members[0]
    fittest_member_list.append(copy.deepcopy(fittest_member))

    with open(lstm_bpnn_ga_log_file, 'a') as lf:
        lf.write('****** last generation ******\n')
        for index, member in enumerate(members):
            lf.write('member ' + str(index) + '\n')
            for key in member:
                lf.write(key + ':\t' + str(member[key]) + '\n')
            lf.write('score:\t' + str(fitness_scores[index]) + '\n')
        lf.close()

    # the best in all the generations
    final_max_score = np.max(max_score)
    final_fittest_member = fittest_member_list[max_score.index(final_max_score)]

    # and print its parameters
    with open(lstm_bpnn_ga_result_file, 'w') as ga_f:
        ga_f.write("The best member in the last generation:\n")
        print("The best member in the last generation:")
        for param in genome.param_choices:
            print(param, ": ", fittest_member[param])
            ga_f.write(param + ": " + str(fittest_member[param]) + '\n')
        print("Highest fitness score: ", fitness_score)
        ga_f.write("Highest fitness score: " + str(fitness_score))

        ga_f.write("\n\nThe best member in all the generations:\n")
        print("The best member in all the generations:")
        for param in genome.param_choices:
            print(param, ": ", final_fittest_member[param])
            ga_f.write(param + ": " + str(final_fittest_member[param]) + '\n')
        print("Highest fitness score: ", final_max_score)
        ga_f.write("Highest fitness score: " + str(final_max_score))
        ga_f.close()


if __name__ == '__main__':
    # np.random.seed(0)
    evolutionary_NN(NB_GENS)

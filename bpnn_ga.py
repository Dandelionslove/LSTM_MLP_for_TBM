# -*- coding:utf-8 -*-

from operator import itemgetter
import math
from VariablesFunctions import *
from bpnn_model import BPNN
import numpy as np
import copy


class Genetic(object):
    def __init__(self):
        self.nb_hidden_layers = range(4, 13)
        self.nb_hidden_nodes = [i for i in range(4, bp_input_length*3, 4)]
        self.activations = ["relu", "sigmoid", "tanh",  'selu', 'elu']
        self.optimizers = ["Adam", "Adagrad", "Adadelta", "sgd"]
        self.init_lr = [1e-1, 1e-2, 3e-2, 1e-3]
        self.batch_sizes = [32, 64, 96, 128, 256]
        self.param_choices = ['hidden_layer_num',
                              'hidden_layer_dims',
                              'activation',
                              'lr',
                              'batch_size',
                              'optimizer']

    def create_population(self, strength):
        '''
        :param strength: number of members in the generation, each member with parameters
        :return: members
        '''
        members = []
        for _ in range(strength):
            parameters = dict()
            bp_hidden_layer_num = np.random.choice(self.nb_hidden_layers)
            parameters['hidden_layer_num'] = bp_hidden_layer_num
            parameters['hidden_layer_dims'] = list()
            # 构建多个不同的模型
            for _ in range(bp_hidden_layer_num):
                parameters['hidden_layer_dims'].append(np.random.choice(self.nb_hidden_nodes))
            parameters['optimizer'] = np.random.choice(self.optimizers)
            parameters['activation'] = np.random.choice(self.activations)
            parameters['lr'] = np.random.choice(self.init_lr)
            parameters['batch_size'] = np.random.choice(self.batch_sizes)
            members.append(parameters)
        return members

    # 遗传算法的培育操作
    def breed(self, mother, father, mother_score, father_score, pc, K=1):
        '''
        :param mother: parameters dictionary of a neural net
        :param father: parameters dictionary of another neural net
        @:param fitness_scores: list
        @:param mother_score: float
        @:param father_score: float
        :param K: number of children required
        :return: children
        '''
        children = []
        for _ in range(K):
            child = dict()
            if mother_score > father_score:
                p_dom = mother
                p_sub = father
            else:
                p_dom = father
                p_sub = mother
            # crossover_rate = compute_pc(fitness_scores, mother_score, father_score)
            for gene in self.param_choices:
                child[gene] = p_dom[gene]
            # 交叉
            if np.random.random() < pc:
                # 需要进行交叉
                cross_gene = np.random.choice(self.param_choices)
                if cross_gene == 'hidden_layer_num' or cross_gene == 'hidden_layer_dims':
                    child['hidden_layer_dims'] = p_sub['hidden_layer_dims']
                    child['hidden_layer_num'] = len(child['hidden_layer_dims'])
                else:  # optimizer or batch_size or lr or activation
                    child[cross_gene] = p_sub[cross_gene]
            else:
                child = self.mutate(child)
            children.append(child)
        return children

    # 遗传算法的变异操作
    def mutate(self, nn_parameters):
        '''
        randomly mutate parameters of the NN
        :param nn_parameters: dict of parameters of the network
        :return: nn_parameters after mutating
        '''

        param_to_be_mutated = np.random.choice(self.param_choices)
        # bp_hidden_layer_num
        if param_to_be_mutated == 'hidden_layer_num':
            new_layer_num = np.random.choice(self.nb_hidden_layers)
            if new_layer_num <= nn_parameters['hidden_layer_num']:
                for _ in range(nn_parameters['hidden_layer_num'] - new_layer_num):
                    nn_parameters['hidden_layer_dims'].pop()
            else:
                for _ in range(new_layer_num - nn_parameters['hidden_layer_num']):
                    nn_parameters['hidden_layer_dims'].append(np.random.choice(self.nb_hidden_nodes))
            nn_parameters['hidden_layer_num'] = len(nn_parameters['hidden_layer_dims'])
        # hidden layer dims
        elif param_to_be_mutated == 'hidden_layer_dims':
            index = np.random.choice(range(nn_parameters['hidden_layer_num']))
            new_dim = np.random.choice(self.nb_hidden_nodes)
            nn_parameters['hidden_layer_dims'][index] = new_dim
        elif param_to_be_mutated == 'activation':
            nn_parameters['activation'] = np.random.choice(self.activations)
        elif param_to_be_mutated == 'lr':
            nn_parameters['lr'] = np.random.choice(self.init_lr)
        elif param_to_be_mutated == 'batch_size':
            nn_parameters['batch_size'] = np.random.choice(self.batch_sizes)
        # optimizer
        else:
            nn_parameters['optimizer'] = np.random.choice(self.optimizers)

        return nn_parameters

    def evolve(self, population):
        # get fitness scores for all members
        trained_pop = list()
        for m_index, member in enumerate(population):
            fitness_score = self.evaluate_fitness(member)
            if math.isnan(fitness_score):
                continue
            else:
                trained_pop.append((fitness_score, member))
        # trained_pop = [(self.evaluate_fitness(member, model_name='population_'+str(i)), member)
        #                for i, member in enumerate(population)]

        # arrange members according to their fitness scores, 降序排列
        sorted_scores = sorted(trained_pop, key=itemgetter(0), reverse=True)
        members = [pair[1] for pair in sorted_scores]
        fitness_scores = [pair[0] for pair in sorted_scores]

        # best_member = members[0]
        # best_member_score = fitness_scores[0]

        with open(bpnn_ga_log_file, 'a') as lf:
            for index, member in enumerate(members):
                lf.write('member ' + str(index) + '\n')
                for key in member:
                    lf.write(key + ':\t' + str(member[key]) + '\n')
                lf.write('score:\t' + str(fitness_scores[index]) + '\n')
                # lf.write('parents length:\t' + str(len(parents)) + '\n')
                # lf.write('children length:\t' + str(len(children)) + '\n\n')
            lf.close()

        retain_length = int(np.ceil(len(members) * retain_value))
        parents = members[:retain_length]
        parents_scores = fitness_scores[:retain_length]
        for _i, _member in enumerate(members[retain_length:]):
            if np.random.random() < select_value:
                parents.append(_member)
                parents_scores.append(fitness_scores[_i + retain_length])

        # time to make babies
        # for now, preserve the strength of population
        nb_babies = len(population) - len(parents)
        children = []
        while len(children) < nb_babies:
            mother_index = np.random.choice(range(len(parents)))
            mother = parents[mother_index]
            mother_fitness_score = parents_scores[mother_index]
            father_index = np.random.choice(range(len(parents)))
            while father_index == mother_index:
                father_index = np.random.choice(range(len(parents)))
            father = parents[father_index]
            father_fitness_score = parents_scores[father_index]

            # mutate
            for p_i, p_score in enumerate(parents_scores):
                pm = compute_pm(parents_scores, p_score)
                if np.random.random() < pm:
                    parents[p_i] = self.mutate(parents[p_i])

            # breed
            K = 1 + int(3 * np.random.random())
            pc = compute_pc(parents_scores, mother_fitness_score, father_fitness_score)
            babies = self.breed(mother, father, mother_fitness_score, father_fitness_score, pc, K)
            for baby in babies:
                if len(children) < nb_babies:
                    children.append(baby)

        if len(children) > 0:
            parents.extend(children)

        return fitness_scores, members, parents

    def evaluate_fitness(self, nn_parameters):
        # if nn_parameters is None:
        #     return None
        nn = BPNN(nn_parameters)
        fitness_score = nn.train()
        return fitness_score


if __name__ == '__main__':
    # ga = Genetic()
    # population = ga.create_population(3)
    # print(population)
    # pass
    a = dict()
    b = dict()
    a['a'] = 1
    a['b'] = 2
    b['a'] = a['a']
    b['b'] = a['b']
    b['b'] = 3
    print(a)
    print(b)
    print(id(a))
    print(id(b))

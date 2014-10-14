#!/usr/bin/env python
import numpy as np
import MultiNEAT as NEAT

import warnings
warnings.simplefilter("error")

# TODO Make these settable params
BASE_LEARNING_RATE = 0.1
MOVING_AVERAGE_ALPHA = 0.1

class Network(object):
    def __init__(self, substrate, nfeatures):
        self.substrate = substrate
        self.nfeatures = nfeatures
        self.feature_values = np.zeros(nfeatures) # TODO: Include bias.
        self.output_weights = np.zeros(nfeatures) # TODO: Include bias.
        self.squared_norm_ema = 0.0
        self.genome_archive = {} # gID: [index, weight_magnitude_ema, feature_net]

    def _calculate_feature_values(self, genome_list, input_vals):
        for genome in genome_list:
            feature = self.genome_archive[genome.GetID()][2]
            feature.Flush()
            feature.Input(input_vals)
            # FIXME: Depth should not be hardcoded.
            for _ in xrange(2): # This is supposed to be depth? Can one use genome.GetDepth or something?
                feature.Activate()

            idx = self.genome_archive[genome.GetID()][0]
            self.feature_values[idx] = feature.Output()[0]

    def median_weight_magnitudes_ema(self):
        return np.median([val[1] for val in self.genome_archive.values()])

    def train(self, genome_list, input_vals, target, iteration):
        # TODO: genome_archive is a quite ugly hack.
        assert(len(genome_list) == len(self.output_weights))
        self._calculate_feature_values(genome_list, input_vals)

        squared_feature_norm = np.dot(self.feature_values, self.feature_values)

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        output = np.dot(self.feature_values, self.output_weights)
        error = target - output
        
        # TODO: Do backprop?

        for i in xrange(len(genome_list)):
            g = self.genome_archive[genome_list[i].GetID()]
            self.output_weights[g[0]] += \
                    learning_rate * error * self.feature_values[g[0]]
            g[1] = (MOVING_AVERAGE_ALPHA * abs(self.output_weights[g[0]])
                    + (1 - MOVING_AVERAGE_ALPHA) * g[1])

        return error**2, output

    def activate(self, genome_list, input_vals):
        self._calculate_feature_values(genome_list, input_vals)

        return np.dot(self.feature_values, self.output_weights)

class NEATOeelm(object):
    def __init__(self, neat_params, neat_genome, neat_substrate, noutputs):
        self.population = NEAT.Population(neat_genome, neat_params, True, 1.0)
        self.noutputs = noutputs
        self.substrate = neat_substrate
        self.net = Network(neat_substrate, neat_params.PopulationSize)
        # TODO: Should this be here or left to caller?
        self.generation = 0

    def train(self, input_data, target):
        genome_list = NEAT.GetGenomeList(self.population)
        if self.generation == 0:
            self._init_genome_archive(genome_list)
        assert(len(genome_list) == self.net.nfeatures)

        squared_error, output = self.net.train(genome_list, input_data, target,
                                     self.generation)
        self._zip_fitness(genome_list)

        deleted_genome = NEAT.Genome()
        new_genome = self.population.Tick(deleted_genome)
        # TODO: Move to function. Does same in _init_genome_archive
        old_idx = self.net.genome_archive[deleted_genome.GetID()][0]
        feature = NEAT.NeuralNetwork()
        new_genome.BuildHyperNEATPhenotype(feature, self.substrate)
        self.net.genome_archive[new_genome.GetID()] = [
                old_idx,
                self.net.median_weight_magnitudes_ema(),
                feature     
        ]
        self.net.output_weights[old_idx] = 0.0
        del self.net.genome_archive[deleted_genome.GetID()]

        self.generation += 1

        return (squared_error, output)

    def _init_genome_archive(self, genome_list):
        for i in xrange(len(genome_list)):
            g_id = genome_list[i].GetID()
            if g_id not in self.net.genome_archive:
                feature = NEAT.NeuralNetwork()
                genome_list[i].BuildHyperNEATPhenotype(feature, self.substrate)
                self.net.genome_archive[g_id] = [
                        i,
                        0.0,
                        feature
                ]

    def _zip_fitness(self, genome_list):
        for genome in genome_list:
            fitness = self.net.genome_archive[genome.GetID()][1]
            genome.SetFitness(fitness)
            genome.SetEvaluated()

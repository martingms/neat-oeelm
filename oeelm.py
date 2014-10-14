#!/usr/bin/env python
import numpy as np
import MultiNEAT as NEAT

import warnings
warnings.simplefilter("error")

"""
import cv2
cv2.namedWindow('NN', 0)
cv2.namedWindow('CPPN', 0)
"""

# TODO Make these settable params
BASE_LEARNING_RATE = 0.1
MOVING_AVERAGE_ALPHA = 0.1

class Network(object):
    def __init__(self, substrate, nfeatures):
        self.substrate = substrate
        self.nfeatures = nfeatures
        self.output_weights = np.zeros(nfeatures) # TODO: Include bias.
        self.squared_norm_ema = 0.0
        # TODO: Now that we use .Tick(), we might be able to keep track of this
        # in a better way.
        self.genome_archive = {} # gID: [weight, weight_magnitude_ema, feature_net]

    def _calculate_feature_values(self, genome_list, input_vals):
        def evaluate(genome):
            feature = self.genome_archive[genome.GetID()][2]
            feature.Flush()
            feature.Input(input_vals)
            # FIXME: Depth should not be hardcoded.
            for _ in xrange(2): # This is supposed to be depth? Can one use genome.GetDepth or something?
                feature.Activate()
            return feature.Output()[0]

        return np.asarray([evaluate(genome) for genome in genome_list])

    def median_weight_magnitudes_ema(self):
        return np.median([val[1] for val in self.genome_archive.values()])

    def train(self, genome_list, input_vals, target, iteration):
        # TODO: genome_archive is a quite ugly hack.
        assert(len(genome_list) == len(self.output_weights))
        feature_values = self._calculate_feature_values(genome_list, input_vals)

        squared_feature_norm = np.dot(feature_values, feature_values)

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               
        output = np.dot(feature_values, output_weights)
        error = target - output
        
        # TODO: Do backprop?

        for i in xrange(len(genome_list)):
            g = self.genome_archive[genome_list[i].GetID()]
            g[0] += learning_rate * error * feature_values[i]
            g[1] = (MOVING_AVERAGE_ALPHA * abs(g[0])
                    + (1 - MOVING_AVERAGE_ALPHA) * g[1])

        return error**2, output

    def activate(self, genome_list, input_vals):
        feature_values = self._calculate_feature_values(genome_list, input_vals)
        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               

        return np.dot(feature_values, output_weights)

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
        del self.net.genome_archive[deleted_genome.GetID()]
        # TODO: Move to function. Does same in _init_genome_archive
        feature = NEAT.NeuralNetwork()
        new_genome.BuildHyperNEATPhenotype(feature, self.substrate)
        self.net.genome_archive[new_genome.GetID()] = [
                0.0,
                self.net.median_weight_magnitudes_ema(),
                feature     
        ]

        self.generation += 1

        return (squared_error, output)

    def _init_genome_archive(self, genome_list):
        for genome in genome_list:
            g_id = genome.GetID()
            if g_id not in self.net.genome_archive:
                feature = NEAT.NeuralNetwork()
                genome.BuildHyperNEATPhenotype(feature, self.substrate) 
                self.net.genome_archive[g_id] = [
                        0.0,
                        0.0,
                        feature
                ]

    def _zip_fitness(self, genome_list):
        for genome in genome_list:
            fitness = self.net.genome_archive[genome.GetID()][1]
            genome.SetFitness(fitness)
            genome.SetEvaluated()

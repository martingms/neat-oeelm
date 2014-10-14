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
        self.genome_archive = {} # gID: [weight, weight_magnitude_ema]

    def calculate_feature_values(self, genome_list, input_vals):
        def evaluate(genome):
            # TODO: Visualization of all kinds should be callable.
            """
            net = NEAT.NeuralNetwork()
            genome.BuildPhenotype(net) 
            img = np.zeros((250, 250, 3), dtype=np.uint8)
            img += 10
            NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
            cv2.imshow("CPPN", img)
            """

            # TODO: See if this can be moved outside loop.
            feature = NEAT.NeuralNetwork()
            genome.BuildHyperNEATPhenotype(feature, self.substrate) 
            """
            img = np.zeros((250,250,3), dtype=np.uint8)
            img += 10
            NEAT.DrawPhenotype(img, (0, 0, 250, 250), feature, substrate=True)
            cv2.imshow("NN", img)
            cv2.waitKey(1)
            """

            feature.Flush()
            feature.Input(input_vals)
            # FIXME: See if this is really necessary.
            for _ in range(2): # This is supposed to be depth? Can one use genome.GetDepth or something?
                feature.Activate()
            return feature.Output()[0]

        return np.asarray([evaluate(genome) for genome in genome_list])

    def train(self, genome_list, input_vals, target, iteration):
        # TODO: genome_archive is a quite ugly hack.
        assert(len(genome_list) == len(self.output_weights))
        feature_values = self.calculate_feature_values(genome_list, input_vals)

        squared_feature_norm = np.dot(feature_values, feature_values)
        median_w_m_ema = 0

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)
            median_w_m_ema = np.median([val[1] for val in self.genome_archive.values()])

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        # FIXME: With .Tick() we know that this is only one element.
        for i in range(len(genome_list)):
            g_id = genome_list[i].GetID()
            if g_id not in self.genome_archive:
                self.genome_archive[g_id] = [0.0, median_w_m_ema]

        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               
        output = np.dot(feature_values, output_weights)
        error = target - output
        
        # TODO: Do backprop?

        for i in range(len(genome_list)):
            g = self.genome_archive[genome_list[i].GetID()]
            g[0] += learning_rate * error * feature_values[i]
            g[1] = (MOVING_AVERAGE_ALPHA * abs(g[0])
                    + (1 - MOVING_AVERAGE_ALPHA) * g[1])

        return error**2, output

    def activate(self, genome_list, input_vals):
        feature_values = self.calculate_feature_values(genome_list, input_vals)
        for i in range(len(genome_list)):
            g_id = genome_list[i].GetID()
            if g_id not in self.genome_archive:
                self.genome_archive[g_id] = [0.0, median_w_m_ema]

        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               

        return np.dot(feature_values, output_weights)

class NEATOeelm(object):
    def __init__(self, neat_params, neat_genome, neat_substrate, noutputs):
        self.population = NEAT.Population(neat_genome, neat_params, True, 1.0)
        self.noutputs = noutputs
        self.net = Network(neat_substrate, neat_params.PopulationSize)
        # TODO: Should this be here or left to caller?
        self.generation = 0

    def train(self, input_data, target):
        genome_list = NEAT.GetGenomeList(self.population)
        assert(len(genome_list) == self.net.nfeatures)

        squared_error, output = self.net.train(genome_list, input_data, target,
                                     self.generation)
        self._zip_fitness(genome_list)

        deleted_genome = NEAT.Genome()
        new_genome = self.population.Tick(deleted_genome)
        del self.net.genome_archive[deleted_genome.GetID()]

        self.generation += 1

        return (squared_error, output)

    def _zip_fitness(self, genome_list):
        for genome in genome_list:
            fitness = self.net.genome_archive[genome.GetID()][1]
            genome.SetFitness(fitness)
            genome.SetEvaluated()

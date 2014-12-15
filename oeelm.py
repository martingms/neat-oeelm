#!/usr/bin/env python
import numpy as np
import MultiNEAT as NEAT

import warnings
#warnings.simplefilter("error")

# TODO Make these settable params
BASE_LEARNING_RATE = 0.1
MOVING_AVERAGE_ALPHA = 0.1

# TODO: remove
def isNaN(a):
    return np.isnan(np.sum(a))

class Network(object):
    def __init__(self, ninputs, nfeatures, noutputs):
        self.nfeatures = nfeatures
        self.feature_weights = np.zeros((nfeatures, ninputs))
        self.feature_values = np.zeros(nfeatures + 1)
        self.feature_values[-1] = 1.
        self.output_weights = np.zeros((nfeatures + 1, noutputs))
        self.weight_magnitude_emas = np.zeros(nfeatures)
        self.squared_norm_ema = 0.0
        self.genome_archive = {} # gID: index

    def _sigmoid(self, val):
        # Signed version:
        #return ((1. / (1 + np.exp(-1.0 * val))) - 0.5) * 2
        return 1. / (1 + np.exp(-1.0 * val))

    def _calculate_feature_values(self, input_vals):
        assert not isNaN(input_vals)
        assert not isNaN(self.feature_weights)
        assert not isNaN(self.feature_values)
        #print "feature_weights:", self.feature_weights
        #print "input_vals:", input_vals
        #print "dot(fw, iv)", np.dot(self.feature_weights, input_vals)
        #print "sigmoided", self._sigmoid(np.dot(self.feature_weights, input_vals))
        # TAG: Might be wrong
        self.feature_values[:-1] = self._sigmoid(np.dot(self.feature_weights,
            input_vals))
        #self.feature_values[:-1] = np.dot(self.feature_weights, input_vals)
        assert not isNaN(self.feature_values)
        #print self.feature_values
        #print "f_v min:", np.min(self.feature_values)

    def median_weight_magnitudes_ema(self):
        return np.median(self.weight_magnitude_emas)

    def train(self, genome_list, input_vals, target, iteration):
        assert not isNaN(input_vals)
        assert not isNaN(target)
        self._calculate_feature_values(input_vals)
        #print "FEATURE VALUES:", self.feature_values
        assert not isNaN(self.feature_values)

        squared_feature_norm = np.dot(self.feature_values, self.feature_values)

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        #output = self._sigmoid(np.dot(self.feature_values, self.output_weights))
        output = np.dot(self.feature_values, self.output_weights)
        assert not isNaN(output)
        error = target - output
        #print "OUTPUT:", output, "TARGET:", target, "ERR:", error
        assert not isNaN(target)
        assert not isNaN(error)
        
        # TODO: Do backprop?

        # FIXME: This should probably be done in the OEELM-class, so the
        # network does not need to know about the genome_list.
        assert not isNaN(self.output_weights)
        assert not isNaN(self.weight_magnitude_emas)
        for i in xrange(len(genome_list)):
            genome = genome_list[i]
            idx = self.genome_archive[genome.GetID()]
            #print "LR:", learning_rate, "ERR:", error, "FV", \
                #self.feature_values[idx], "OWBefore", self.output_weights[idx]
            self.output_weights[idx] += \
                    learning_rate * error * self.feature_values[idx]
            #print "OWafter", self.output_weights[idx]
            #print "WMema before", self.weight_magnitude_emas[idx]
            self.weight_magnitude_emas[idx] = \
                    (MOVING_AVERAGE_ALPHA * max(abs(self.output_weights[idx])) # max/abs changed
                    + (1 - MOVING_AVERAGE_ALPHA) \
                            * self.weight_magnitude_emas[idx])
            #print "WMema after", self.weight_magnitude_emas[idx]
            genome.SetFitness(self.weight_magnitude_emas[idx])
            genome.SetEvaluated()

        assert not isNaN(self.output_weights)

        self.output_weights[-1] += learning_rate * error * 1.

        return error**2, output

    def activate(self, genome_list, input_vals):
        self._calculate_feature_values(genome_list, input_vals)

        output = np.dot(self.feature_values, self.output_weights)
        return _sigmoid(output)

class NEATOeelm(object):
    def __init__(self, neat_params, neat_genome, neat_substrate, ninputs, noutputs):
        self.population = NEAT.Population(neat_genome, neat_params, True, 1.0)

        assert len(neat_substrate) == ninputs
        self.ninputs = ninputs
        self.substrate = neat_substrate

        self.net = Network(ninputs, neat_params.PopulationSize, noutputs)
        # TODO: Should this be here or left to caller?
        self.generation = 0
        genome_list = NEAT.GetGenomeList(self.population)
        self._init_genome_archive(genome_list)

    def train(self, input_data, target):
        genome_list = NEAT.GetGenomeList(self.population)

        #if self.generation == 0:
        #    self._init_genome_archive(genome_list)

        squared_error, output = self.net.train(genome_list, input_data,
                                               target, self.generation)

        # TODO: Config
        for _ in xrange(5):
            self._tick()

        self.generation += 1

        return (squared_error, output)

    def _tick(self):
        deleted_genome = NEAT.Genome()
        new_genome = self.population.Tick(deleted_genome)
        old_idx = self.net.genome_archive[deleted_genome.GetID()]
        del self.net.genome_archive[deleted_genome.GetID()]

        self.net.genome_archive[new_genome.GetID()] = old_idx
        self._generate_feature_weights(new_genome)
        self.net.output_weights[old_idx] = 0.0
        self.net.weight_magnitude_emas[old_idx] = \
                self.net.median_weight_magnitudes_ema()


    def activate(self, input_vals):
        genome_list = NEAT.GetGenomeList(self.population)
        return self.net.activate(genome_list, input_vals)

    def _generate_feature_weights(self, genome):
        cppn = NEAT.NeuralNetwork()
        genome.BuildPhenotype(cppn)
        genome.CalculateDepth()

        assert len(self.substrate) == self.ninputs
        for i in xrange(self.ninputs):
            # TODO: Remove asserts
            assert type(self.substrate[i]) == list
            assert len(self.substrate[i]) == 2
            cppn.Flush()
            cppn.Input(self.substrate[i] + [1.])

            #for _ in xrange(3):
            #for _ in xrange(genome.GetDepth() + 1):
            # FIXME: Incredibly bad proxy for depth, but GetDepth isn't working...
            for _ in xrange(genome.NumNeurons()-3):
                cppn.Activate()

            output = cppn.Output()
            assert len(output) == 1
            output = output[0]

            if output >= 0:
                output = 1
            else:
                output = -1
            #output = NEAT.Clamp(output, -1, 1)
            #if abs(output) > 0.2:
            #    if output < 0:
            #        output = NEAT.Scale(output, -1, -0.2, -5.0, 0)
            #    else:
            #        output = NEAT.Scale(output, 0.2, 1, 0, 5.0)

            g_id = genome.GetID()
            self.net.feature_weights[self.net.genome_archive[g_id]][i] = \
                output


    def _init_genome_archive(self, genome_list):
        for i in xrange(len(genome_list)):
            g_id = genome_list[i].GetID()
            assert g_id not in self.net.genome_archive

            self.net.genome_archive[g_id] = i
            self._generate_feature_weights(genome_list[i])

        assert not isNaN(self.net.feature_weights)
        print "INITIALIZED FEATURE WEIGHTS:", self.net.feature_weights
        print "MEAN:", np.mean(self.net.feature_weights)
        #assert False

#!/usr/bin/env python

import numpy as np
import MultiNEAT as NEAT
#import cv2
#cv2.namedWindow('CPPN', 0)

# TODO Move all params to files.
GENERATIONS = 10000
BASE_LEARNING_RATE = 0.1
MOVING_AVERAGE_ALPHA = 0.1

def get_NEAT_params():
    params = NEAT.Parameters()
    params.PopulationSize = 5

    params.DynamicCompatibility = True
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 1
    params.MaxSpecies = 25
    params.RouletteWheelSelection = False

    params.MutateRemLinkProb = 0.02
    params.RecurrentProb = 0
    params.OverallMutationRate = 0.15
    params.MutateAddLinkProb = 0.08
    params.MutateAddNeuronProb = 0.01
    params.MutateWeightsProb = 0.90
    params.MaxWeight = 8.0
    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 1.0

    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.5
    params.MinActivationA = -2.0 # TODO
    params.MaxActivationA = 6.0

    params.MutateNeuronActivationTypeProb = 0.03;

    # Probabilities for a particular activation function appearance
    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
    params.ActivationFunction_Tanh_Prob = 1.0;
    params.ActivationFunction_TanhCubic_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 1.0;
    params.ActivationFunction_UnsignedStep_Prob = 0.0;
    params.ActivationFunction_SignedGauss_Prob = 1.0;
    params.ActivationFunction_UnsignedGauss_Prob = 0.0;
    params.ActivationFunction_Abs_Prob = 0.0;
    params.ActivationFunction_SignedSine_Prob = 1.0;
    params.ActivationFunction_UnsignedSine_Prob = 0.0;
    params.ActivationFunction_Linear_Prob = 1.0;

    return params

class Network(object):
    def __init__(self, inputmapping, noutputs, nfeatures):
        self.inputmapping = inputmapping
        self.noutputs = noutputs
        self.nfeatures = nfeatures
        x, y, _ = self.inputmapping.shape
        self.feature_weights = np.zeros((nfeatures, x, y))
        self.output_weights = np.zeros(nfeatures + 1) # Including bias.
        self.squared_norm_ema = 0.0
        self.weight_magnitudes_ema = abs(self.output_weights)

    def calculate_feature_weights(self, cppn):
        x, y, _ = self.inputmapping.shape
        weights = np.zeros((x,y)) # TODO: dtype?

        #img = np.zeros((250, 250, 3), dtype=np.uint8)
        #NEAT.DrawPhenotype(img, (0, 0, 250, 250), cppn)
        #cv2.imshow("CPPN", img)
        #cv2.waitKey(10)
        for x, y in np.ndindex(weights.shape):
            values = self.inputmapping[x][y]
            cppn.Input(values)
            cppn.Activate()
            output = cppn.Output()
            assert(len(output) == 1)
            weights[x][y] = output[0]

        return weights

    def calculate_all_feature_weights(self, genome_list):
        print "len(genome_list)", len(genome_list)
        print "len(feature_weights)", len(self.feature_weights)
        assert(len(genome_list) == len(self.feature_weights))
        # TODO: Hopefully this can be reused?
        cppn = NEAT.NeuralNetwork()
        for i in range(len(genome_list)):
            genome_list[i].BuildPhenotype(cppn)
            # TODO: Probably  better to modfiy instead of replacing.
            self.feature_weights[i] = self.calculate_feature_weights(cppn)

    def calculate_feature_values(self, input_vals):
        # TODO: J. Auerbach has sigmoid output as an option too.
        feature_inputs = np.ndarray.flatten(np.dot(self.feature_weights, input_vals))
        print "FEATURE_INPUTS:", feature_inputs
        
        return np.hstack((np.asarray(np.greater(feature_inputs,
                                               0.5), # TODO: Evolve thresholds aswell?
                                    dtype=float),
                         np.asarray([1.]))) # TODO: Make bias an option?

    def train(self, input_vals, target, iteration):
        print "\n\n\n"
        print "========================================================"
        print "INPUT:", input_vals
        print "FEATUTRE_WEIGHTS:", self.feature_weights
        feature_values = self.calculate_feature_values(input_vals)
        print "FEATURE_VALUES:", feature_values

        squared_feature_norm = np.dot(feature_values, feature_values)

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        print "OUTPUT_WEIGHTS:", self.output_weights
        output = np.dot(self.output_weights, feature_values)
        print "OUTPUT:", output
        print "TARGET:", target
        print "========================================================"
        print "\n\n\n"
        error = target - output

        self.output_weights += learning_rate * error * feature_values

        # TODO: Do backprop?

        self.weight_magnitutes_ema = (
                MOVING_AVERAGE_ALPHA * abs(self.output_weights)
                + (1 - MOVING_AVERAGE_ALPHA) * self.weight_magnitudes_ema)

        return error**2

# TODO: Move to function, make module conveniently callable, and move
# experiment code in different file.
if __name__ == "__main__":
    params = get_NEAT_params()

    # 2 inputs, 1 output
    genome = NEAT.Genome(0, 2, 0, 1, False,
        NEAT.ActivationFunction.UNSIGNED_SIGMOID,
        NEAT.ActivationFunction.UNSIGNED_SIGMOID,
        0, params)

    pop = NEAT.Population(genome, params, True, 1.0)

    inputmapping = np.asarray([[(0,-2), (0,-1), (0,0), (0,1), (0,2)]])

    net = Network(inputmapping, 1, 5)

    #### Creating random function, code stolen from oeelm.py
    import oeelm # J. Auerbach
    import time
    from numpy.random import RandomState 

    rng = RandomState(int(time.time()))
    target_features = [oeelm.BinaryFeature.create_random(rng, 5, 0.6)
                       for _ in range(5)]

    target_weights = np.concatenate((rng.normal(0,1,5),
                                     [0])) 

    target_func = oeelm.Learner(target_features, target_weights)

    ###########################

    for generation in range(GENERATIONS):
        genome_list = NEAT.GetGenomeList(pop)

        if genome_list > net.nfeatures:
            genome_list = genome_list[:5]
        net.calculate_all_feature_weights(genome_list)
        # TODO: This is a hack since we're unable to know which features are new
        # (in which case we should set median w_m_ema, so we let them prove
        # themselves for a couple of generations before we check fitness.
        for i in range(50):
            rand_input = rng.rand(5)
            target = target_func.get_output(rand_input) # TODO: Add noise?
             
            mse = net.train(rand_input, target, i)

            # TODO: Set fitness!
            # THIS AINT WORKING
            #for j in range(len(net.weight_magnitudes_ema)):
            #   genome_list[i].SetFitness(net.weight_magnitudes_ema[i])

            #print "MSE:", mse

        pop.Epoch()

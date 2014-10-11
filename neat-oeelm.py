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

# TODO Move all params to files.
GENERATIONS = 10000000
BASE_LEARNING_RATE = 0.1
MOVING_AVERAGE_ALPHA = 0.1

def get_NEAT_params():
    params = NEAT.Parameters()
    params.PopulationSize = 10
    """
    params.YoungAgeFitnessBoost = 1.0 # No boost

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
    """

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
    def __init__(self, substrate, nfeatures):
        self.substrate = substrate
        self.nfeatures = nfeatures
        self.output_weights = np.zeros(nfeatures) # TODO: Include bias.
        self.squared_norm_ema = 0.0
        # TODO: This will continuously grow each generation.
        # An ID that disappears one generation will probably not come back,
        # so can just delete it.
        self.genome_archive = {} # gID: [weight, weight_magnitude_ema]

    def calculate_feature_values(self, genome_list, input_vals):
        def evaluate(genome):
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
        #print "FEATURE_VALUES", feature_values

        squared_feature_norm = np.dot(feature_values, feature_values)
        median_w_m_ema = 0

        if iteration == 0:
            self.squared_norm_ema = squared_feature_norm
        else:
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 +
                    squared_feature_norm * 0.001)
            median_w_m_ema = np.median([val[1] for val in self.genome_archive.values()])

        learning_rate = BASE_LEARNING_RATE / self.squared_norm_ema

        for i in range(len(genome_list)):
            g_id = genome_list[i].GetID()
            if g_id not in self.genome_archive:
                self.genome_archive[g_id] = [0.0, median_w_m_ema]

        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               
        output = np.dot(feature_values, output_weights)
        print "target:", target
        print "output:", output
        #print "error:", target-output
        error = target - output
        
        # TODO: Do backprop?

        for i in range(len(genome_list)):
            g = self.genome_archive[genome_list[i].GetID()]
            g[0] += learning_rate * error * feature_values[i]
            g[1] = (MOVING_AVERAGE_ALPHA * abs(g[0])
                    + (1 - MOVING_AVERAGE_ALPHA) * g[1])

        #weight_magnitude_emas = [val[1] for val in self.genome_archive.values()]
        return error**2

    def activate(self, genome_list, input_vals):
        feature_values = self.calculate_feature_values(genome_list, input_vals)
        for i in range(len(genome_list)):
            g_id = genome_list[i].GetID()
            if g_id not in self.genome_archive:
                self.genome_archive[g_id] = [0.0, median_w_m_ema]

        output_weights = [self.genome_archive[g.GetID()][0] for g in genome_list]               

        return np.dot(feature_values, output_weights)

# TODO: Move to function, make module conveniently callable, and move
# experiment code in different file.
if __name__ == "__main__":
    params = get_NEAT_params()

    #substrate = NEAT.Substrate([(0,-2), (0,-1), (0,0), (0,1), (0,2)],
    substrate = NEAT.Substrate([(0,-1), (0,0), (0,1)],
                               [], # TODO
                               [(2,0)])

    genome = NEAT.Genome(0,
        substrate.GetMinCPPNInputs(),
        0,
        substrate.GetMinCPPNOutputs(),
        False,
        NEAT.ActivationFunction.SIGNED_SIGMOID,
        NEAT.ActivationFunction.SIGNED_SIGMOID,
        0,
        params)

    pop = NEAT.Population(genome, params, True, 1.0)

    net = Network(substrate, 10)

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

    #### XOR
    xorpatterns = [([1., 0., 1.], 1.), ([0., 1., 1.], 1.), ([1., 1., 1.], 0.), ([0., 0., 1.], 0.)] 
    ###########################

    for generation in range(GENERATIONS):
        genome_list = NEAT.GetGenomeList(pop)

        if genome_list > net.nfeatures:
            genome_list = genome_list[:net.nfeatures]

        #rand_input = rng.rand(5)
        #target = target_func.get_output(rand_input) # TODO: Add noise?
        rand_input, target = xorpatterns[rng.randint(0, len(xorpatterns))]
         
        mse = net.train(genome_list, rand_input, target, generation)
        for genome in genome_list:
            fitness = net.genome_archive[genome.GetID()][1]
            genome.SetFitness(fitness)
            genome.SetEvaluated()

        print "MSE:", mse

        error = 0
        print "=============================="
        for pattern in xorpatterns:
            output = net.activate(genome_list, pattern[0])
            
            error += abs(output-pattern[1]-pattern[1])

            print "TARGET:", pattern[1]
            print "OUTPUT:", output
        print "=============================="

        print "XOR:",(4-error)

        deleted_genome = NEAT.Genome()
        new_genome = pop.Tick(deleted_genome)

        del net.genome_archive[deleted_genome.GetID()]

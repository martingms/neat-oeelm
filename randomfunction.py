import oeelm
import numpy as np
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 100

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.0 # TODO
params.MaxActivationA = 10.0

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

substrate = NEAT.Substrate([(0,-2), (0,-1), (0,0), (0,1), (0,2)],
                           [],
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

#### Creating random function
import originaloeelm # J. Auerbach
import time
from numpy.random import RandomState 

rng = RandomState(int(time.time()))
target_features = [originaloeelm.BinaryFeature.create_random(rng, 5, 0.6)
                   for _ in range(5)]

target_weights = np.concatenate((rng.normal(0,1,5), [0])) 

target_func = originaloeelm.Learner(target_features, target_weights)
###########################

no = oeelm.NEATOeelm(params, genome, substrate, 1)

for generation in range(1000000):
    rand_input = rng.rand(5)
    target = target_func.get_output(rand_input)
    err, output = no.train(rand_input, target)

    print "========================================="
    print "GENERATION:", generation
    print "INPUT:", rand_input
    print "TARGET:", target
    print "OUTPUT:", output
    print "SQUARED_ERROR:", err

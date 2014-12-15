import oeelm
import numpy as np
import MultiNEAT as NEAT
import random
import time

random.seed(int(time.time()))

params = NEAT.Parameters()
params.PopulationSize = 100
params.YoungAgeFitnessBoost = 1.0
"""
params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.0 # TODO
params.MaxActivationA = 10.0
"""

# Seemingly crucial for performance:
params.MutateNeuronActivationTypeProb = 0.05

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

#substrate = NEAT.Substrate([(0,-1), (0,0), (0,1)],
#                           [],
#                           [(2,0)])
substrate = [[0,-1], [0,0], [0,1]]

genome = NEAT.Genome(0,
                     4,#substrate.GetMinCPPNInputs(),
                     0,
                     1,#substrate.GetMinCPPNOutputs(),
                     False,
                     NEAT.ActivationFunction.SIGNED_SIGMOID,
                     NEAT.ActivationFunction.SIGNED_SIGMOID,
                     0,
                     params)


xorpatterns = [([1., 0., 1.], [1.]), ([0., 1., 1.], [1.]),
               ([1., 1., 1.], [0.]), ([0., 0., 1.], [0.])]

no = oeelm.NEATOeelm(params, genome, substrate, 3, 1)

mva = 0.5
for generation in range(20000):
    rand_input, target = xorpatterns[random.randint(0,3)]
    err, output = no.train(rand_input, target)
    
    mva = mva * 0.9 + err[0] * 0.1
    print generation, mva
    #print rand_input

print "FW:", no.net.feature_weights
print "FV:", no.net.feature_values
print "OW:", no.net.output_weights
print "WMEMA:", no.net.weight_magnitude_emas

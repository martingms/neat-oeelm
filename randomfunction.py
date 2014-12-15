import oeelm
import numpy as np
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 100
params.YoungAgeFitnessBoost = 1.0

params.CrossoverRate = 0.05
params.CompatTreshChangeInterval_Evaluations = 100
params.SurvivalRate = 0.10
"""
params.MutateWeightsSevereProb = 0.2
params.MinSpecies = 1
params.MaxSpecies = 1
params.RouletteWheelSelection = True
#params.WeightMutationRate = 0.5

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.0 # TODO
params.MaxActivationA = 1.0

params.MutateNeuronActivationTypeProb = 0.03;
"""

# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
params.ActivationFunction_Tanh_Prob = 0.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 0.0;
params.ActivationFunction_UnsignedStep_Prob = 1.0;
params.ActivationFunction_SignedGauss_Prob = 0.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 0.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 0.0;


#substrate = NEAT.Substrate([(0,-10), (0,-9), (0,-8), (0,-7), (0,-6),
#                            (0,-5), (0,-4), (0,-3), (0,-2), (0,-1),
#                            (0,0), (0,1), (0,2), (0,3), (0,4),
#                            (0,5), (0,6), (0,7), (0,8), (0,9)],
substrate = NEAT.Substrate([(0,-2), (0,-1), (0,0), (0,1), (0,2)],
                           [],
                           [(2,0)])

genome = NEAT.Genome(0,
                     substrate.GetMinCPPNInputs(),
                     0,
                     substrate.GetMinCPPNOutputs(),
                     False,
                     NEAT.ActivationFunction.SIGNED_STEP,
                     NEAT.ActivationFunction.SIGNED_STEP,
                     0,
                     params)

#### Creating random function
import originaloeelm # J. Auerbach
import time
from numpy.random import RandomState 

rng = RandomState(int(time.time()))
target_features = [originaloeelm.BinaryFeature.create_random(rng, 5, 0.6)
                   for _ in range(5)]
#target_features = [originaloeelm.BinaryFeature.create_random(rng, 20, 0.6)
#                   for _ in range(20)]

target_weights = np.concatenate((rng.normal(0,1,5), [0])) 
#target_weights = np.concatenate((rng.normal(0,1,20), [0])) 

target_func = originaloeelm.Learner(target_features, target_weights)
###########################

no = oeelm.NEATOeelm(params, genome, substrate, 1)

#mva = 0
mse_hist = np.zeros(10000)
for generation in range(1000000):
    #rand_input = rng.choice([0,1], 20)
    rand_input = rng.choice([0,1], 5)
    target = target_func.get_output(rand_input) + rng.normal(0,1)
    #target = target_func.activate(rand_input)
    """
    if generation > 0:
        print "Prepre:", no.activate(rand_input)[0]
    """
    err, output = no.train(rand_input, target)

    mse_hist[generation%10000] = err
    if (generation+1) % 100 == 0:
        if generation < 10000:
            print np.mean(mse_hist[0:generation])
        else:
            print np.mean(mse_hist)
    #if generation == 0:
    #    mva = err[0]
    """
    print "========================================="
    print "GENERATION:", generation
    print "INPUT:", rand_input
    print "TARGET:", target
    print "OUTPUT:", output
    print "SQUARED_ERROR:", err
    """
    #mva = mva*0.99 + err[0] *0.01
    #print mva

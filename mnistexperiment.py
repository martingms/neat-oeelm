import oeelm
import numpy as np
import MultiNEAT as NEAT
import random
import time
from mnist import mnist_substrate, load_mnist


random.seed(int(time.time()))

params = NEAT.Parameters()
params.PopulationSize = 100
params.YoungAgeFitnessBoost = 1.0

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

substrate = NEAT.Substrate(mnist_substrate,
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

no = oeelm.NEATOeelm(params, genome, substrate, 10)

images, labels = load_mnist("training", path="mnistdata")
images /= 255.0

labeltranslations = np.identity(10)

splitlabels = np.zeros((len(labels), 10))
for i in range(len(labels)):
    splitlabels[i] = labeltranslations[labels[i]]
    
for generation in range(len(images)):
    err, output = no.train(images[generation], splitlabels[generation])

    print "========================================="
    print "INPUT:", "... some image"
    print "TARGET:", splitlabels[generation], labels[generation]
    print "OUTPUT:", output
    print "SQUARED_ERROR:", err
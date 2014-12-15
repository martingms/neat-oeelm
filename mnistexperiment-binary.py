import oeelm
import numpy as np
import MultiNEAT as NEAT
import random
import time
from mnist import mnist_substrate, load_mnist

random.seed(int(time.time()))

params = NEAT.Parameters()
params.PopulationSize = 1000
params.YoungAgeFitnessBoost = 1.0
params.CrossoverRate = 0.20

# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 0.0;
params.ActivationFunction_UnsignedStep_Prob = 1.0;
params.ActivationFunction_SignedGauss_Prob = 0.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 0.0;
params.ActivationFunction_UnsignedSine_Prob = 1.0;
params.ActivationFunction_Linear_Prob = 1.0;

substrate = NEAT.Substrate(mnist_substrate,
                           [],
                           [(2,0)])

genome = NEAT.Genome(0,
                     substrate.GetMinCPPNInputs(),
                     0,
                     substrate.GetMinCPPNOutputs(),
                     False,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                     0,
                     params)

no = oeelm.NEATOeelm(params, genome, substrate, 1)

images, labels = load_mnist("training", digits=[1,8], path="mnistdata")
images /= 255.0

for generation in xrange(len(images)):
    target = 1
    if labels[generation] == 8:
        target = 0

    err, output = no.train(images[generation], [target])
    print len(images[generation])

    print "Generation:", generation, "/", len(images), "Output:", output, "Target:", target, "Err:", err

images, labels = load_mnist("testing", digits=[1,8], path="mnistdata")
images /= 255.0

correct, wrong = 0, 0
for test in xrange(len(images)):
    output = no.activate(images[test])

    if output[0] >= 0.5 and labels[test] == 1:
        correct += 1
    elif output[0] < 0.5 and labels[test] == 8:
        correct += 1
    else:
        wrong += 1

    print "Test:", test, "/", len(images)

print "Correct:", correct, "Wrong:", wrong, "Total:", len(images)

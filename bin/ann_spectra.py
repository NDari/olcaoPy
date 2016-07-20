import olcaoPy.fileOps as fo
import os
import shutil
import sys
import subprocess
import multiprocessing
import math
import copy
import random

class Candidate(object):
    def __init__(self, candidName):
        """
        candidate object
        """
        self.weights = []
        self.fitness = 0.0
        self.velocity = []
        self.pBestWeights = []
        self.pBestFitness = 1000000
        self.name = candidName

    def initWeights(self, numItems):
        # seed the random number generator with current system clock
        random.seed()
        for i in range(numItems):
            d = []
            d.append(20.0 * random.random() - 10.0)
            d.append(20.0 * random.random() - 10.0)
            self.weights.append(d)
        return self

    def initVelocities(self, numItems):
        for i in range(numItems):
            d = []
            d.append(0.0)
            d.append(0.0)
            self.velocity.append(d)
        return self
    
    def initPersonalBests(self):
        self.pBestFitness = self.fitness
        self.pBestWeights = copy.deepcopy(self.weights)
        return self
    
    def updatePersonalBests(self):
        if self.fitness < self.pBestFitness:
            self.pBestFitness = self.fitness
            for i in range(len(self.weights)):
                self.pBestWeights[i][0] = self.weights[i][0]
                self.pBestWeights[i][1] = self.weights[i][1]
        return self

    def moveTo(self, other):
        for i in range(len(self.velocity)):
            self.velocity[i][0] = 0.729 * (self.velocity[i][0] +
                    (random.random() * 2.05 * (self.pBestWeights[i][0] - self.weights[i][0])) +
                    (random.random() * 2.05 * (other.pBestWeights[i][0] - self.weights[i][0])))

            #if self.velocity[i][0] > 0.2:
            #    self.velocity[i][0] = 0.2
            #if self.velocity[i][0] < -0.2:
            #    self.velocity[i][0] = -0.2

            self.weights[i][0] += self.velocity[i][0]
            
            self.velocity[i][1] = 0.729 * (self.velocity[i][1] +
                    (random.random() * 2.05 * (self.pBestWeights[i][1] - self.weights[i][1])) +
                    (random.random() * 2.05 * (other.pBestWeights[i][1] - self.weights[i][1])))
            
            #if self.velocity[i][0] > 0.2:
            #    self.velocity[i][0] = 0.2
            #if self.velocity[i][0] < -0.2:
            #    self.velocity[i][0] = -0.2

            self.weights[i][1] += self.velocity[i][1]

        return self


def neighborList(numCandids, cols):
    '''
    Von Neumann topology
    '''
    neighborList = []
    for i in range(numCandids):
        neighbor = []
        # up neighbor
        if i < cols:
            n = numCandids - cols + i
        else:
            n = i - cols
        neighbor.append(n)

        # left
        if (i%cols == 0):
            n = i + 4
        else:
            n = i - 1
        neighbor.append(n)

        # right neighbor
        if ((i+1)% cols == 0):
            n = i - 4
        else:
            n = i + 1
        neighbor.append(n)

        # down neighbor
        if ((i+cols) >= numCandids):
            n = i % cols
        else:
            n = i + cols
        neighbor.append(n)
        
        neighborList.append(neighbor)

    return neighborList

def evalFitness(name, weights, inputs):
    if os.path.exists(name):
        sys.exit("Directory " + name + " exists. Aborting.")
    os.makedirs(name)
    os.chdir(name)
    res = []
    for i in range(len(inputs)):
        ans = []
        ans.append(0.0)
        for j in range(len(weights)):
            ans[0] += (weights[j][1] * math.tanh(weights[j][0] * inputs[i]))
        res.append(ans)
    fo.writeFile("output", res)
    os.chdir("..")

def getRes(name, correctOutput):
    os.chdir(name)
    s = fo.readFloats("output")
    fitness = 0.0
    for i in range(len(s)):
        fitness += ((128 - i) * abs((correctOutput[i]) - (s[i][0])))
    os.chdir("..")
    shutil.rmtree(name)
    return fitness

numWeightPairs = 16
numCandids = 50 # dont change this anymore. or change neighborlist to match
numProcs = 4

cols = 5


names = [str(i) for i in range(numCandids)]
swarm = []

###neighborList = neighborList(numCandids, cols)

inp = fo.readFile("expcf252")
inputs = []

for i in range(len(inp)):
    inputs.append(float(inp[i][0]))

cor = fo.readFile("cf")
correctOutput = []
for i in range(len(cor[0])):
    correctOutput.append(float(cor[0][i]))

for i in range(len(names)):
    swarm.append(Candidate(names[i]))

# initialize swarm members
for i in swarm:
    i.initWeights(numWeightPairs).initVelocities(numWeightPairs)

print "weights and velocities of the swarm initialized\n"

if (len(sys.argv) > 1):
    if sys.argv[1] == "-c":
        swarm[numCandids - 1].weights = fo.readFloats("contWeights")
    else:
        sys.exit("Unknown command line parameter")

# parallel
#pool = multiprocessing.Pool(numProcs)
#for i in swarm:
#    pool.apply_async(evalFitness, (i.name, i.weights, inputs))
#pool.close()
#pool.join()

## serial
for i in swarm:
    evalFitness(i.name, i.weights, inputs)
    print "finished",i.name

for i in swarm:
    i.fitness = getRes(i.name, correctOutput)
    print i.name, i.fitness
    i.initPersonalBests()

iteration = 0;

print "starting execution of the main loop\n"

### main execution location
while (iteration < 10000):
    gb = 0
    for i in range(1, len(swarm)):
        if swarm[i].pBestFitness < swarm[gb].pBestFitness:
            gb = i
    

    if swarm[gb].pBestFitness == 0.0:
        print "optimum band gap achieved. writing final weights..."
        fo.writeFile("final_Weights", swarm[gb].pBestWeights)
        break


    #for i in range(numCandids):
    #    target = swarm[i]
    #    for j in range(len(neighborList[i])):
    #        if target.fitness > swarm[neighborList[i][j]].fitness:
    #            target = swarm[neighborList[i][j]]
    #    swarm[i].moveTo(target)

    for i in swarm:
        i.moveTo(swarm[gb])
    
    # parallel
    #pool = multiprocessing.Pool(numProcs)
    #for i in swarm:
    #    pool.apply_async(evalFitness, (i.name, i.weights, inputs))
    #pool.close()
    #pool.join()
    
    ## serial
    for i in swarm:
        evalFitness(i.name, i.weights, inputs)
    
    for i in swarm:
        i.fitness = getRes(i.name, correctOutput)
        i.updatePersonalBests()

    if (iteration % 200 == 0):
        print "global best in iteration", iteration, "is candidate", gb
        print "The bandgap is now", swarm[gb].pBestFitness, "ev away from perfect"
        fo.writeFile("contWeights", swarm[gb].weights)

    iteration +=1




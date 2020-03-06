# /usr/bin/env python
# -*- coding: UTF-8 -*-
#
# sugar.py
#
# Cognitive model object for 停车对标.
# Michael Engelhart, 2014
# Nadia Said, 2014
#

# from algopy import UTPM, exp, tanh, log
from math import exp, tanh, log
from math import pi, ceil, floor
from numpy.random import seed, random, logistic
import sys
from tools import mytanh

# --------------------------------------------------------------------
# Set Number of Rounds
# --------------------------------------------------------------------
roundsNum = 40
newsuggestionSpeed = [40, 39, 38, 37, 38, 38, 34, 30, 25, 24, 24, 24, 24, 23, 22, 22, 21, 20, 19, 18,
                              17, 17, 17, 17, 16, 16, 16, 15, 14, 14, 13, 12, 4, 2, 1, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]

# --------------------------------------------------------------------
# Class Chunk
# --------------------------------------------------------------------
class Chunk:
    def __init__(self, slot0, slot1, slot2):
        self.slots = [0 for i in range(3)]
        self.slots[0] = slot0  # 当前实际挡位
        self.slots[1] = slot1  # 当前实际速度
        self.slots[2] = slot2  # 根据1和2计算得到的新速度
        print 'slot:',self.slots[0],self.slots[1],self.slots[2]

    def __getitem__(self, key):
        return self.slots[key]


# --------------------------------------------------------------------
# Class CognitiveModel
# --------------------------------------------------------------------
class CognitiveModel:
    def __init__(self, h, a, s, initialProduction, inputPath="inputs/inputSugar_100_start_values/inputSugar_0/",
                 randomSeed=-1, inputNum=-1, algorithmicDiff=False, optimize=-1):
        self.h = h
        self.a = a
        self.s = s
        self.T = 0.05
        self.name = "停车对标"
        self.filename = "ParkingBenchmarking"
        self.inputPath = inputPath
        self.results = None
        self.inputs = None#随机达到目标因素，未达到目标因素，随机因素
        self.chunks = None
        self.randomWorkers = None
        self.randomSeed = randomSeed
        self.inputNum = inputNum
        self.initialProduction = initialProduction  # 初始产量

        self.algorithmicDiff = algorithmicDiff

        self.optimize = optimize

        self.chunkActivated = None
        self.chunkActivationPerRound = None  # 每轮激活值
        self.onTargetChunkActivated = None  # 目标内激活值

        self.actr = None

        self.createChunks()
        self.readInputs()

        if (randomSeed >= 0):
            seed(randomSeed)

    def setACTR(self, actr):
        self.actr = actr

    def getName(self):
        return self.name

    def getFilename(self):
        return self.filename

    def setRoundsNum(self, n):
        self.roundsNum = n
        self.production = [-1.0 for j in range(self.roundsNum + 1)]
        self.production[0] = self.initialProduction  # 初始产量
        self.workers = [-1.0 for j in range(self.roundsNum)]#当前挡位
        self.results = [-1.0 for j in range(self.roundsNum)]
        self.chunkActivationPerRound = [-1.0 for j in range(self.roundsNum)]
        self.onTargetChunkActivated = [-1.0 for j in range(self.roundsNum)]
        self.randomWorkers = [-1.0 for j in range(self.roundsNum + 1)]
        self.randomWorkers[0] = 3.0 + self.inputs[0][0]
        if (self.randomSeed >= 0):
            self.noise = [logistic(loc=0.0, scale=self.s) for i in range(self.roundsNum)]
        else:
            self.noise = [0.0 for i in range(self.roundsNum)]

    def Heaviside(self, x, h=-1):
        if h < 0:
            h = self.h
        d = 1e-3
        return 0.5 * mytanh(h * (x + d)) + 0.5

    def delta(self, x):
        return exp(-self.a * (x ** 2.0))  # !!!!!

    def getLifetime(self, j, i):
        L = (j - self.actr.Time(j, i)) + self.T
        return L

    def getNoise(self, j):
        return self.noise[j]

    def getSimilarity(self, j, i, showInput, showMisc, output, h=-1):
        #  Mi1 := − | pj − ci2 | / max(pj, ci2);两值之间的相似性，最大为0，最小为 - 1，实际速度与记忆块中速度的相似性
        #      Mi2 := − | 9 − ci3 | / max(9, ci3);
        s1 = -self.Heaviside(self.production[j] - self.chunks[i][1], h) * (self.production[j] - self.chunks[i][1]) / \
             self.production[j] - (1.0 - self.Heaviside(self.production[j] - self.chunks[i][1], h)) * (
                         self.chunks[i][1] - self.production[j]) / self.chunks[i][1]
        # 假设只有四十轮变换的机会，也即经过四十次挡位变换，速度从40降到0，每个实际速度所对应的挡位对应下一时刻的速度，要使的下一时刻列车实际速度接近推荐速度，
        # 驾驶员通过看到当前速度选取激活值最高的知识块，该知识块中包含该速度所对应的挡位

        # print len(newsuggestionSpeed)
        s2 = -self.Heaviside(newsuggestionSpeed[j]- self.chunks[i][2], h) * (newsuggestionSpeed[j]- self.chunks[i][2]) / newsuggestionSpeed[j]\
             - (1.0 - self.Heaviside(newsuggestionSpeed[j]- self.chunks[i][2], h)) * (self.chunks[i][2] - newsuggestionSpeed[j]) / self.chunks[i][2]
        # showInput即为产量
        # print 's1,s2,chunk[i][1],self.production[j],chunk[i][2],chunk[i][2]:'
        # print s1,s2,self.chunks[i][1],self.production[j],self.chunks[i][2],self.chunks[i][2]
        showInput.append(self.production[j])

        return (s1 + s2)

    # 获取最大激活值对应的动作
    def getAction(self, j, maxActivation, activation, showInput, showMisc, h=-1):
        # Calculate activation with Heaviside   用Heaviside计算激活值
        # activation[:] = [self.Heaviside(x - maxActivation, h) for x in activation]
        # Calculate activation with Delta Function 利用delta函数计算激活
        # activation[:] = [self.delta(x - maxActivation, 10000000) for x in activation]
        # activation = [activation[i]*self.chunks[i][0] for i in range(len(activation))]
        # action = sum(activation)*self.Heaviside (maxActivation - self.actr.tau, h) + (1.0-self.Heaviside (maxActivation-self.actr.tau, h)) * self.randomWorkers[j]
        # self.chunkActivated = self.Heaviside (maxActivation - self.actr.tau, h)

        # with open ("activation %d_argmax.csv" %(j,), "w") as f:
        #     for a in activation:
        #         f.write("%e\n" %(a,))

        activation[:] = [self.Heaviside(x - maxActivation, h) for x in
                         activation]  # !!![self.Heaviside(x - maxActivation, 1e7) ...]

        activation = [activation[i] * self.chunks[i][0] for i in range(len(activation))]
        # 动作即为最大激活值大于tau所对应的动作
        action = sum(activation) * self.Heaviside(maxActivation - self.actr.tau, h) + (
                    1.0 - self.Heaviside(maxActivation - self.actr.tau, h)) * self.randomWorkers[j]
        # 被激活的块
        self.chunkActivated = self.Heaviside(maxActivation - self.actr.tau, h)

        return action

    def getActiveIndex(self, j, action, showResult, showMisc, h=-1):
        randomOffTarget = self.inputs[0][j + 1]
        randomOnTarget = self.inputs[1][j + 1]
        randomFactor = self.inputs[2][j]

        self.workers[j] = action
        # 下一时刻对应得速度，应该满足列车动力学模型
        # v[j+1]=v[j]-a*0.2 + randomFactor
        # newProduction = -self.workers[j]*2+self.production[j] + randomFactor
        newProduction = -self.workers[j] * 2 + self.production[j]
        # 当前速度应该小于40大于0
        self.production[j + 1] = 40.0 * self.Heaviside(newProduction - 40.0, h) + \
                                 (1.0 - self.Heaviside(newProduction - 40.0, h)) * \
                                 (self.Heaviside(0.0 - newProduction, h) + \
                                  (1.0 - self.Heaviside(0.0 - newProduction, h)) * newProduction)

        # 这儿还存在问题
        # 下一轮新得随机挡位选取在，如果下一时刻速度为9即加上randomOnTarget,如果下一轮速度不为9，加上randomOffTarget
        newRandomWorkers = (self.workers[j] + ((self.delta(self.production[j + 1] - newsuggestionSpeed[j]) * randomOnTarget) + \
                                               (1.0 - self.delta(self.production[j + 1] - newsuggestionSpeed[j])) * randomOffTarget))
        # 下一轮新的随机挡位在4和1之间，
        self.randomWorkers[j + 1] = 4.0 * self.Heaviside(newRandomWorkers - 4.0, h) + \
                                    (1.0 - self.Heaviside(newRandomWorkers - 4.0, h)) * \
                                    (self.Heaviside(0.0 - newRandomWorkers, h) + \
                                     (1.0 - self.Heaviside(0.0 - newRandomWorkers, h)) * newRandomWorkers)

        index = self.workers[j] + 4 * self.production[j] + 160 * self.production[j + 1] - (1.0 + 4.0 + 160.0)  # Attention: values in [1,12]!

        # result = (1.0- self.Heaviside (self.production[j+1]-10.4, self.h))* \
        #                          self.Heaviside (self.production[j+1] -7.6, self.h)
        # 如果制糖达标，8≤pj≤10，则𝑢_𝑗^𝑜𝑛  ∈{−1，…，1}添加到当前劳动力中
        # 如果达标，也即下一时刻速度为8，9，10。即为1，否咋即为0
        result = self.delta(self.production[j + 1] - 9.0) + self.delta(self.production[j + 1] - 8.0) + self.delta(
            self.production[j + 1] - 10.0)
        print'result:',result
        self.results[j] = result

        # if (self.algorithmicDiff == False):

        #    self.results[j] = round (result)

        #    self.chunkActivationPerRound[j] = round(self.chunkActivated)
        #    if round (result) == 1 and round(self.chunkActivated) == 1 :
        #           self.onTargetChunkActivated[j] = 1
        #    else:
        #           self.onTargetChunkActivated[j] = 0

        showMisc.append(self.randomWorkers[j + 1])  # 随机工人数量
        showResult.append(self.production[j + 1])  # 产量
        return index

    def getResult(self):
        totalWon = sum(self.results)
        NumberOfChunksActivated = sum(self.chunkActivationPerRound)
        NumChunksActOnTarget = sum(self.onTargetChunkActivated)
        perRound = self.results
        strategy = [-1 for j in range(self.roundsNum)]
        optimalStrategyChunk = [-1 for j in range(self.roundsNum)]
        optStrat = [-1 for j in range(int(ceil(self.roundsNum / 20)))]
        optStratChunk = [-1 for j in range(int(ceil(self.roundsNum / 10)))]
        wonRounds = [-1 for j in range(int(ceil(self.roundsNum / 20)))]
        numChunksAct = [-1 for j in range(int(ceil(self.roundsNum / 10)))]
        numChunksActOnT = [-1 for j in range(int(ceil(self.roundsNum / 10)))]
        target = 9  # 目标产量值

        for i in range(len(self.workers)):
            if (round(self.workers[i]) == floor((target + round(self.production[i])) / 2) or round(
                    self.workers[i]) == ceil((target + round(self.production[i])) / 2)):
                strategy[i] = 1
                if self.chunkActivationPerRound[i] == 1:
                    optimalStrategyChunk[i] = 1
                else:
                    optimalStrategyChunk[i] = 0
                    # print ("Optimal strategy! {0} vs. {1} ({2})".format(round(self.workers[i]),floor ((target + round(self.production[i]))/2), round(self.production[i])))
            else:
                strategy[i] = 0
                optimalStrategyChunk[i] = 0
                # print ("No optimal strategy! {0} vs. {1} ({2})".format(round(self.workers[i]),floor ((target + round(self.production[i]))/2), round(self.production[i])))

        for i in range(int(ceil(self.roundsNum / 20))):
            optStrat[i] = sum(strategy[i * 20:(i + 1) * 20])
            wonRounds[i] = sum(perRound[i * 20:(i + 1) * 20])

        for i in range(int(ceil(self.roundsNum / 10))):
            optStratChunk[i] = sum(optimalStrategyChunk[i * 10:(i + 1) * 10])
            numChunksAct[i] = sum(self.chunkActivationPerRound[i * 10:(i + 1) * 10])
            numChunksActOnT[i] = sum(self.onTargetChunkActivated[i * 10:(i + 1) * 10])
        if self.optimize == 1:
            return totalWon
        else:
            return str(totalWon) + ',' + str(wonRounds).strip('[]') + ',' + str(sum(optStrat)).strip('[]') + ',' + str(
                optStrat).strip('[]') + ',' + str(NumberOfChunksActivated) + ',' + str(numChunksAct).strip(
                '[]') + ',' + str(NumChunksActOnTarget) + ',' + str(numChunksActOnT).strip('[]') + ',' + str(
                sum(optStratChunk)).strip('[]') + ',' + str(optStratChunk).strip('[]') + ',' + str(
                self.randomSeed) + ',' + str(self.inputNum) + ',' + str(self.s) + ',' + str(self.initialProduction)

    def getChunksNum(self):
        return len(self.chunks)

    # Generate all chunks in correct order
    def createChunks(self):
        self.chunks = []
        for i in range(1, 41):
            for j in range(1, 41):
                for k in range(1, 5):
                    c = Chunk(k, j, i)
                    print '上一行对应个数：',k+j*4+i*160-(4+160)
                    # print '上一行对应激活值：', k + j * 4 + i * 160 - (4 + 160)
                    self.chunks.append(c)


    # Read inputs from external files
    def readInputs(self):
        self.inputs = []
        print ("Reading inputs from: {i}".format(i=self.inputPath))

        try:
            with open(self.inputPath + "randomFactor.py") as f:
                inputString = f.readlines()
                randomFactor = [int(x) for x in inputString[0].strip("[]\n").split(",")]

            with open(self.inputPath + "randomOnTarget.py") as f:
                inputString = f.readlines()
                randomOnTarget = [int(x) for x in inputString[0].strip("[]\n").split(",")]

            with open(self.inputPath + "randomOffTarget.py") as f:
                inputString = f.readlines()
                randomOffTarget = [int(x) for x in inputString[0].strip("[]\n").split(",")]

            self.inputs = [randomOffTarget, randomOnTarget, randomFactor]
            return True
        except ValueError as detail:
            sys.exit("CognitiveModel readInputs(): Error reading inputs! {}".format(detail))
            return False
        except IOError as detail:
            sys.exit("CognitiveModel readInputs(): Error opening input files! {}".format(detail))
            return False

#
# End of file sugar.py
#

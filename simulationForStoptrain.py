#/usr/bin/env python
# -*- coding: UTF-8 -*-

#
# simulation.py
#
# Simulation of an ACT-R cognitive model
# Michael Engelhart, 2014
# Nadia Said, 2015
#

import time
startTime = time.clock()

import argparse
import csv

import algopy
from algopy import UTPM

#--------------------------------------------------------------------
# Parse arguments
#--------------------------------------------------------------------
parser = argparse.ArgumentParser (description='Python model for simulating ACT-R cognitive models.')
parser.add_argument ('-tau',        type=float, help="Activation threshold, default value is -2.5")
parser.add_argument ('-d',          type=float, help="Decay parameter of base level learning, default value is 0.5, between 0 and 1")
parser.add_argument ('-P',          type=float, help="Weighting of similarities, default value is 10.0")
parser.add_argument ('-s',          type=float, help="Transient noise value, default value is 0.2 (Chung & Byrne, 2008)")
parser.add_argument ('-InProd',     type=int,   help="Initial value of sugar production (integer)")#初始糖产量
parser.add_argument ('-L',          type=int,   help="Print headline each L lines (integer)")
parser.add_argument ('--file',      type=str,   help="Filename for csv output")
parser.add_argument ('--input',     type=str,   help="Path for input files", dest="inputPath")
parser.add_argument ('--seed',      type=int,   help="Seed for random numbers")
parser.add_argument ('--inputNum',  type=int,   help="Number of Input used")
parser.add_argument ('--no-output', dest='output',   action='store_false', help="Do not print simulation output")
parser.add_argument ('--no-csv',    dest='csv',      action='store_false', help="Do not write output to csv-file")
parser.set_defaults (tau = -2.5)#激活限值
parser.set_defaults (d   =  0.5)#延迟率
parser.set_defaults (P   = 10.0)#相似值得权重
parser.set_defaults (s   = 0.2)
parser.set_defaults (L   = 10)
parser.set_defaults (InProd = 40)#初始速度设置为40
parser.set_defaults (seed = -1)
parser.set_defaults (inputNum = -1)
parser.set_defaults (inputPath = "inputs/inputSugar_100_start_values/inputSugar_0/")
parser.set_defaults (output = True)
parser.set_defaults (csv    = True)
parser.set_defaults (AlgDiff    = False, help="Algorithmic Differentiation tool: AlgoPy")
args = parser.parse_args ()

#--------------------------------------------------------------------
# Load models
#--------------------------------------------------------------------
#from blackjack import CognitiveModel, Chunk, roundsNum
from stopTrain import CognitiveModel, Chunk, roundsNum
from actr import CognitiveArchitecture

#--------------------------------------------------------------------
# Set parameters
#--------------------------------------------------------------------
tau = args.tau
d   = args.d
P   = args.P
s   = args.s

#-------------------------, first -------------------------------------------
# Set constants设置常量
#--------------------------------------------------------------------
h     =  1e9 #100
a     =  1e2
Amin  = -1e3

#--------------------------------------------------------------------
# Set options
#--------------------------------------------------------------------
printResults      = args.output
inputPath         = args.inputPath
printHeadline     = args.L
writeCsv          = args.csv
randomSeed        = args.seed
inputNum          = args.inputNum
outputFilename    = args.file
initialProduction = args.InProd
algorithmicDiff   = args.AlgDiff

#--------------------------------------------------------------------
# Run simulation
#--------------------------------------------------------------------

print ("\nCoMoKoS cognitive model simulation")
print (  "==================================\n")

model = CognitiveModel (h, a, s,  initialProduction, inputPath, randomSeed, inputNum, algorithmicDiff)
actr  = CognitiveArchitecture (d, h, a, Amin, printResults, printHeadline, writeCsv, outputFilename, algorithmicDiff)

model.setRoundsNum (roundsNum)
model.setACTR (actr)
actr.setModel (model)
actr.setRoundsNum (roundsNum)

x      = [tau,P]

result = actr.simulate(x)
print "result:",result
#result, result_Delta, result_Diff = actr.simulate(x)

#print result, result_Delta, result_Diff

endTime = time.clock()
print ("Runtime was {t: .2f}s (start={s:.2f}, end={e:.2f})\n\n".format( \
       t=endTime-startTime, e=endTime, s=startTime))

exit()
#--------------------------------------------------------------------
# Algorithmic Differentiation算法微分
#--------------------------------------------------------------------

startTime = time.clock()

algorithmicDiff   = True
writeCsv          = False
print algorithmicDiff
model = CognitiveModel (h, a, s,  initialProduction, args.inputPath, randomSeed, inputNum, algorithmicDiff)
actr  = CognitiveArchitecture (d, h, a, Amin, printResults, printHeadline, writeCsv, outputFilename, algorithmicDiff)

model.setRoundsNum (roundsNum)
model.setACTR (actr)
actr.setModel (model)
actr.setRoundsNum (roundsNum)

x = UTPM.init_jacobian([tau,P])
y = actr.simulate(x)
print y
print y.data[0][0]
algopy_jacobian = UTPM.extract_jacobian(y)
print('jacobian = ',algopy_jacobian)

exit()

# reverse mode using a computational graph使用计算图形的反向模式
# ----------------------------------------

# STEP 1: trace the function evaluation
cg = algopy.CGraph()
x = algopy.Function([tau,P])
y = actr.simulate(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]

# STEP 2: use the computational graph to evaluate derivatives
#print('gradient =', cg.gradient([tau,P]))
#print('Jacobian =', cg.jacobian([tau,P]))
#print('Hessian =', cg.hessian([tau,P]))
#print('Hessian vector product =', cg.hess_vec([tau,P],[tau,P])


#--------------------------------------------------------------------
# Write Results To File (Algorithmic Differentiation)
#--------------------------------------------------------------------

#if (algorithmicDiff):
#        with open('AlgoPy_Results.csv', 'a') as csvfile:
#               writeResult = csv.writer (csvfile, delimiter=',', quotechar='|', \
#                       quoting=csv.QUOTE_MINIMAL)
#               writeResult.writerow([tau,P, algopy_jacobian[0], algopy_jacobian[1]])
#csvfile.close()


endTime = time.clock()
print ("Runtime was {t: .2f}s (start={s:.2f}, end={e:.2f})\n\n".format( \
       t=endTime-startTime, e=endTime, s=startTime))

#
# End of file simulation.py
#

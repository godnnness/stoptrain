#/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# actr.py
#
# Model of ACT-R for cognitive models.
# Michael Engelhart, 2014
# Nadia Said, 2015
#

#from algopy import UTPM, exp, tanh, log
from math import exp, tanh, log
from sys import float_info
import csv
import os.path
from math import pi
from tools import mytanh, softargmax

class CognitiveArchitecture:
	def __init__(self, d, h, a, Amin, output=True, headlineRepeat=10, writeCsv=True, filename="", algorithmicDiff =True):
		self.d    = d
		self.h    = h
		self.a    = a
		self.Amin = Amin
		
		self.filename        = filename
		self.headlineRepeat  = headlineRepeat
		self.output          = output
		self.writeCsv        = writeCsv
		self.debug           = True
		self.algorithmicDiff = algorithmicDiff
		
		self.model     = None
		self.roundsNum = 0
		self.chunksNum = 0


	def delta (self, x):
		return exp (-self.a*(x**2.0))

	def Heaviside (self, x, h=-1):
		if h < 0:
			h = self.h
		d = 1e-6
		return 0.5*mytanh(h*(x+d)) + 0.5

	def Max (self, a, b):
		H = self.Heaviside(a-b)
		return a*H + (1.0-H)*b

	def setModel (self, model, printName=True):
		self.model     = model
		self.chunksNum = self.model.getChunksNum()
		if printName:
			print ("Model: {0}".format(self.model.getName()))
	# 设置轮数
	def setRoundsNum (self, j):
		self.roundsNum = j

	def Time (self, j, i):
		self.t[i][j+1] = self.t[i][j] + self.delta(i-self.iAct[j])*(1-self.e[i][j])*(j-self.t[i][j])
		t = self.t[i][j+1]
		return t
	        
	# 计算每一轮各个块的激活值并选取激活值最大的块,以及该块所对应的第几个块，以及所对应的工人数
	def simulate (self, x):# Set Parameters
		self.tau  = x[0]
		self.P    = x[1]

		# Rounds and chunks set?
		if (self.roundsNum > 0 and self.chunksNum > 0):
			self.printOptions()
			print ("Simulating {j} rounds with {i} chunks...\n".format (i=self.chunksNum, j=self.roundsNum))
			# Initialize all variables
			self.t = [[0.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]
			self.n = [[1.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]
			self.e = [[0.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]
			self.L = [[1.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]
			self.B = [[0.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]
			self.A = [[0.0 for j in range (self.roundsNum+1)] for i in range (self.chunksNum)]

			self.x             = [-1.0 for j in range (self.roundsNum+1)]#工人数
			self.iAct          = [-1.0 for j in range (self.roundsNum+1)]#激活值对应的块标
			
			# Do simulation
			for j in range(self.roundsNum):
				self.printHeadline (j)
				self.showInput  = []
				self.showMisc   = []
				self.showResult = []
				#!!!
				self.Sim = [0.0 for x in range (self.chunksNum)]
				self.numPres = [0.0 for x in range (self.chunksNum)]
				#!!!
				for i in range(self.chunksNum):
					#Attention: It is possible that a chunk is "presented" more than once during one round!
					# 块出现的次数，可能在一轮中出现了几次
					self.n[i][j+1] = self.n[i][j] + self.delta(i-self.iAct[j])*(self.e[i][j]*self.n[i][j]-self.n[i][j]+1)
					self.e[i][j+1] = self.e[i][j] + self.delta(i-self.iAct[j])*(1-self.e[i][j])
					# 基极激活度
					self.B[i][j+1] = log (self.n[i][j+1]/(1-self.d)) - self.d*log(self.model.getLifetime(j,i))
					# A为每个块的激活值
					self.A[i][j+1] = self.B[i][j+1] + self.P*self.model.getSimilarity (j, i,self.showInput, self.showMisc, False) + (1-self.e[i][j+1])*self.Amin + self.model.getNoise(j)
					self.Sim[i] = self.P*self.model.getSimilarity (j, i, self.showInput, self.showMisc, False)
					self.numPres[i] = self.delta(i-self.iAct[j])
                                                                       
				Activation  = [row[j+1] for row in self.A]
				#某一轮中最大的激活值
				maxActivation = self.A[0][j+1]
				for i in range(1, self.chunksNum):
					maxActivation = self.Max(self.A[i][j+1], maxActivation)
				with open ("Activation_argmax %d.csv" %(j,), "w") as f:
					for i in range(len(Activation )):
						f.write("%e; %e; %e; %e \n" %(Activation[i], self.B[i][1], self.Sim[i], self.numPres[i]))
				self.foo = []
				# 被激活的块所对应的动作
				self.x[j+1]    = self.model.getAction (j, maxActivation, Activation, self.showInput, self.showMisc)
				self.iAct[j+1] = self.model.getActiveIndex (j, self.x[j+1], self.showResult, self.showMisc)
				self.printResultLine (j, maxActivation)

			#self.writeResult()
			print ("\nSimulation finished.")
			return sum(self.model.results)
		else:
			print ('CognitiveArchitecture.simulate(): Fatal error! No rounds and/or chunks.')
			return False, -1
	
	def printResultLine (self, j, maxActivation):
		if (len(self.showInput) < 1):
			self.showInput.append(0)
		
		if (len(self.showResult) < 1):
			self.showResult.append(0)
		
		if (len(self.showMisc) < 1):
			self.showMisc.append(0)
		
		if (self.output):
			if (self.algorithmicDiff == False):
				print ("{0:<7}  {1:<12.0f} {2:<20.3e}  {3:<20.8f}  {4:<30.4e}  {5:<25.4e} {6:<20f}".format(j,self.showInput[0], maxActivation,self.x[j+1], self.showMisc[0], self.showResult[0], self.iAct[j+1]))
			else:
				print j
				print self.showInput[0]
				print maxActivation
			# print '--------------------------sdfsdf-----------------------'
			# print self.x[j+1]
			# print self.showMisc[0]
			# print self.showResult[0]
			# print self.iAct[j+1]
			# print "Round", j, "finished"
		
		if (self.writeCsv):
			with open(self.getFilename(), 'a') as csvfile:
				#sumMax    = self.iMax[j+1] % 18 + 4
				#resultMax = floor ((self.iMax[j+1]%36)/18)
				#actionMax = floor (self.iMax[j+1]/36)
				sumAct    = self.iAct[j+1] % 18 + 4
				
				writeResult = csv.writer (csvfile, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
				#writeResult.writerow([j, self.showInput[0], sumMax, resultMax, \
                                                      #actionMax, maxActivation, self.x[j+1],\
                                                      #self.showResult[0], self.iAct[j+1], sumAct])

	
	def printHeadline (self, j):
		if (self.output):
			if (j % self.headlineRepeat == 0):
				if (j != 0):
					print (" ")
				print ("#(j)  input(速度pj)  activation(最大激活值)  x(下一轮挡位)  misc(showMisc[0]，下一轮随机挡位数量)  result(showResult下一轮速度)  iAct")
				print ("-----  -----------   ---------------------   -------------  ------------------------------------   --------------------------    ----")
			
		if (j == 0 and self.writeCsv):
			with open(self.getFilename(), 'w') as csvfile:
				writeHead = csv.writer (csvfile, delimiter=',', quotechar='|', \
				                        quoting=csv.QUOTE_MINIMAL)
				writeHead.writerow (['Round','Input','iMax','sumMax','resultMax','actionMax', \
				                     'Activation','Action','Result','iAct','sumAct'])
	
	def printOptions (self):
		if (self.output):
			print ("Printing output.")
		else:
			print ("Will not print output.")
		
		if (self.writeCsv):
			fileExists = "";
			if (os.path.isfile (self.getFilename())):
				fileExists = "Existing file will be overwritten."
			print ("Saving results in {file}.csv. {exists}".format(file=self.model.getFilename(), exists=fileExists))
		else:
			print ("Will not save data.")
		
		if (self.debug):
			print ("Printing debug information.")
		else:
			print ("Will not print debug information.")

		if (self.algorithmicDiff):
			print ("Saving AlgoPy results in AlgoPy_Results.csv.")
	
	def getFilename (self):
		filename = ""
		
		if (self.filename != None):
			filename = self.filename
		elif (self.model != None):
			filename = self.model.getFilename()+".csv"
		else:
			filename = "result.csv"
			
		return filename
	
	def writeResult (self):
		result = self.model.getResult()
		print 'result: ',result
		if (self.filename == None):
			with open("results.csv", 'a') as csvfile:
				writeResult = csv.writer (csvfile, delimiter=',', escapechar=" ",quoting=csv.QUOTE_NONE)
				writeResult.writerow ([self.model.tau, self.d, self.P, result])
		else:
			with open(self.filename, 'a') as csvfile:
				writeResult = csv.writer (csvfile, delimiter=',', escapechar=" ",quoting=csv.QUOTE_NONE)
				writeResult.writerow ([self.model.tau, self.d, self.P, result])
#		
# End of file actr.py
#

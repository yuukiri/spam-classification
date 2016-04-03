### spam_clas.py ###
### including a spamClassification class to acquire data and perform actual calculation ###
import numpy, time
from sklearn import preprocessing
from MacLearning import GaussianNaiveBayes as GNB
from MacLearning import LearningUtils
from MacLearning import NeuralNetworks as NN
from sklearn.svm import SVC
class spamClassification(object):
	def __init__(self, normalize=True):
		self.lu = LearningUtils()
		self.__getData(normalize)

	def __getData(self, normalize=True):
		## assuming all the data are placed in the same folder
		self.spamData = numpy.genfromtxt('spambase.data', delimiter=',')
		trainIndex = numpy.random.choice(len(self.spamData), numpy.rint(len(self.spamData)*0.8), replace=False)
		testIndex = list(set(range(len(self.spamData))) - set(trainIndex))
		self.xTrain = self.spamData[trainIndex,:-1]
		self.yTrain = self.spamData[trainIndex,-1]
		self.xTest = self.spamData[testIndex, :-1]
		self.yTest = self.spamData[testIndex,-1]
		if normalize:
			# normalize data to mean 0 and std 1
			self.xTrain = preprocessing.scale(self.xTrain)
			self.xTest = preprocessing.scale(self.xTest)

	def run1Classification(self, clfObj, drawLearningCurve=True, lCurveFile=None, calcPAR=True, kFoldCV=True, featureSelection=False):
		clfObj.fit(self.xTrain, self.yTrain)
		yPred = clfObj.predict(self.xTest)
		mse = numpy.mean((yPred-self.yTest)**2)
		print("mean-squared error: %s" % mse)
		if drawLearningCurve:
			lCurve = self.lu.learningCurve(self.xTrain, self.yTrain, clfObj)
			if lCurveFile is not None:
				try:
					numpy.savetxt(lCurveFile, lCurve)
				except ValueError:
					print("cannot save the file")
			else:
				print(lCurve)
		if kFoldCV:
			mseList = self.lu.kFoldCrossValidation(self.spamData[:,:-1], self.spamData[:,-1], clfObj)
			print("k-fold CV result: %s" % mseList)
		if featureSelection:
			print("running feature selection...")
			useInd = self.lu.simpleFeatureSelection(self.xTrain, self.yTrain, clfObj)
			print("The selected indices are: %s" % useInd)
			clfObj.fit(self.xTrain[:,useInd], self.yTrain)
			yPred = clfObj.predict(self.xTest[:,useInd])
			mse = numpy.mean((yPred-self.yTest)**2)
			print("mean-squared error: %s" % mse)
		if calcPAR:
			par = self.lu.accuracyPrecision(yPred, self.yTest)
			print("Accuracy: %s" % par[0])
			print("Precision: %s" % par[1])
			print("Recall: %s" % par[2])
		return clfObj

if __name__ == '__main__':
	spamClf = spamClassification()
	#note the time evaluation here is merely for reference purpose
	start_time = time.time()
	clf = GNB()
	gnb = spamClf.run1Classification(clf, lCurveFile="gnbLCurve.txt")
	difftime = time.time() - start_time
	print("took %s seconds" % difftime)
	start_time = time.time()
	clf = NN()
	nnet = spamClf.run1Classification(clf, lCurveFile="nnetLCurve.txt")
	difftime = time.time() - start_time
	print("took %s seconds" % difftime)
	start_time = time.time()
	clf = SVC()
	nusvc = spamClf.run1Classification(clf, lCurveFile="svcLCurve.txt")
	difftime = time.time() - start_time
	print("took %s seconds" % difftime)
	## for my home made ones, neural network is better
	## but still behind compared to svm
	print("Done!")

### implementation of naive bayes, neural networks, and svm by chenxu ###
import numpy
class GaussianNaiveBayes(object):
	def __init__(self):
		#default values, default model
		#classes is the unique labels
		self.classes = numpy.array([0,1])
		# mean and std associated with each class
		self.mu = numpy.zeros((self.classes.shape[0],1))
		self.sigma = numpy.ones((self.classes.shape[0],1))
		# p(y==0) and p(y==1)
		self.prior = numpy.zeros(self.classes.shape[0])

	def fit(self, X, y, epsilon=1.0e-9):
		'''
		return a fitted Gaussian Naive Bayes Classifier
		Params:
		X: ndarray, features
		y: 1-d array, labels
		data: n+1 ndarray with features and labels as the last column
		'''
		#get no. of rows and features
		rows, features = X.shape
		#reset classes
		self.classes = numpy.unique(y)
		#reinitialize params
		classnum = self.classes.shape[0]
		self.mu = numpy.zeros((classnum, features))
		self.sigma = numpy.zeros((classnum, features))
		self.prior = numpy.zeros(classnum)
		# calculate params
		for i, label in enumerate(self.classes):
			xRows = X[y == label, :]
			self.mu[i,] = numpy.mean(xRows, axis=0)
			self.sigma[i,] = numpy.var(xRows, axis=0) + epsilon # epsilon here is used to guarantee that sigma is not zero
			self.prior[i] = numpy.float(xRows.shape[0]) / float(rows)
		return self

	def __calcLikelyHood(self, X):
		'''
		calculate joint log-likelihood
		Params:
		X: ndarray, test features
		'''
		likelihood = numpy.zeros((X.shape[0], len(self.classes)))
		for i in range(len(self.classes)):
			logPrior = numpy.log(self.prior[i])
			logPdf = -0.5 * numpy.sum(numpy.log(2.0 * numpy.pi*self.sigma[i,])) \
					-0.5 * numpy.sum(((X - self.mu[i,])**2)/self.sigma[i,],1)
			lklh = logPrior + logPdf
			likelihood[:,i] = lklh
		return likelihood

	def predict(self, X):
		'''
		predict the test set
		Params:
		X: ndarray, test features
		'''
		lklh = self.__calcLikelyHood(X)
		return self.classes[numpy.argmax(lklh, axis=1)]

class NeuralNetworks(object):
	'''
	1-hidden layer neural network, specially for classification
	which learns f(x) = W2*g(W1^T*x + b1) + b2
	see http://ufldl.stanford.edu/wiki/index.php/Neural_Networks for details
	'''
	def __init__(self, hiddenUnit=10, outLayer=1, learningRate=0.01, regularization=0.01):
		self.nUnit = hiddenUnit
		self.oLayer = outLayer
		# Reasonable values are in the [0.001, 1] range.
		# can also adjust with learningRate/rows of the training set
		# but this may cause the Gradient descent to converge slow
		self.lRate = learningRate
		# regularization strength to prevent overfitting
		self.alpha = regularization
		# parameters of the neural network
		self.params = self.__initializeParamas()
		# loss
		self.loss = []

	def __initializeParamas(self):
		#W1 is the input weight, W2 the hidden weight
		#b1 is the bias for hidden, b2 the bias for the output layer
		W1 = numpy.ones((1, self.nUnit))
		b1 = numpy.zeros((1, self.nUnit))
		W2 = numpy.ones((self.nUnit, 1))
		b2 = numpy.zeros((1, 1))
		return (W1, b1, W2, b2)

	def setHiddenUnits(self, x):
		try:
			nunit = int(x)
			self.nUnit = nunit
		except ValueError:
			print("The input is not a number.")

	def setLearningRate(self, x):
		try:
			lrate = float(x)
			self.lRate = lrate
		except ValueError:
			print("The input is not a number.")

	def setRegularization(self, x):
		try:
			reg = float(x)
			self.alpha = reg
		except ValueError:
			print("The input is not a number.")

	def __sigmoid(self, x):
		#use sigmoid function for binary classification, this is a centered version:
		val = 1.0/(1.0 + numpy.exp(-(x - numpy.mean(x))/numpy.std(x)))
		return val

	def __softmax(self, x):
		ez = numpy.exp(x)
		val = ez/sum(ez)
		return val

	def __calcLoss(self, X, y, params=None):
		'''
		loss function, Corss-Entropy: L = -1/N*sum[y(n,i)*log(yhat(n,i))] + alpha * (W1^2 + W2^2)
		'''
		if params is None:
			W1, b1, W2, b2 = self.params
		else:
			W1, b1, W2, b2 = params
		# forward propagation
		z1 = X.dot(W1) + b1
		a1 = numpy.tanh(z1)
		z2 = a1.dot(W2) + b2
		yhat = self.__sigmoid(z2)
		loss = -numpy.log(yhat)[:,0].dot(y)/len(y) + self.alpha * (numpy.sum(numpy.square(W1))+numpy.sum(numpy.square(W2)))
		return loss

	def __fit(self, X, y, params=None):
		'''also available for independent usage'''
		if params is None:
			W1, b1, W2, b2 = self.params
		else:
			W1, b1, W2, b2 = params
		# update paramas using Gradient descent
		z1 = X.dot(W1) + b1
		a1 = self.__sigmoid(z1)#numpy.tanh(z1)#should consider sigmoid
		z2 = a1.dot(W2) + b2
		yhat = self.__sigmoid(z2)
		# backward propagation
		delta3 = yhat[:,0] - y
		delta3 = delta3.reshape((delta3.shape[0],1))
		dW2 = (a1.T).dot(delta3)
		db2 = numpy.sum(delta3)
		delta2 = delta3.dot(W2.T) * (1 - numpy.square(a1))#numpy.exp(-a1)/numpy.square(1+numpy.exp(-a1))
		dW1 = numpy.dot(X.T, delta2)
		db1 = numpy.sum(delta2, axis=0)

		dW2 += self.alpha * W2
		dW1 += self.alpha * W1
		# update with learning rate
		W1 += -self.lRate * dW1
		b1 += -self.lRate * db1
		W2 += -self.lRate * dW2
		b2 += -self.lRate * db2
		return (W1, b1, W2, b2)

	def fit(self, X, y, iteration=100, randSeed=0, calculateLoss = False):
		# Initialize the parameters to random values.
		# this is only used to meet what is available in nnet in R
		numpy.random.seed(randSeed)
		W1 = numpy.random.randn(X.shape[1], self.nUnit)
		b1 = numpy.zeros((1, self.nUnit))
		W2 = numpy.random.randn(self.nUnit, self.oLayer)
		b2 = numpy.zeros((1, self.oLayer))
		self.params = (W1, b1, W2, b2)
		for k in xrange(iteration):
			self.params = self.__fit(X, y)
			if calculateLoss:
				loss = self.__calcLoss(X,y)
				self.loss.append(loss)
				print("Iteration %s, loss: %s" % (k+1, loss))
		return self

	def predict(self, X):
		W1, b1, W2, b2 = self.params
		# forward propagation
		z1 = X.dot(W1) + b1
		a1 = self.__sigmoid(z1)#numpy.tanh(z1)
		z2 = a1.dot(W2) + b2
		yhat = self.__sigmoid(z2)
		yhat = yhat.reshape((yhat.shape[0],))
		boolClass = yhat >= 0.5
		classifier = boolClass.astype(int)
		return classifier

class LearningUtils(object):
	'''simple learning utilities'''
	def __init__(self):
		pass

	def simpleFeatureSelection(self, X, y, learningObj, percentile=25):
		'''
		simple feature selection method, returns the list of indeices of optimized features
		Params:
		X: training set
		Y: training labels
		learningObj: a learning model
		percentile: remove the features with errors greater than the percentile of the total error
		'''
		# here use the 80:20 split
		# this should be customizable, but I do not want too many parameters for a simple feature selection method
		rows = X.shape[0]
		trainIndex = numpy.random.choice(rows, numpy.rint(rows*0.8), replace=False)
		testIndex = list(set(range(rows)) - set(trainIndex))
		xTrain = X[trainIndex,]
		yTrain = y[trainIndex,]
		xTest = X[testIndex,]
		yTest = y[testIndex,]
		mseList = []
		for k in range(X.shape[1]):
			#list of indices
			featureIndex = range(X.shape[1])
			# remove one and then train the model
			featureIndex.remove(k)
			newX = xTrain[:,featureIndex]
			learningObj.fit(newX, yTrain)
			yPred = learningObj.predict(xTest[:,featureIndex])
			# calculate mean-squared error as the selection standard
			mse = numpy.mean((yPred-yTest)**2)
			mseList.append(mse)
		msePer = numpy.percentile(mseList, percentile)
		useIndex = [k for (k, mse) in enumerate(mseList) if mse > msePer]
		return useIndex

	def learningCurve(self, X, y, learningObj,datapoint=20, minTrain=100):
		interv = (X.shape[0] - minTrain)/datapoint
		# a list of different training sets
		trainSizes = [minTrain + interv * k for k in xrange(datapoint)]
		trainErrors = [0.0]*datapoint
		lRes = numpy.array([trainSizes,trainErrors, trainErrors]).T
		rows = X.shape[0]
		for k, tSize in enumerate(trainSizes):
			trainIndex = numpy.random.choice(rows, tSize, replace=False)
			testIndex = list(set(range(rows)) - set(trainIndex))
			xTrain = X[trainIndex,]
			yTrain = y[trainIndex,]
			xTest = X[testIndex,]
			yTest = y[testIndex,]
			learningObj.fit(xTrain, yTrain)
			#training error
			yPred = learningObj.predict(xTrain)
			mse = numpy.mean((yPred - yTrain)**2)
			lRes[k, 1] = mse
			#testing error
			yPred = learningObj.predict(xTest)
			mse = numpy.mean((yPred-yTest)**2)
			lRes[k, 2] = mse
		return lRes

	def accuracyPrecision(self, yhat, y, positive=1):
		'''
		calculate accuracy, precision, and recall
		assuming a binary classification, and positive is 1
		'''
		accuracy = sum(yhat==y)/float(len(y))
		truePosInd = numpy.where((yhat==y)&(yhat==positive))
		truePos = sum(yhat[truePosInd])
		precision = truePos/float(sum(yhat))
		falseNegInd = numpy.where((yhat!=y)&(y==positive))
		falseNeg = sum(y[falseNegInd])
		recall = truePos/float(truePos + falseNeg)
		return (accuracy, precision, recall)

	def kFoldCrossValidation(self, X, y, learningObj, k=10):
		totalRows = X.shape[0]/k*k
		# sample 10 different index set
		indices = range(totalRows)
		#indList = []
		singleSize = totalRows/k
		mseList = []
		for i in xrange(k):
			testIndex = numpy.random.choice(indices, size=singleSize, replace=False)
			#indList.append(iList)
			indices = list(set(indices) - set(testIndex))
			trainIndex = list(set(range(totalRows)) - set(testIndex))
			xTrain = X[trainIndex,]
			yTrain = y[trainIndex,]
			xTest = X[testIndex,]
			yTest = y[testIndex,]
			learningObj.fit(xTrain, yTrain)
			#training error
			yPred = learningObj.predict(xTrain)
			mse = numpy.mean((yPred - yTrain)**2)
			mseList.append(mse)
		return mseList



#class NeuralNetwork

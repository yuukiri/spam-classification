### spam_class.r ###
# classifier comparison #
# script by chenxu #
library(nnet)#neural networks
library(e1071)#svm, naive bayes
setwd(getwd())
##### read data and split into train and test sets #####
spam.data <- read.table('spambase.data',sep = ',')
spam.index <- seq(1, nrow(spam.data))
#use 80% of data as training set
spam.train.index <- sample(spam.index, size=round(nrow(spam.data)*0.8))
spam.train <- spam.data[spam.train.index,]
#the rest as testing set
spam.test.index <- spam.index[-spam.train.index]
spam.test <- spam.data[spam.test.index,]
#split the features and the predictors
y.train <- as.factor(spam.train[,58])
x.train <- spam.train[,1:57]
y.test <- as.factor(spam.test[,58])
x.test <- spam.test[1:57]
#standardize features: center and scale all predictors to mean 0 and std 1
# this is very important to nnet #
for(i in seq(1,57)){
	x.train[,i] <- scale(x.train[,i],center = T, scale = T)
	x.test[,i] <- scale(x.test[,i], center = T, scale = T)
}
########## end of reading ###########
### declaring functions ###
### calculate accuracy, precision, and recall ###
accuracyPrecision <- function(yhat, y){
	res = rep(0,3)
	accuracy = sum(yhat==y)/length(y)
	truePosInd = which((yhat==y)&(yhat==1))
	truePos = sum(yhat[truePosInd]==1)
	precision = truePos/sum(yhat==1)
	falseNegInd = which((yhat!=y) & (y==1))
	falseNeg = sum(y[falseNegInd]==1)
	recall = truePos/(truePos + falseNeg)
	res[1] = accuracy
	res[2] = precision
	res[3] = recall
	return(res)
}

kFoldCrossValidation <- function(x, y, method, k=10){
	total.rows <- round(nrow(x)/k)*k
	indices <- rep(T, total.rows)
	cv.mse <- rep(0, k)
	for(i in seq(k)){
		#get available indices
		a.indices <- which(indices==T)
		test.index <- sample(a.indices, size=round(total.rows/k))
		indices[test.index] <- F
		x.test <- x[test.index,]
		y.test <- y[test.index]
		x.train <- x[-test.index,]
		y.train <- y[-test.index]
		if(method=="svm"){
			model <- svm(x=x.train, y=y.train, type='nu-classification', kernel = 'radial', nu = 0.2)
		}
		if(method=="naivebayes"){
			model <- naiveBayes(x.train, y.train)
		}
		if(method=="nnet"){
			model <- nnet(y.train~., data = x.train, size = 10 , rang = 0.5)
		}
		y.pred <- predict(model, newdata = x.test, type = "class")
		y.pred <- as.factor(y.pred)
		mse <- mean((as.numeric(y.pred)-as.numeric(y.test))^2)
		cv.mse[i] <- mse
	}
	return(cv.mse)
	#smooth for output purpose
	#cv.mse.smooth <- lowess(seq(1,10), cv.mse)
	#return(cv.mse.smooth$y)
}

### for different size of training set, calculate the training and testing errors (mse) ###
learningCurve <- function(x, y, method, datapoint=20, minTrain=100){
	interv = round((nrow(x) - minTrain)/datapoint)
	# a list of different training sets
	trainSizes = seq(0,datapoint-1)*interv + minTrain
	trainErrors = rep(0, datapoint)
	testErrors = rep(0, datapoint)
	lRes = data.frame(trainSizes, trainErrors, testErrors)
	x.index <- seq(1, nrow(x))
	#vary training size and calculate mse for different training sizes
	for( k in seq(1, datapoint)){
		tSize = trainSizes[k]
		x.train.index <- sample(x.index, size=tSize)
		x.test.index <- x.index[-x.train.index]
		x.train = x[x.train.index,]
		y.train = y[x.train.index]
		x.test = x[x.test.index,]
		y.test = y[x.test.index]
		if(method=="svm"){
			model <- svm(x=x.train, y=y.train, type='nu-classification', kernel = 'radial', nu = 0.2)
		}
		if(method=="naivebayes"){
			model <- naiveBayes(x.train, y.train)
		}
		if(method=="nnet"){
			model <- nnet(y.train~., data = x.train, size = 10 , rang = 0.5)
		}
		y.pred <- predict(model, newdata = x.test, type = "class")
		y.pred <- as.factor(y.pred)
		mse <- mean((as.numeric(y.pred)-as.numeric(y.test))^2)
		lRes$testErrors[k] <- mse
		y.pred <- predict(model, newdata = x.train, type = "class")
		y.pred <- as.factor(y.pred)
		mse <- mean((as.numeric(y.pred)-as.numeric(y.train))^2)
		lRes$trainErrors[k] <- mse
	}
	#smooth the errors, for plotting purpose
	smooth.train <- lowess(lRes$trainSizes,lRes$trainErrors)
	smooth.test <- lowess(lRes$trainSizes,lRes$testErrors)
	lRes$trainErrors <- smooth.train$y
	lRes$testErrors <- smooth.test$y
	return(lRes)
}
### drawing a learning curve ###
drawLearningCurve <- function(lRes, method, expecting.err=0.05){
	mainTitle = paste("Learning curve of", method)
	# this is for ylim
	y.train.min <- min(lRes$trainErrors)
	y.train.max <- max(lRes$trainErrors)
	y.test.min <- min(lRes$testErrors)
	y.test.max <- max(lRes$testErrors)
	y.min <- min(y.train.min, y.test.min, expecting.err)
	y.max <- max(y.train.max, y.test.max, expecting.err)
	plot(x=lRes$trainSizes, y=lRes$testErrors, type="l", col=2, log="y", main = mainTitle, xlab = "Train size", ylab = "Classification error", ylim = c(y.min, y.max))
	grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
	lines(x=lRes$trainSizes, y=lRes$trainErrors, col=3)
	abline(h = expecting.err, untf = FALSE, lty = 2, col = 'darkred')
	legend('bottomright', inset = 0.015, c("Test Error","Train Error", "Expected Error"), pch=c(20,20,20),col=c("red","green", "darkred"), cex = 0.7)
}

### simple feature selection ###
featureSelection <- function(x.train, y.train, x.test, y.test, method){
	fs.err <- rep(0, ncol(x.train))
	# the purpose is to find out, if leave one out, if the error rate will decrease
	for(i in seq(1, ncol(x.train))){
		if(method=="svm"){
			fs.model <- svm(x=x.train[,-i], y=y.train, type='nu-classification', kernel = 'radial', nu = 0.2)
		}
		if(method=="naivebayes"){
			fs.model <- naiveBayes(x.train[,-i], y.train)
		}
		if(method=="nnet"){
			fs.model <- nnet(y.train~., data = x.train[,-i], size = 10 , rang = 0.5)
		}
		y.fs.pred <- predict(fs.model, x.test[,-i], type="class")
		y.fs.pred <- as.factor(y.fs.pred)
		#calculate mse for each case
		fs.err[i] <- mean((as.numeric(y.fs.pred)-as.numeric(y.test))^2)
	}
	# get the 25% features that increase the classification error
	bad.feat <- which(fs.err < quantile(fs.err, probs = 0.25)[1])
	print(bad.feat)
	if(method=="svm"){
		new.model <- svm(x = x.train[,-bad.feat], y = y.train, type='nu-classification', kernel = 'radial', nu = 0.2)
	}
	if(method=="naivebayes"){
		new.model <- naiveBayes(x.train[,-bad.feat], y.train)
	}
	if(method=="nnet"){
		new.model <- nnet(y.train~., data = x.train[,-bad.feat], size = 10 , rang = 0.5)
	}
	y.new.pred <- predict(new.model, x.test[,-bad.feat], type="class")
	y.new.pred <- as.factor(y.new.pred)
	new.mse <- mean((as.numeric(y.new.pred)-as.numeric(y.test))^2)
	print(new.mse)
}
######## end of function declaration #######

#### testing different learning method ####
### the methods tested here include naive bayes, svm (nu-classification), and 10-hidden layer neural network ###
### naive bayes ###
nb.model <- naiveBayes(x.train, y.train)
nb.pred <- predict(nb.model, newdata = x.test, type = "class")
nb.mse <- mean((as.numeric(nb.pred)-as.numeric(y.test))^2)
nb.apr <- accuracyPrecision(nb.pred, y.test)
#nb.fs <- featureSelection(x.train, y.train, x.test, y.test, method="naivebayes")
nb.lres <- learningCurve(x.train, y.train, method="naivebayes")
nb.cv.mse <- kFoldCrossValidation(spam.data[,1:57], as.factor(spam.data[,58]), "naivebayes")
### end of naive bayes ###

### support vector machine, with nu-classification ###
### here only radial basis function is used as the kernel ###
svm.model <- svm(x = x.train, y = y.train, type='nu-classification', kernel = 'radial', nu = 0.2)
svm.pred <- predict(svm.model, newdata = x.test, type="class")
svm.mse <- mean((as.numeric(svm.pred)-as.numeric(y.test))^2)
svm.apr <- accuracyPrecision(svm.pred, y.test)
svm.fs <- featureSelection(x.train, y.train, x.test, y.test, method="svm")
svm.lres <- learningCurve(x.train, y.train, method="svm")
svm.cv.mse <- kFoldCrossValidation(spam.data[,1:57], as.factor(spam.data[,58]), "svm")
### end of svm ###

### neural networks, with 10 hidden layers ###
nnet.model <- nnet(y.train~., data = x.train, size = 10 , rang = 0.5)
nnet.pred <- predict(nnet.model, newdata = x.test, type="class")
nnet.pred <- as.factor(nnet.pred)
nnet.mse <- mean((as.numeric(nnet.pred)-as.numeric(y.test))^2)
nnet.apr <- accuracyPrecision(nnet.pred, y.test)
nnet.fs <- featureSelection(x.train, y.train, x.test, y.test, method="nnet")
nnet.lres <- learningCurve(x.train, y.train, method="nnet")
nnet.cv.mse <- kFoldCrossValidation(spam.data[,1:57], as.factor(spam.data[,58]), "nnet")
### end of nnet ###

### generate learning curves ###
pdf(file="learning_curve_nb.pdf", width = 6, height = 6, colormodel = "cmyk");
drawLearningCurve(nb.lres,"Naive Bayes")
dev.off()

pdf(file="learning_curve_svm.pdf", width = 6, height = 6, colormodel = "cmyk");
drawLearningCurve(svm.lres,"SVM")
dev.off()

pdf(file="learning_curve_nnet.pdf", width = 6, height = 6, colormodel = "cmyk");
drawLearningCurve(nnet.lres,"Neural Networks (57-10-1)")
dev.off()

### plot cross validation ###
pdf(file="cross-validation.pdf", width = 6, height = 6, colormodel = "cmyk");
cv.y.min <- 0#min(nb.cv.mse, svm.cv.mse, nnet.cv.mse)
cv.y.max <- max(nb.cv.mse, svm.cv.mse, nnet.cv.mse)
plot(x=seq(1,10), y=nb.cv.mse, type="l", col=2, main = "10-Fold Cross Validation", xlab = "K", ylab = "Classification error", ylim = c(cv.y.min, cv.y.max))
grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
lines(x=seq(1, 10), y=svm.cv.mse, col=3)
lines(x=seq(1, 10), y=nnet.cv.mse, col="blue")
legend('bottomright', inset = 0.015, c("Naive Bayes","Neural Network", "Support-Vector Machine"), pch=c(20,20,20),col=c("red","green", "blue"), cex = 0.7)
dev.off()

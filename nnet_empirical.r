### nnet_empirical.r ###
######## study on nnet with spam.data ###########
# script by chenxu shao #
library(nnet)#neural networks
setwd(getwd())
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
y.test <- spam.test[,58]
x.test <- spam.test[1:57]

#standardize features: center and scale all predictors to mean 0 and std 1
# this is very important to nnet #
for(i in seq(1,57)){
	x.train[,i] <- scale(x.train[,i],center = T, scale = T)
	x.test[,i] <- scale(x.test[,i], center = T, scale = T)
}

############# neural networks ###############
nUnit = 10
nSet = 10

### test error vs hidden units ###
err.test <- matrix(nrow = nSet, ncol = nUnit)
for(i in seq(1,nUnit)){#1-10 hidden units
	#set.seed(i)
	for(j in seq(1,nSet)){
		set.seed(j);
		spam.fit <- nnet(y.train~., data = x.train, size = i , rang = 0.5)
		y.pred <- predict(spam.fit, newdata = x.test, type = "class")
		y.mse <- mean((as.numeric(y.pred)-y.test)^2)#mean-squared error
		err.test[j,i] <- y.mse
	}
}
pdf(file="spam_nnet_mse.pdf", width = 8, height = 6, colormodel = "cmyk")
boxplot(err.test, main = "Test errors vs. numbers of hidden units, with different starting values", xlab = "No. of hidden units", ylab = "Mean-squared test errors")
dev.off()
err.test.mean <- colMeans(err.test)
print(err.test.mean)
#smallest mse: 0.06434783
iMin <- which.min(err.test.mean)
print(iMin)
### finding the best model ###
err.test <- matrix(nrow = nSet, ncol = nUnit+1)
for(i in seq(0,nUnit,by=1)){
	#set.seed(i)
	for(j in seq(1,nSet)){
		set.seed(j);
		spam.fit <- nnet(y.train~., data = x.train, size = iMin , rang = 0.5, decay = 0.1*i)
		y.pred <- predict(spam.fit, newdata = x.test, type = "class")
		y.mse <- mean((as.numeric(y.pred)-y.test)^2)#mean-squared error
		err.test[j,i+1] <- y.mse
	}
}
pdf(file="spam_nnet_mse_decay.pdf", width = 8, height = 6, colormodel = "cmyk")
x.axis <- c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
boxplot(err.test, names = x.axis, main = "Test errors vs. weight decay parameters, with different starting values", xlab = "Weight decay parameters", ylab = "Mean-squared test errors")
dev.off()
#axis(1, at = seq(1,11,by=1), labels = seq(0, 1, by=0.1))
err.test.mean <- colMeans(err.test)
print(err.test.mean)
wMin <- 0.1*(which.min(err.test.mean)-1)
print(wMin)

### use decay parameters ###
pdf(file="spam_nnet_mse_decay_2.pdf", width = 8, height = 6, colormodel = "cmyk")
x <- seq(0,1,by = 0.1)
plot(x, err.test.mean, type = "b", main = "Averaged test errors vs. weight decay parameters", xlab = "Weight decay parameters", ylab = "Mean-squared test errors")
box()
dev.off()

### finding a filter as a threshold probability ###
prob.seq <- seq(0.5, 1.0, by = 0.01)
prob.len <- length(prob.seq)
mis.err <- matrix(nrow = nSet, ncol = prob.len)
for(j in seq(1,prob.len)){
	#set.seed(j)
	for(i in seq(1,nSet)){
		set.seed(i)
		spam.fit <- nnet(y.train~., data = x.train, size = iMin, rang = 0.5, decay = wMin)
		y.pred <- predict(spam.fit, newdata = x.test, type = "raw")
		mis.err[i,j] <- sum(as.numeric(y.pred >= prob.seq[j] & y.test == 0))/sum(y.test == 0)
	}
}
mis.err.mean <- colMeans(mis.err)
print(mis.err.mean)
prob.min <- prob.seq[which(mis.err.mean <= 0.01)]
print(prob.min)

#early stopping rule
nEp = 10
mis.err.mat <- matrix(nrow = 10, ncol = nEp)
mis.err <- matrix(nrow = 1, ncol = nEp)
w <- c()
for(j in seq(1,10)){
	set.seed(j)
	spam.fit <- nnet(y.train~., data = x.train, size = 3, rang = 0.5, maxit = 10)
	y.pred <- predict(spam.fit, newdata = x.test, type = "class")
	mis.err[j] <- sum(y.pred != 0 && y.test == 0)/sum(y.test == 0)
	#initial weights for the iteration
	w <- cbind(w, spam.fit$wts)
}

for(i in seq(1,10)){#10 networks
	for(j in seq(1,nEp)){
		spam.fit <- nnet(y.train~., data = x.train, Wts = w[,i], size = 3, maxit = 10)
		y.pred <- predict(spam.fit, newdata = x.test, type = "class")
		mis.err.mat[i,j] <- mean((as.numeric(y.pred)-y.test)^2)#mean-squared error
		w[,i] <- spam.fit$wts
	}
}
mis.err <- colMeans(mis.err.mat)
print(mis.err)
pdf(file="spam_nnet_epoch.pdf", width = 8, height = 6, colormodel = "cmyk")
epoch <- seq(10, nEp*10, by = 10)
plot(epoch,mis.err, type = "b", main = "Misclassification error vs. epochs", xlab = "No. of Epochs", ylab = "Misclassification Error")
axis(1, at = epoch)
dev.off()

#######################################################################################################################
## Diabetes data set (binary classification)
## Author: Tina Dhekial-Phukan
## Data source: https://www.kaggle.com/datasets/whenamancodes/predict-diabities
## Primary objective of analysis: To diagnostically predict whether a patient has diabetes, based on certain diagnostic 
                                  #measurements included in the dataset.
##File includes code for: 
  #(1) Data pre-processing and exploratory data analysis.
  #(2) Decision Tree analysis for classification.
#######################################################################################################################

options(digits=12)

## Install libraries
install.packages(c('tibble', 'dplyr', 'readr'))
library(tibble)
library(dplyr)
library(readr)
install.packages("ggplot2")
install.packages("lattice")
library(caret)
library(MASS)
library(dplyr)

## Read in file
diabetes <- read.csv(file="diabetes.csv")
View(diabetes)
dim(diabetes)
str(diabetes)

################################################################################
## Exploratory data analysis
################################################################################

## Checking for columns with missing values
NA.sum <- colSums(is.na(diabetes))
NA.sum

## Descriptive summary statistics
summary(diabetes)

## Checking for zero values in the columns
sum(diabetes$Glucose == 0)
sum(diabetes$BloodPressure == 0)
sum(diabetes$SkinThickness == 0)
sum(diabetes$Insulin == 0)
sum(diabetes$BMI == 0)
sum(diabetes$DiabetesPedigreeFunction == 0)
sum(diabetes$Age == 0)

## Creating a new outcome variable as factor type
diabetes$Disease <- as.factor(ifelse(diabetes$Outcome == 0, "NO", "YES"))

## Create a temporary data set with the 0 values for Glucose, BloodPressure, SkinThickness, Insulin, BMI
## imputed to NA so as to be able to compute the correlation matrix for pairwise complete observations.
temp.dat <- diabetes
temp.dat$Glucose[which(temp.dat$Glucose == 0)] <- "NA"
temp.dat$Glucose <- as.numeric(temp.dat$Glucose)
temp.dat$BloodPressure[which(temp.dat$BloodPressure == 0)] <- "NA"
temp.dat$BloodPressure <- as.numeric(temp.dat$BloodPressure)
temp.dat$SkinThickness[which(temp.dat$SkinThickness == 0)] <- "NA"
temp.dat$SkinThickness <- as.numeric(temp.dat$SkinThickness)
temp.dat$Insulin[which(temp.dat$Insulin == 0)] <- "NA"
temp.dat$Insulin <- as.numeric(temp.dat$Insulin)
temp.dat$BMI[which(temp.dat$BMI == 0)] <- "NA"
temp.dat$BMI <- as.numeric(temp.dat$BMI)

## Box plots of the numeric variables grouped by Disease
library(patchwork)
bxplt1 <- ggplot(temp.dat, aes(x=Disease, y=Pregnancies,color=Disease))+ geom_boxplot()
bxplt2 <- ggplot(temp.dat, aes(x=Disease, y=Glucose,color=Disease))+ geom_boxplot()
bxplt3 <- ggplot(temp.dat, aes(x=Disease, y=BloodPressure,color=Disease))+ geom_boxplot()
bxplt4 <- ggplot(temp.dat, aes(x=Disease, y=SkinThickness,color=Disease))+ geom_boxplot()
bxplt5 <- ggplot(temp.dat, aes(x=Disease, y=Insulin,color=Disease))+ geom_boxplot()
bxplt6 <- ggplot(temp.dat, aes(x=Disease, y=BMI,color=Disease))+ geom_boxplot()
bxplt7 <- ggplot(temp.dat, aes(x=Disease, y=DiabetesPedigreeFunction,color=Disease))+ geom_boxplot()
bxplt8 <- ggplot(temp.dat, aes(x=Disease, y=Age,color=Disease))+ geom_boxplot()
(bxplt1/bxplt2|bxplt3/bxplt4)
(bxplt5/bxplt6|bxplt7/bxplt8)

## Plotting correlation matrix of the numeric variables to check multicollinearity
install.packages("corrplot")
library(corrplot)
corr_mat <- cor(temp.dat[c(-9, -10)], method = "pearson", use = "pairwise.complete.obs")
par(mfrow=c(1,1))
corrplot(corr_mat, method = 'number', tl.cex=0.7, number.cex = 0.6, type = "lower", diag = FALSE)
corrplot(corr_mat, method = 'color', tl.cex=0.6, type = "lower", diag = FALSE,
         mar=c(0,0,1,0), cex.main = 0.9)

## Impute the zero values of Glucose, BloodPressure, SkinThickness, Insulin, BMI
## For rows with Outcome = 0
diabetes$Glucose[which(diabetes$Glucose == 0 & diabetes$Outcome == 0)] <- median(diabetes$Glucose[which(diabetes$Glucose != 0 & diabetes$Outcome == 0)] )
diabetes$BloodPressure[which(diabetes$BloodPressure == 0 & diabetes$Outcome == 0)] <- median(diabetes$BloodPressure[which(diabetes$BloodPressure != 0 & diabetes$Outcome == 0)] )
diabetes$SkinThickness[which(diabetes$SkinThickness == 0 & diabetes$Outcome == 0)] <- median(diabetes$SkinThickness[which(diabetes$SkinThickness != 0 & diabetes$Outcome == 0)] )
diabetes$Insulin[which(diabetes$Insulin == 0 & diabetes$Outcome == 0)] <- median(diabetes$Insulin[which(diabetes$Insulin != 0 & diabetes$Outcome == 0)] )
diabetes$BMI[which(diabetes$BMI == 0 & diabetes$Outcome == 0)] <- median(diabetes$BMI[which(diabetes$BMI != 0 & diabetes$Outcome == 0)] )
## For rows with Outcome = 1
diabetes$Glucose[which(diabetes$Glucose == 0 & diabetes$Outcome == 1)] <- median(diabetes$Glucose[which(diabetes$Glucose != 0 & diabetes$Outcome == 1)] )
diabetes$BloodPressure[which(diabetes$BloodPressure == 0 & diabetes$Outcome == 1)] <- median(diabetes$BloodPressure[which(diabetes$BloodPressure != 0 & diabetes$Outcome == 1)] )
diabetes$SkinThickness[which(diabetes$SkinThickness == 0 & diabetes$Outcome == 1)] <- median(diabetes$SkinThickness[which(diabetes$SkinThickness != 0 & diabetes$Outcome == 1)] )
diabetes$Insulin[which(diabetes$Insulin == 0 & diabetes$Outcome == 1)] <- median(diabetes$Insulin[which(diabetes$Insulin != 0 & diabetes$Outcome == 1)] )
diabetes$BMI[which(diabetes$BMI == 0 & diabetes$Outcome == 1)] <- median(diabetes$BMI[which(diabetes$BMI != 0 & diabetes$Outcome == 1)] )

## Plot univariate histograms of the numeric variables
vnames <- names(diabetes[c(-9, -10)])

## Histogram of untransformed variables
par(mfrow=c(2,2))
for (v in vnames) 
{
  hist(diabetes[, v], main="", ylab="Freq", xlab = paste(v, sep=""))
}

## Histogram of log transformed variables
par(mfrow=c(2,2))
for (v in vnames) 
{
  hist(log(diabetes[, v]), main="", ylab="Freq", xlab = paste(v, sep="", "_log"))
}

## Histogram of square root transformed variables
par(mfrow=c(2,2))
for (v in vnames) 
{
  hist(sqrt(diabetes[, v]), main="", ylab="Freq", xlab = paste(v, sep="", "_sqrt"))
}

## Histogram of inverse transformed variables
par(mfrow=c(2,2))
for (v in vnames) 
{
  hist(1/(diabetes[, v]), main="", ylab="Freq", xlab = paste(v, sep="", "_inv"))
}

summary(diabetes)

################################################################################
## Create training and test data sets using stratified sampling
################################################################################

## Tabulate frequencies for the two levels of the categorical variables
table(diabetes$Disease)

set.seed(1)
train <- createDataPartition(paste(diabetes$Disease, sep = ""), p = 0.7, list = FALSE)
train.data <- diabetes[train, ]
dim(train.data)
test.data <- diabetes[-train, ]
dim(test.data)
Disease.train <- diabetes$Disease[train]
length(Disease.train)
Disease.test <- diabetes$Disease[-train]
length(Disease.test)

################################################################################
## Fit a classification tree to the training data, with Disease as the response  
## and the other variables as predictors.
################################################################################

library(tree)
## Fitting classification tree 
tree.diabetes <- tree(Disease ~ ., data = train.data[-9], method = "recursive.partition", 
                      split = "deviance")
summary(tree.diabetes)

## Get detailed text output of the tree object
tree.diabetes

## Create a plot of the tree
par(mfrow = c(1,1))
plot(tree.diabetes)
text(tree.diabetes, pretty = 0, cex= 0.5)

## Compute training error rates for the unpruned tree
tree.train.pred <- predict(tree.diabetes, train.data[-9], type = "class")
table(tree.train.pred, Disease.train)
mean(tree.train.pred == Disease.train)
mean(tree.train.pred != Disease.train)

## Compute test error rates for the unpruned tree 
tree.test.pred <- predict(tree.diabetes, test.data[-9], type = "class")
table(tree.test.pred, Disease.test)
mean(tree.test.pred == Disease.test)
mean(tree.test.pred != Disease.test)

## Check whether pruning the tree might lead to improved results and determine 
## the optimal tree size
set.seed(2)
cv.Disease <- cv.tree(tree.diabetes, FUN = prune.misclass, K = 10)
names(cv.Disease)

## Get detailed text output of the cv object
cv.Disease

## Plot the error rate as a function of size
par(mfrow=c(1,1))
plot(cv.Disease$size, cv.Disease$dev, type = "b") # tree size versus classification error

## Apply the prune.missclass() to prune the tree to obtain the 5-node tree
prune.Disease <- prune.misclass(tree.diabetes, best = 5)
plot(prune.Disease)
text(prune.Disease, pretty = 0, cex= 0.5)

## Get detailed text output of the tree object
prune.Disease

## Compute training error rates for the pruned tree
tree.pruned.pred.train <- predict(prune.Disease, train.data[-9], type = "class")
table(tree.pruned.pred.train, Disease.train)
mean(tree.pruned.pred.train == Disease.train)
mean(tree.pruned.pred.train != Disease.train)

## Predict the response on the test data, and compute test error rates for the 
## pruned tree 
tree.pruned.pred.test <- predict(prune.Disease, test.data[-9], type = "class")
table(tree.pruned.pred.test, Disease.test)
mean(tree.pruned.pred.test == Disease.test)
mean(tree.pruned.pred.test != Disease.test)

################################################################################
## Bagging
################################################################################

library(randomForest)
## Fit the model with the training data set
set.seed(3)
bag.diabetes <- randomForest(Disease ~., data = train.data[-9], mtry = 8, 
                             importance = TRUE)
bag.diabetes
importance(bag.diabetes)
par(mfrow = c(1,1))
varImpPlot(bag.diabetes)

## Assess performance of the bagged model on the test data set
yhat.bag <- predict(bag.diabetes, newdata = test.data[-9])
plot(x = yhat.bag, y = Disease.test)

## Compute training error rates for the bagged tree 
bag.test.train <- predict(bag.diabetes, train.data[-9], type = "class")
table(bag.test.train, Disease.train)
mean(bag.test.train == Disease.train)
mean(bag.test.train != Disease.train)

## Compute test error rates for the bagged tree 
bag.test.pred <- predict(bag.diabetes, test.data[-9], type = "class")
table(bag.test.pred, Disease.test)
mean(bag.test.pred == Disease.test)
mean(bag.test.pred != Disease.test)

## Change the number of trees grown to 25
bag.diabetes1 <- randomForest(Disease ~., data = train.data[-9], mtry = 8, ntree = 25)
yhat.bag1 <- predict(bag.diabetes1, newdata = test.data[-9])
plot(x = yhat.bag1, y = Disease.test)

## Compute training error rates for the bagged tree with ntree = 25
bag.train.pred1 <- predict(bag.diabetes1, train.data[-9], type = "class")
table(bag.train.pred1, Disease.train)
mean(bag.train.pred1 == Disease.train)
mean(bag.train.pred1 != Disease.train)

## Compute test error rates for the bagged tree with ntree = 25
bag.test.pred1 <- predict(bag.diabetes1, test.data[-9], type = "class")
table(bag.test.pred1, Disease.test)
mean(bag.test.pred1 == Disease.test)
mean(bag.test.pred1 != Disease.test)

importance(bag.diabetes1)
varImpPlot(bag.diabetes1)

################################################################################
## Boosting
################################################################################

install.packages("gbm")
library(gbm)

## Fit the model with the training data set
set.seed(4)
boost.diabetes <- gbm(Outcome ~., data = train.data[-10], distribution = "bernoulli",
                      n.trees = 5000, interaction.depth = 4)
summary(boost.diabetes)

## Use the boosted model to predict probability of "Disease" on the test set. 
boost.test.probs <- predict(boost.diabetes, test.data[-10], type = "response")
boost.test.pred <- rep("NO", 230)
boost.test.pred[boost.test.probs > 0.5] <- "YES"

## Produce a confusion matrix for test data
table(boost.test.pred, Disease.test)
mean(boost.test.pred == Disease.test)
mean(boost.test.pred != Disease.test)

## Tuning the shrinkage parameter
hyper_grid <- expand.grid(
  shrinkage = c(0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.001),
  n.minobsinnode = c(5, 10, 15),
  min_train.error = 0  # a place to hold the error results
)

# total number of combinations
nrow(hyper_grid)

# grid search 
for(i in 1:nrow(hyper_grid)) {
  # reproducibility
  set.seed(123)
  # train model
  gbm.tune <- gbm(
    formula = Outcome ~ .,
    distribution = "bernoulli",
    data = train.data[-10],
    n.trees = 5000,
    interaction.depth = 4,
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = 0.5,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  # add min training error to grid
  hyper_grid$min_train.error[i] <- which.min(gbm.tune$train.error)
  }

hyper_grid %>% 
  dplyr::arrange(min_train.error) %>%
  head(10)

# for reproducibility
set.seed(123)
# train final GBM model with the tuned parameters
gbm.fit.final <- gbm(
  formula = Outcome ~ .,
  distribution = "bernoulli",
  data = train.data[-10],
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.4,
  n.minobsinnode = 5,
  bag.fraction = 0.5, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

par(mar = c(5, 8, 1, 1))
summary(gbm.fit.final, cBars = 10, method = relative.influence, las = 2)

install.packages("vip")
library(vip)
vip::vip(gbm.fit.final)

## Partial dependence plots for "Insulin", "Glucose", "Age" and "SkinThickness"
plot(gbm.fit.final, i = "Insulin")
plot(gbm.fit.final, i = "Glucose")
plot(gbm.fit.final, i = "Age")
plot(gbm.fit.final, i = "SkinThickness")

## Use the tuned model to predict probability of "Disease" on the test set. 
tuned.test.probs <- predict(gbm.fit.final, test.data[-10], type = "response")
tuned.test.pred <- rep("NO", 230)
tuned.test.pred[tuned.test.probs > 0.5] <- "YES"

## Produce a confusion matrix for the test data
table(tuned.test.pred, Disease.test)
mean(tuned.test.pred == Disease.test)
mean(tuned.test.pred != Disease.test)
contrasts(Disease.test)

## Create the ROC curve
library(pROC)
ROC.boost <- roc(Disease.test, tuned.test.probs)
plot(ROC.boost, col = "blue", main = " ROC - tuned boosted model", cex.main = 0.9,
     ylim = c(0, 1.02))
auc(ROC.boost)

################################################################################


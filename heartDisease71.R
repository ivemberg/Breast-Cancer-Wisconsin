#########################################
#
# The process of a machine learning project may not be linear, 
# but there are a number of well-known steps:
#
# 1.Define Problem
# 2.Prepare Data
# 3.Evaluate Algorithms
# 4.Improve Results
# 5.Present Results
#
#########################################

# Set your workspace path
# setwd("~/R/Machine Learning/ProgettoMachineLearning")
install.packages("caret")
library(caret)

# Load Data
dataset = 
  read.csv("76.csv",
    header = FALSE,
    sep=";",
    col.names = 
     c("age",
       "sex",
       "painloc",
       "painexer",
       "relrest",
       "pncaden",
       "cp",
       "trestbps",
       "htn",
       "chol",
       "smoke",
       "cigs",
       "years",
       "fbs",
       "dm",
       "famhist",
       "restecg",
       "ekgmo",
       "ekgmday",
       "ekgyr",
       "dig",
       "prop",
       "nitr",
       "pro",
       "diuretic",
       "proto",
       "thaldur",
       "thaltime",
       "met",
       "thalach",
       "thalrest",
       "tpeakbps",
       "tpeakbpd",
       "dummy",
       "trestbpd",
       "exang",
       "xhypo",
       "oldpeak",
       "slope",
       "rldv5",
       "rldv5e",
       "ca",
       "restckm",
       "exerckm",
       "restef",
       "restewm",
       "exeref",
       "exerwm",
       "thal",
       "thalsev",
       "thalpul",
       "earlrobe",
       "cmo",
       "cday",
       "cyr",
       "target",
       "lmt",
       "ladprox",
       "laddist",
       "diag",
       "cxmain",
       "ramus",
       "om1",
       "om2",
       "rcaprox",
       "rcadist",
       "lvx1",
       "lvx2",
       "lvx3",
       "lvx4",
       "lvf",
       "cathef",
       "junk")
  )

# to remove rows with null attribute values 
dataset = dataset[complete.cases(dataset), ]
# factor for target attribute
dataset$V9 = ifelse(dataset$V9>1,1,0)
dataset$V9 = factor(dataset$V9)

# create a list of 80% of the rows in the original dataset for training
validation_index = createDataPartition(dataset$V9, p=0.80, list=FALSE)
# select 20% of the data for validation
validation = dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset = dataset[validation_index,]

# type of data
class(dataset)
# names of attributes
names(dataset)
# dimensions of dataset
dim(dataset)
# list types for each attribute
sapply(dataset, class)
# take a peek (dare un'occhiata) at the first 5 rows of the data
head(dataset)

# summarize the class distribution
# Let's now take a look at the number of instances (rows) that belong to each
# class. We can view this as an absolute count and as a percentage.
percentage = prop.table(table(dataset$V9)) * 100
cbind(freq=table(dataset$V9), percentage=percentage)

# summarize attribute distributions
# Now we can take a look at a summary of each attribute.
# This includes the mean, the min and max values as well as some 
# percentiles (25th, 50th or media and 75th e.g. values at this points if 
# we ordered all the values for an attribute).
summary(dataset)

# Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate their accuracy
# on unseen data.
# Here is what we are going to cover in this step:
# 1.Set-up the test harness to use 10-fold cross validation.
# 2.build 5 different models to predict species from flower measurements
# 3.Select the best model.

# Test Harness
# We will use 10-fold crossvalidation to estimate accuracy.
# This will split our dataset into 10 parts, train in 9 and test on 1 
# and release for all combinations of train-test splits. We will also repeat
# the process 3 times for each algorithm with different splits of the data
# into 10 groups, in an effort to get a more accurate estimate.
#
# We are using the metric of "Accuracy" to evaluate models. This is a ratio
# of the number of correctly predicted instances in divided by the total
# number of instances in the dataset multiplied by 100 to give a percentage

# 3 repeats of 10-fold cross validation
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# Build Models
# We don't know which algorithms would be good on this problem or what
# configurations to use. We get an idea from the plots that some of the
# classes are partially linearly separable in some dimensions, so we are 
# expecting generally good results.
#
# Let's evaluate 5 different algorithms:
# 1.Linear Discriminant Analysis (LDA)
# 2.Classification and Regression Trees (CART).
# 3.k-Nearest Neighbors (kNN).
# 4.Support Vector Machines (SVM) with a linear kernel.
# 5.Random Forest (RF)
#
# This is a good mix of simple linear (LDA), nonlinear (CART, kNN) and
# complex nonlinear methods (SVM, RF). We reset the random number seed 
# before reach run to ensure that the evaluation of each algorithm is 
# performed using exactly the same data splits. 
# It ensures the results are directly comparable.
#
# basically set.seed() function will help to reuse the same set of random
# variables , which we may need in future to again evaluate particular
# task again with same random varibales.
# We just need to declare it before using any random numbers generating function.
#
# SVM
set.seed(7)
fit.svm <- train(V9~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(V9~., data=dataset, method="rf", metric=metric, trControl=control)
# Neural Network
set.seed(7)
fit.nnet <- train(V9~., data=dataset, method="nnet", metric=metric, trControl=control)
# Naive bayes
set.seed(7)
fit.nb <- train(V9~., data=dataset, method="nb", metric=metric, trControl=control)

# Select Best Model
# We now have 5 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.
# We can report on the accuracy of each model by first creating a list of 
# the created models and using the summary function.
#
# We can see the accuracy of each classifier and also other metrics like Kappa
list_models = list(svm=fit.svm, rf=fit.rf, nnet=fit.nnet, nb=fit.nb)
results = resamples(list_models)
summary(results)

# We can also create a plot of the model evaluation results and compare the
# spread and the mean accuracy of each model. There is a population of accuracy
# measures for each algorithm because each algorithm was evaluated 10 times 
# (10 fold cross validation).
dotplot(results)

# We get the model with best accurancy
maxAcc = 0
for(item in list_models){
  meanAcc = mean(item[["resample"]][["Accuracy"]])
  if(meanAcc>maxAcc){
    maxAcc=meanAcc
    bestModel=item
  }
}

# We can see that the most accurate model in this case was SVM
# The results for just the SVM model can be summarized
# summarize Best Model
print(bestModel)

# save the model to disk
saveRDS(bestModel, "./final_model.rds")

# later...

# load the model
super_model <- readRDS("./final_model.rds")
print(super_model)
  
# Make Predictions
# Now we want to get an idea of the
# accuracy of the model on our validation set.
#
# This will give us an independent final check on the accuracy of the best 
# model. It is valuable to keep a validation set just in case you made a 
# slip during such as overfitting to the training set or a data leak. 
# Both will result in an overly optimistic result.
#
# We can run the model directly on the validation set and summarize the
# results in a confusion matrix.
#
# We can see that the accuracy is 100%. 
# It was a small validation dataset (20%), but this result is within
# our expected margin of 97% +/-4% suggesting we may have an accurate
# and a reliably (affidabile) accurate model.  
predictions = predict(super_model, validation)
confusionMatrix(predictions, validation$V9)


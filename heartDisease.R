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
setwd("~/R/Machine Learning/ProgettoMachineLearning")
install.packages("caret")
library(caret)

# Load Data
# Attribute Information:
#
#3 (age)
#4 (sex)
#9 (cp)
#10 (trestbps)
#12 (chol)
#16 (fbs)
#19 (restecg)
#32 (thalach)
#38 (exang)
#40 (oldpeak)
#41 (slope)
#44 (ca)
#51 (thal)
#58 (num) (the predicted attribute)
#
# riga 288 aveva un valore ? cioe null
# na.strings = "NULL"
# convertire num in factor colClasses=c("num"="factor") per
# visualizzare il dataset con l'intenzione di non fare una multiclassificazione
# e non una regressione
# 
# usiamo na.action=na.omit per omettere le righe contenenti null
#
# Error in table "all arguments must have the same length"
# per ora eliminiamo tutti i valori null
#
dataset = 
  read.csv(
  url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"), 
  header = FALSE,
  na.strings = "?",
  colClasses=c("num"="factor"),
  #skipNul = TRUE,
  col.names = 
    c("age",
      "sex",
      "cp",
      "trestbps",
      "chol",
      "fbs",
      "restecg",
      "thalach",
      "exang",
      "oldpeak",
      "slope",
      "ca",
      "thal",
      "num"
    )
  )

# to remove nul
dataset = dataset[complete.cases(dataset), ]

# create a list of 80% of the rows in the original dataset for training
validation_index = createDataPartition(dataset$num, p=0.80, list=FALSE)
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
percentage = prop.table(table(dataset$num)) * 100
cbind(freq=table(dataset$num), percentage=percentage)

# summarize attribute distributions
# Now we can take a look at a summary of each attribute.
# This includes the mean, the min and max values as well as some 
# percentiles (25th, 50th or media and 75th e.g. values at this points if 
# we ordered all the values for an attribute).
summary(dataset)

# We are going to look at two types of plots:
# 1.Univariate plots to better understand each attribute.
# split input and output
x <- dataset[,1:13]
y <- dataset[,14]
# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(dataset)[i])
}

# barplot for class breakdown (generally uninteresting in this 
# case because they're even).
plot(y)

# density plots for each attribute by class value
# Like the boxplots, we can see the difference in distribution of each 
# attribute by class value. We can also see the Gaussian-like distribution 
# (bell curve) of each attribute.
scales = list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

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
# a) linear algorithms
set.seed(7)
fit.lda = train(num~., data=dataset, method="lda", metric=metric, trControl=control, na.action=na.omit)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(num~., data=dataset, method="rpart", metric=metric, trControl=control, na.action=na.omit)
# kNN
set.seed(7)
fit.knn <- train(num~., data=dataset, method="knn", metric=metric, trControl=control, na.action=na.omit)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(num~., data=dataset, method="svmRadial", metric=metric, trControl=control, na.action=na.omit)
# Random Forest
set.seed(7)
fit.rf <- train(num~., data=dataset, method="rf", metric=metric, trControl=control, na.action=na.omit)

# Select Best Model
# We now have 5 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.
# We can report on the accuracy of each model by first creating a list of 
# the created models and using the summary function.
#
# We can see the accuracy of each classifier and also other metrics like Kappa
results = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# We can also create a plot of the model evaluation results and compare the
# spread and the mean accuracy of each model. There is a population of accuracy
# measures for each algorithm because each algorithm was evaluated 10 times 
# (10 fold cross validation).
dotplot(results)

# We can see that the most accurate model in this case was SVM
# The results for just the SVM model can be summarized
# summarize Best Model
print(fit.svm)
  
# Make Predictions
# The SVM was the most accurate model. Now we want to get an idea of the
# accuracy of the model on our validation set.
#
# This will give us an independent final check on the accuracy of the best 
# model. It is valuable to keep a validation set just in case you made a 
# slip during such as overfitting to the training set or a data leak. 
# Both will result in an overly optimistic result.
#
# We can run the SVM model directly on the validation set and summarize the
# results in a confusion matrix.
#
# We can see that the accuracy is 100%. 
# It was a small validation dataset (20%), but this result is within
# our expected margin of 97% +/-4% suggesting we may have an accurate
# and a reliably (affidabile) accurate model.  
predictions = predict(fit.svm, validation)
confusionMatrix(predictions, validation$num)


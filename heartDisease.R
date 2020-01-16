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
setwd("~/R/Machine Learning/heartdisease")

install.packages("caret")
install.packages("FactoMineR")
install.packages("factoextra") 
install.packages("corrplot")
install.packages("pROC") 
library(gridExtra)
library(caret)
library(FactoMineR) 
library(factoextra)
library(corrplot)
library(pROC)

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
dataset = 
  read.csv(
    "14.csv",
    header = FALSE,
    na.strings = "?",
    fileEncoding="UTF-8-BOM",
    sep=";",
    dec=",",
    col.names = c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target")
  )

# To remove rows with null attribute values
dataset = dataset[complete.cases(dataset), ]

# Factor for attributeS
dataset$target = ifelse(dataset$target>0, "YES", "NO")
dataset$target = factor(dataset$target)

#dataset$sex = ifelse(dataset$sex==0,'FEMALE','MALE')
#dataset$sex = factor(dataset$sex)
#dataset$exang = ifelse(dataset$exang==1,'YES','NO')
#dataset$exang = factor(dataset$exang)
#dataset$cp = ifelse(dataset$cp == 1, "ATYPICAL ANGINA", ifelse(dataset$cp == 2, "NON-ANGINAL PAIN", "ASYMPTOMATIC"))
#dataset$cp = factor(dataset$cp)
#dataset$restecg = ifelse(dataset$restecg == 0, "NORMAL", ifelse(dataset$restecg == 1, "ABNORMALITY", "PROBABLE OR DEFINIT"))
#dataset$restecg = factor(dataset$restecg)


# Names of attributes
names(dataset)
# Dimensions of dataset
dim(dataset)
# List types for each attribute
sapply(dataset, class)
# Take a peek (dare un'occhiata) at the first 5 rows of the data
head(dataset)

# PCA
# Target distribution
ggplot(dataset, aes(x=dataset$target, fill=dataset$target)) + 
  geom_bar() +
  xlab("Presenza di cardiopatia") +
  ylab("Num. di casi") +
  ggtitle("Distribuzione del target") +
  scale_fill_discrete(name = "Cardiopatia", labels = c("NO", "YES"))

# If issue overlap plot -> dev.off()

# Age distribution
ggplot(dataset, aes(x=dataset$sex, fill=dataset$sex)) + 
  geom_bar() +
  xlab("Presenza di cardiopatia") +
  ylab("Num. di casi") +
  ggtitle("Distribuzione del sex") +
  scale_fill_discrete(name = "Cardiopatia", labels = c("FEMALE", "MALE"))


hist(dataset$age, main="Età del paziente", xlab = "Anni") 

# Correlation
corr = cor(dataset[,1:13])
corrplot(corr,type="lower",title = "correlation of variable",tl.col=1,tl.cex=0.7)

# PCA
#divido il subset individuando un numero di righe e colonne "attive" nella PCA
#e altre righe + colonne individueranno degli individui supplementari che saranno predetti dalla PCA
dataset.active = dataset[, 1:13]

#le variabili sono scalate, sopratutto se sono misurate in scale diverse
#la funziona PCA() le standardizza automaticamente
pca <- PCA(dataset.active, scale.unit = TRUE, ncp = 7, graph = TRUE) #ncp è il numero di dimensioni finali

#interpretazione della PCA
#gli autovalori misurano la quantità di variazione mantenuta da ogni componente principale (sono piÃ¹ grandi per i primi)
#I primi PC corrispondono alle direzioni con la massima quantità di variazione nel dataset
#Esaminiamo gli autovalori per determinare il numero di PC da considerare (autovalori e proporzione di varianza, ossia informazioni contenute)
eig.val = get_eigenvalue(pca)
#osserviamo che il 55% delle variazioni sono spiegate dai primi 4 autovalori
eig.val 

#Un autovalore > 1 indica che il PC rappresenta una varianza maggiore rispetto a una delle variabili originali 
#nei dati standardizzati. Questo Ã¨ comunemente usato come punto di interruzione per il quale i PC vengono conservati
#Visualizzo graficamente gli autovalori
fviz_eig(pca, addlabels = TRUE, ylim = c(0,25))

#Estraggo i risultati - VARIABILI
var = get_pca_var(pca)

#la correlazione fra una variabile e un PCA¨ usata come coordinata della variabile sulla PC
head(var$coord, 5)

fviz_pca_var(pca, col.var = "red")

corrplot(var$contrib, is.corr=FALSE)   
p1 <- fviz_contrib(pca, choice="var", axes=1, fill="pink", color="grey", top=10)
p2 <- fviz_contrib(pca, choice="var", axes=2, fill="skyblue", color="grey", top=10)
p3 <- fviz_contrib(pca, choice="var", axes=3, fill="pink", color="grey", top=10)
p4 <- fviz_contrib(pca, choice="var", axes=4, fill="skyblue", color="grey", top=10)
p5 <- fviz_contrib(pca, choice="var", axes=5, fill="pink", color="grey", top=10)
p6 <- fviz_contrib(pca, choice="var", axes=6, fill="skyblue", color="grey", top=10)
p7 <- fviz_contrib(pca, choice="var", axes=7, fill="pink", color="grey", top=10)

grid.arrange(p1,p2,p3,p4,p5,p6,p7,ncol=4)

#le variabili correlate positivamente sono raggruppate insieme 
#(fbs, chol, trestbps, restecg, age, ca)
#(sex, thal, exang, cp, slope, oldpeak)

#le variabili correlate negativamente sono posizionate in quadranti opposti
#slope e thalach? o thalach e tutte quelle nel quadrante di slope?

#la distanza fra variabile e origine misura la qualitÃ  delle variabili (piÃ¹ sono lontante, meglio sono rappresentate)
#es. thal 

#INDIVIDUALS
#coordinate, correlazione fra individuals e assi, cos2 e contribuzione
ind = get_pca_ind(pca)
ind
fviz_pca_ind(pca, col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

############################################
# create a list of 80% of the rows in the original dataset for training
validation_index = createDataPartition(dataset$target, p=0.80, list=FALSE)
# select 20% of the data for validation
testset = dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
trainset = dataset[validation_index,]

# names of attributes
names(trainset)
# dimensions of dataset
dim(trainset)
# list types for each attribute
sapply(trainset, class)
# take a peek (dare un'occhiata) at the first 5 rows of the data
head(trainset)

# summarize the class distribution
# Let's now take a look at the number of instances (rows) that belong to each
# class. We can view this as an absolute count and as a percentage.
percentage = prop.table(table(trainset$target)) * 100
cbind(freq=table(trainset$target), percentage=percentage)

# summarize attribute distributions
# Now we can take a look at a summary of each attribute.
# This includes the mean, the min and max values as well as some 
# percentiles (25th, 50th or media and 75th e.g. values at this points if 
# we ordered all the values for an attribute).
summary(trainset)

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
control = trainControl(method="repeatedcv", number=10, repeats=3)
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
# Random Forest
# set.seed(7)
# fit.rf <- train(num~., data=trainset, method="rf", metric=metric, trControl=control)

# SVM
set.seed(7)
fit.svm <- train(target~., data=trainset, method="svmRadial", metric=metric, trControl=control)

# Neural Network
set.seed(7)
fit.nnet <- train(target~., data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Select Best Model
# We now have 5 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.
# We can report on the accuracy of each model by first creating a list of 
# the created models and using the summary function.
#
# We can see the accuracy of each classifier and also other metrics like Kappa
list_models = list(svm=fit.svm, nnet=fit.nnet)
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
predictions = predict(super_model, testset)
confusionMatrix(predictions, testset$target)

#####################################################
#####################################################
control = trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
metric = "ROC"

# SVM
set.seed(7)
fit.svm <- train(target~., data=trainset, method="svmRadial", metric=metric, trControl=control)

# Neural Network
set.seed(7)
fit.nnet <- train(target~., data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Make Predictions
svm.probs = predict(fit.svm, testset, type = "prob")
nnet.probs = predict(fit.nnet, testset, type = "prob")

# Generate the ROC curve of each model, and plot the curve on the same figure
svm.ROC = roc(testset$target, svm.probs$YES, levels=levels(testset$target), direction = "<")
plot(svm.ROC, print.thres="best", col="orange")

nnet.ROC = roc(testset$target, nnet.probs$YES, levels=levels(testset$target), direction = "<")
plot(nnet.ROC, print.thres="best", add=TRUE, col="red")

# To compare the AUC
svm.ROC
nnet.ROC 

# We can also compare the statistics of the generated performance measure
cv.values = resamples(list(svm=fit.svm, nnet = fit.nnet)) 
summary(cv.values) 

# Use dotplot to plot the results in the ROC metric
dotplot(cv.values, metric = "ROC") 

# Or the bwplot 
bwplot(cv.values, layout = c(3, 1)) 

# Or the splom plot
splom(cv.values,metric="ROC") 

# We can also take into account the timings required for training the models
cv.values$timings


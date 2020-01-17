#########################################
#
# Breast Cancer Wisconsin
# Machine Learning
# Project
# Bettini Ivo Junior 806878
# Cocca Umberto 807191
# Traversa Silvia 816435
#
#########################################

# Set workspace path
setwd("~/R/Machine Learning/heartdisease")

#Install packages and library
install.packages(c("caret","FactoMineR","factoextra","corrplot","pROC"))
library(gridExtra)
library(caret)
library(FactoMineR) 
library(factoextra)
library(corrplot)
library(pROC)
library(GGally)

#Dataset
dataset = read.csv("data.csv",
                    header = TRUE,
                    na.strings = "?",
                    colClasses=c("diagnosis"="factor"),
                    stringsAsFactors=F)
# Remove ID
dataset = dataset[,-1]
# Control presence of null value
na_count <- sapply(dataset, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)

# Names of attributes
names(dataset)
# Dimension of dataset
dim(dataset)
# List types for each attribute
sapply(dataset, class)
# Take a peek at the first 5 rows of the data
head(dataset)
summary(dataset)

# Target distribution
ggplot(dataset, aes(x=dataset$diagnosis, fill=dataset$diagnosis)) + 
  geom_bar() +
  xlab("Malignant or Benign diagnosis") + 
  ylab("Num. of cases") +
  ggtitle("Target distribution") +
  scale_fill_manual(name = "B/M", labels = c("B", "M"), values = c("#ee8572", "#63b7af")) +
  theme_light()


# Correlation
corr = cor(dataset[,2:31])
corrplot(corr,type="lower",title = "Correlation of variables",tl.col=1,tl.cex=0.7)

ggpairs(dataset, columns = 1:10, title = "titolo",  
        axisLabels = "show", columnLabels = colnames(dataset[,1:10]))

# PCA 
dataset.active = dataset[, 2:31] #rimuovo il target
pca <- PCA(dataset.active, scale.unit = TRUE, ncp = 10, graph = TRUE)

# Examinate the PCA's result with the eigenvalues
#
# Eigenvalues
eig.val = get_eigenvalue(pca)
eig.val

# Graphical representation of eigenvalues
fviz_eig(pca, addlabels = TRUE, ylim = c(0,50))

# Variables extraction
var = get_pca_var(pca)
head(var$coord, 30)

# Graphical representation of PCA's variables
fviz_pca_var(pca, col.var = "red")

# Dimension's study
p1 <- fviz_contrib(pca, choice="var", axes=1, fill="pink", color="grey", top=10)
p2 <- fviz_contrib(pca, choice="var", axes=2, fill="skyblue", color="grey", top=10)
p3 <- fviz_contrib(pca, choice="var", axes=3, fill="pink", color="grey", top=10)
p4 <- fviz_contrib(pca, choice="var", axes=4, fill="skyblue", color="grey", top=10)
p5 <- fviz_contrib(pca, choice="var", axes=5, fill="pink", color="grey", top=10)
p6 <- fviz_contrib(pca, choice="var", axes=6, fill="skyblue", color="grey", top=10)
p7 <- fviz_contrib(pca, choice="var", axes=7, fill="pink", color="grey", top=10)
p8 <- fviz_contrib(pca, choice="var", axes=8, fill="skyblue", color="grey", top=10)
p9 <- fviz_contrib(pca, choice="var", axes=9, fill="pink", color="grey", top=10)
p10 <- fviz_contrib(pca, choice="var", axes=10, fill="skyblue", color="grey", top=10)

grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,ncol=5)

# Individuals
ind = get_pca_ind(pca)
ind
fviz_pca_ind(pca, col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

# Create a list of 75% of the rows in the original dataset for training
index <- createDataPartition(dataset$diagnosis, p=0.75, list=FALSE)
# Select 25% of the data for validation
trainset <- dataset[ index,]
# Use the remaining 75% of data to training and testing the models
testset <- dataset[-index,]

# Names of attributes
names(trainset)
# Dimensions of dataset
dim(trainset)
# List types for each attribute
sapply(trainset, class)
# Take a peek (dare un'occhiata) at the first 5 rows of the data
head(trainset)

# Summarize the class distribution
percentage = prop.table(table(trainset$diagnosis)) * 100
cbind(freq=table(trainset$target), percentage=percentage)

# Summarize attribute distributions
summary(trainset)

# 10-fold cross validation with 3 repeats
control = trainControl(method="repeatedcv", number=10, repeats = 3)
metric = "Accuracy"

# Model applicated only on 10 attributes choosen after PCA
subAttr = diagnosis~concave.points_mean+fractal_dimension_mean+texture_se+texture_worst+
  smoothness_mean+symmetry_worst+fractal_dimension_worst+smoothness_se+concavity_se+symmetry_mean

# SVM
set.seed(7)
fit.svm <- train(subAttr, data=trainset, method="svmRadial", metric=metric, trControl=control)
# Neural networks
set.seed(7)
fit.nnet <- train(subAttr, data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Accurancy
list_models = list(svm=fit.svm, nnet=fit.nnet)
results = resamples(list_models)
summary(results)

# Plot of the model evaluation results and compare the
# spread and the mean accuracy of each model.
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

# Summarize Best Model
print(bestModel)

# Save the model to disk
saveRDS(bestModel, "./final_model.rds")

# later...

# Load the model
super_model <- readRDS("./final_model.rds")
print(super_model)

# Predictions (accuracy of the model on our validation set)
predictions = predict(super_model, testset)
confusionMatrix(predictions, testset$diagnosis)

#Set up the training control
control = trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
metric = "ROC"

# SVM
set.seed(7)
fit.svm <- train(subAttr, data=trainset, method="svmRadial", metric=metric, trControl=control)

# Neural Network
set.seed(7)
fit.nnet <- train(subAttr, data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Make Predictions
svm.probs = predict(fit.svm, testset, type = "prob")
nnet.probs = predict(fit.nnet, testset, type = "prob")

# Generate the ROC curve of each model, and plot the curve on the same figure
svm.ROC = roc(testset$diagnosis, svm.probs$B, levels=levels(testset$diagnosis), direction = ">")
plot(svm.ROC, print.thres="best", col="orange")

nnet.ROC = roc(testset$diagnosis, nnet.probs$B, levels=levels(testset$diagnosis), direction = ">")
plot(nnet.ROC, print.thres="best", add=TRUE, col="red")

# compare the AUC
svm.ROC
nnet.ROC 

# comparison of the statistics of the generated performance measure
cv.values = resamples(list(svm=fit.svm, nnet = fit.nnet)) 
summary(cv.values) 

# dotplot 
dotplot(cv.values, metric = "ROC") 

# bwplot 
bwplot(cv.values, layout = c(3, 1)) 

# splom plot
splom(cv.values,metric="ROC") 

# timings required for training the models
cv.values$timings

# Precision, Recall an F1
realvalues = testset[,1]
classLabel <- confusionMatrix(predictions, realvalues, mode="prec_recall", positive = "B")
classLabel
classLabel$byClass["Precision"]
classLabel$byClass["Recall"]
classLabel$byClass["F1"]

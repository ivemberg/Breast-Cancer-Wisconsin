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
setwd("~/git/heartdisease")

#Install packages and library
install.packages(c("caret","FactoMineR","factoextra","corrplot","pROC","funModeling", "gridExtra"))
library(caret)
library(FactoMineR) 
library(factoextra)
library(corrplot)
library(pROC)
library(funModeling)
library(gridExtra)

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
na_count # no null values

# Exploration of the dataset
names(dataset)
dim(dataset)
sapply(dataset, class)
head(dataset)
summary(dataset)

# Create Train (75% of the original rows) and Test (remaining 25% of the original rows) set
index <- createDataPartition(dataset$diagnosis, p=0.75, list=FALSE)
trainset <- dataset[ index,]
testset <- dataset[-index,]

# Exploration of the train set
names(trainset)
dim(trainset)
sapply(trainset, class)
head(trainset)
summary(trainset)

# Density plot of all variables
plotar(data = dataset, target = "diagnosis", plot_type = "histdens", path_out = "./variablesgraphs.png")

# Summary of the class distribution
percentage = prop.table(table(trainset$diagnosis)) * 100
cbind(freq=table(trainset$target), percentage=percentage)

# Graphs

ggplot(trainset, aes(x=trainset$diagnosis, fill=trainset$diagnosis)) + 
  geom_bar() +
  xlab("Malignant or Benign diagnosis") + 
  ylab("Num. of cases") +
  ggtitle("Target distribution") +
  scale_fill_manual(name = "B/M", labels = c("B", "M"), values = c("#60B267", "#CE1824")) +
  theme_light()


# Correlation
corr = cor(trainset[,2:31])
corrplot(corr,type="lower",tl.col=1,tl.cex=0.7)

# Example of correlated variables (high redundancy)
ggplot(trainset, aes(x = trainset$radius_mean, y = trainset$perimeter_mean, color = trainset$diagnosis)) +
  geom_point() + 
  xlab("Radius mean") +
  ylab("Perimeter mean") +
  scale_color_manual(name = "Diagnosis", values=c("#60B267", "#CE1824")) +
  theme_light()

# Example of uncorrelated variables (low redundancy)
ggplot(trainset, aes(x = trainset$fractal_dimension_se, y = trainset$area_worst, color = trainset$diagnosis)) +
  geom_point() + 
  xlab("Fractal dimension standard error") +
  ylab("Area worst") +
  scale_color_manual(name = "Diagnosis", values=c("#60B267", "#CE1824")) +
  theme_light()

# PCA 
trainset.pca = trainset[,2:31] 
pca <- PCA(trainset.pca, scale.unit = TRUE, ncp = 10, graph = TRUE) 

# Examinate the PCA's result with the eigenvalues
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
p1 <- fviz_contrib(pca, choice="var", axes=1, fill="#6BB7C6", color="grey", top=10)
p2 <- fviz_contrib(pca, choice="var", axes=2, fill="#6BB7C6", color="grey", top=10)
p3 <- fviz_contrib(pca, choice="var", axes=3, fill="#6BB7C6", color="grey", top=10)
p4 <- fviz_contrib(pca, choice="var", axes=4, fill="#6BB7C6", color="grey", top=10)
p5 <- fviz_contrib(pca, choice="var", axes=5, fill="#6BB7C6", color="grey", top=10)
p6 <- fviz_contrib(pca, choice="var", axes=6, fill="#6BB7C6", color="grey", top=10)
p7 <- fviz_contrib(pca, choice="var", axes=7, fill="#6BB7C6", color="grey", top=10)
p8 <- fviz_contrib(pca, choice="var", axes=8, fill="#6BB7C6", color="grey", top=10)
p9 <- fviz_contrib(pca, choice="var", axes=9, fill="#6BB7C6", color="grey", top=10)
p10 <- fviz_contrib(pca, choice="var", axes=10, fill="#6BB7C6", color="grey", top=10)

grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,ncol=5)



# Individuals
ind = get_pca_ind(pca)
fviz_pca_ind(pca, col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

# Model applicated only on 10 attributes choosen after PCA
subTrain = diagnosis~concave.points_mean+fractal_dimension_mean+texture_se+texture_worst+
  smoothness_mean+symmetry_worst+fractal_dimension_worst+smoothness_se+concavity_se+symmetry_mean


# 10-fold cross validation with 3 repeats
control = trainControl(method="repeatedcv", number=10, repeats = 3)
metric = "Accuracy"

# SVM
set.seed(7)
fit.svm <- train(subTrain, data=trainset, method="svmRadial", metric=metric, trControl=control)
# Neural networks
set.seed(7)
fit.nnet <- train(subTrain, data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Accuracy
list_models = list(svm=fit.svm, nnet=fit.nnet)
results = resamples(list_models)
summary(results)

accuracies <- data.frame(fit.svm$results$Accuracy, fit.nnet$results$Accuracy)
names(accuracies) <- c(paste("SVM accuracy"), paste("NNET accuracy"))
boxplot(accuracies,col="#BBE1FA",ylab="value" )

# Predictions SVM (accuracy of the model on our validation set)
predictions_svm = predict(fit.svm, testset)

# Predictions NNET (accuracy of the model on our validation set)
predictions_nnet = predict(fit.nnet, testset)

# Take the model with the best accurancy
maxAcc = 0
for(item in list_models){
  meanAcc = mean(item[["resample"]][["Accuracy"]])
  if(meanAcc>maxAcc){
    maxAcc=meanAcc
    finalModel=item
  }
}

# Summary of the best model
print(finalModel)

# Save the model to disk
saveRDS(finalModel, "./finalModel.rds")

# later...

# Load the model
bestModel <- readRDS("./finalModel.rds")
print(bestModel)

# Predictions (accuracy of the model on our validation set)
predictions = predict(bestModel, testset)
cm = confusionMatrix(predictions, testset$diagnosis)


#Set up the training control
control = trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
metric = "ROC"

# SVM
set.seed(7)
fit.svm <- train(subTrain, data=trainset, method="svmRadial", metric=metric, trControl=control)

# Neural Network
set.seed(7)
fit.nnet <- train(subTrain, data=trainset, method="nnet", metric=metric, trControl=control, trace=FALSE)

# Make Predictions
svm.probs = predict(fit.svm, testset, type = "prob")
nnet.probs = predict(fit.nnet, testset, type = "prob")

# Generate the ROC curve of each model, and plot the curve on the same figure
svm.ROC = roc(testset$diagnosis, svm.probs$B, levels=levels(testset$diagnosis), direction = ">")
plot(svm.ROC, print.thres="best", col="orange")

nnet.ROC = roc(testset$diagnosis, nnet.probs$B, levels=levels(testset$diagnosis), direction = ">")
plot(nnet.ROC, print.thres="best", col="red")

# comparison of the AUC
svm.ROC
nnet.ROC 

# comparison of the statistics of the generated performance measure
cv.values = resamples(list(svm=fit.svm, nnet = fit.nnet)) 
summary(cv.values) 

# Plots
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1)) 
splom(cv.values,metric="ROC") 

# timings required for training the models
cv.values$timings

# Precision, Recall an F1 for SVM model
realvalues = testset[,1]
classLabelSVM <- confusionMatrix(predictions_svm, realvalues, mode="prec_recall", positive = "B")
classLabelSVM
classLabelSVM$byClass["Precision"]
classLabelSVM$byClass["Recall"]
classLabelSVM$byClass["F1"]

# Precision, Recall an F1 for NNET model
classLabelNNET <- confusionMatrix(predictions_nnet, realvalues, mode="prec_recall", positive = "B")
classLabelNNET
classLabelNNET$byClass["Precision"]
classLabelNNET$byClass["Recall"]
classLabelNNET$byClass["F1"]

#Save plots

plots.png.detials <- file.info(plots.png.paths)
plots.png.detials <- plots.png.detials[order(plots.png.detials$mtime),]
sorted.png.names <- gsub(plots.dir.path, "C:/Users/strav/git/heartdisease", row.names(plots.png.detials), fixed=TRUE)
numbered.png.names <- paste0("C:/Users/strav/git/heartdisease", 1:length(sorted.png.names), ".png")

# Rename all the .png files as: 1.png, 2.png, 3.png, and so on.
file.rename(from=sorted.png.names, to=numbered.png.names)

file.copy(from=plots.png.paths, to="C:/Users/strav/git/heartdisease/grafici")

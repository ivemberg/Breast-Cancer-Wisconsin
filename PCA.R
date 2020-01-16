setwd("~/R/Machine Learning/heartdisease")
install.packages("caret")
install.packages(c("FactoMineR", "factoextra")) 
install.packages("C50") 
library("FactoMineR") 
library("factoextra")
library("caret")
library("corrplot")
library("gridExtra")
library("C50") 

dataset = 
  read.csv(
    url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"), 
    header = FALSE,
    na.strings = "?",
    col.names = c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target")
  )

# factor for target attribute
dataset$target[dataset$target > 0] = 1
dataset$target = factor(dataset$target)
# to remove rows with null attribute values 
dataset = dataset[complete.cases(dataset), ]

# list types for each attribute
sapply(dataset, class)
#########################################
#Feature selection
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'target'
predictors<-names(dataset)[!names(dataset) %in% outcomeName]
predProfile <- rfe(dataset[,predictors], dataset[,outcomeName],
                   rfeControl = control)
predProfile
##############################################

summary(dataset)

#distribuzione del target
ggplot(dataset, aes(x=dataset$target, fill=dataset$target)) + 
  geom_bar() +
  xlab("Presenza di cardiopatia") +
  ylab("Num. di casi") +
  ggtitle("Distribuzione del target") +
  scale_fill_discrete(name = "Cardiopatia", labels = c("0", "1","2","3","4","5"))

hist(dataset$age, main="Et� del paziente", xlab = "Anni") 
hist(dataset$sex, main="Sesso del paziente", xlab = "Sesso") #meglio grafico diverso

# correlazione
corr = cor(dataset[,1:13])
corrplot(corr,type="lower",title = "correlation of variable",tl.col=1,tl.cex=0.7)

#PCA secondo lab della prof
#divido il subset individuando un numero di righe e colonne "attive" nella PCA
#e altre righe + colonne individueranno degli individui supplementari che saranno predetti dalla PCA

dataset.active = dataset[, 1:13] #valori presi a caso, proviamo
head(dataset.active[,1:13],5)
#le variabili sono scalate, sopratutto se sono misurate in scale diverse
#la funziona PCA() le standardizza automaticamente
pca <- PCA(dataset.active, scale.unit = TRUE, ncp = 7, graph = TRUE) #ncp è il numero di dimensioni finali
contrib = pca$var$contrib
#interpretazione della PCA
#gli autovalori misurano la quantità di variazione mantenuta da ogni componente principale (sono più grandi per i primi)
#I primi PC corrispondono alle direzioni con la massima quantità di variazione nel dataset
#Esaminiamo gli autovalori per determinare il numero di PC da considerare (autovalori e proporzione di varianza, ossia informazioni contenute)
eig.val = get_eigenvalue(pca)
eig.val #osserviamo che il 55% delle variazioni sono spiegate dai primi 4 autovalori
#Un autovalore > 1 indica che il PC rappresenta una varianza maggiore rispetto a una delle variabili originali 
#nei dati standardizzati. Questo è comunemente usato come punto di interruzione per il quale i PC vengono conservati
#Visualizzo graficamente gli autovalori
fviz_eig(pca, addlabels = TRUE, ylim = c(0,25))

#Estraggo i risultati - VARIABILI
var = get_pca_var(pca)
#la correlazione fra una variabile e un PC è usata come coordinata della variabile sulla PC
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

#la distanza fra variabile e origine misura la qualità delle variabili (più sono lontante, meglio sono rappresentate)
#es. thal 

#INDIVIDUALS
#coordinate, correlazione fra individuals e assi, cos2 e contribuzione
ind = get_pca_ind(pca)
ind
fviz_pca_ind(pca, col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

#Creo training, provo i modelli sul training normale e poi su un subset di training dove considero solo le prime 7 dimensioni per avere varianza 75%
index <- createDataPartition(dataset$target, p=0.75, list=FALSE)
trainset <- dataset[ index,]
testset <- dataset[-index,]

#Provo i modelli 
control = trainControl(method="cv", number=10)
metric = "Accuracy"
#SVM
set.seed(7)
fit.svm <- train(target~., data=trainset, method="svmRadial", metric=metric, trControl=control)
#Reti neurali
set.seed(7)
fit.nnet <- train(target~., data=trainset, method="nnet", metric=metric, trControl=control)

fit.svm

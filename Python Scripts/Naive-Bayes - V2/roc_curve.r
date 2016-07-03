library(ROCR)

#LOAD RESULT FILES
data_fold1 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/ID3/fold1.csv", sep=",", head=T)
data_fold2 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold2.csv", sep=",", head=T)
data_fold3 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold3.csv", sep=",", head=T)
data_fold4 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold4.csv", sep=",", head=T)
data_fold5 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold5.csv", sep=",", head=T)
data_fold6 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold6.csv", sep=",", head=T)
data_fold7 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold7.csv", sep=",", head=T)
data_fold8 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold8.csv", sep=",", head=T)
data_fold9 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold9.csv", sep=",", head=T)
data_fold10 <- read.table("/home/pierre/Drive/Artigo-Conferência/Dados/Naive-Bayes/Resultados-FOLDS/fold10.csv", sep=",", head=T)

#CREATE PREDICITION LIST
#predictions = list(data_fold1$PREDICT.RESULT, data_fold2$PREDICT.RESULT, data_fold3$PREDICT.RESULT, data_fold4$PREDICT.RESULT, data_fold5$PREDICT.RESULT, data_fold6$PREDICT.RESULT, data_fold7$PREDICT.RESULT, data_fold8$PREDICT.RESULT, data_fold9$PREDICT.RESULT, data_fold10$PREDICT.RESULT) 
predictions = list(data_fold1$PREDICT.TRUE, data_fold2$PREDICT.TRUE, data_fold3$PREDICT.TRUE, data_fold4$PREDICT.TRUE, data_fold5$PREDICT.TRUE, data_fold6$PREDICT.TRUE, data_fold7$PREDICT.TRUE, data_fold8$PREDICT.TRUE, data_fold9$PREDICT.TRUE, data_fold10$PREDICT.TRUE) 

#CREATE LABEL LIST
labels = list(data_fold1$REAL.LABEL, data_fold2$REAL.LABEL, data_fold3$REAL.LABEL, data_fold4$REAL.LABEL, data_fold5$REAL.LABEL, data_fold6$REAL.LABEL, data_fold7$REAL.LABEL, data_fold8$REAL.LABEL, data_fold9$REAL.LABEL, data_fold10$REAL.LABEL)

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")

plot(perf,col="black",lty=2)

plot(perf,lwd=3,avg="horizontal",add=TRUE)

lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)

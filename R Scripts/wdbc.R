library("e1071")

#Breast Cancer Wisconsin (Diagnostic) Data Set
dataset <- read.csv("wdbc.data", header=F, sep=",")

index <- 1:nrow(dataset)
testindex <- sample(index, trunc(length(index)*30/100))
testset <- dataset[testindex,]
trainset <- dataset[-testindex,]

names(dataset)

tuned<-tune.svm(V2~., data = trainset, gamma = 10^(-6:-1), cost = 10^(-1:1))

model<-svm(V2~., data = trainset, kernel = "radial", gamma = 0.01, cost = 10)

prediction<-predict(model, testset[,-2])

tab<-table(pred = prediction, true = testset[,2])

classAgreement(tab)

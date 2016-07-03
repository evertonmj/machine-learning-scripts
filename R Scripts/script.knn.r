euclidean<-function(v1, v2){
  sum<-sqrt(sum((v1-v2)^2))
    sum
}

knn.class<-function(train, test, k=1){
  distances<-c()
    cols<-ncol(train)#rotulo
    for(i in 1:nrow(train)){
      distances[i]<-euclidean(train[i,1:(cols-1)], test[1:(cols-1)])
    }

  resp<-list()
    knearestn<-sort.list(distances, decreasing=F)[1:k]
    probabilidades<-table(train[knearestn,cols])
    probabilidades<-probabilidades/sum(probabilidades)
    resp$class<-unique(train[knearestn,cols])
    resp$probabilidades<-probabilidades

    resp

}

knn.reg<-function(train, test, k){
  distances<-c()
    cols<-ncol(train)#rotulo
    for(i in 1:nrow(train)){
      distances[i]<-euclidean(train[i,1:(cols-1)], test[1:(cols-1)])
    }

  knearestn<-sort.list(distances, decreasing=F)[1:k]

    resp<-mean(train[knearestn,cols])
    resp
}

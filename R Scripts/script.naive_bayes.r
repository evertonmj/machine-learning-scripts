naive.bayes<-function(data, class, test){
  niveis<-levels(class)
    p.class<-c()
    tamanho<-length(class)
    for(i in 1:length(niveis)){
      p.class[i]<-length(which(class==niveis[i]))/tamanho
        for(j in 1:ncol(data)){
          
          	index<-which(class==niveis[i])
            result<-table(data[index,j]) / length(index)
            
            result<-result[which(names(result)==test[j])]
            
            p.class[i]<-p.class[i]*result
        }

    }

  resultado<-p.class/sum(p.class)
    names(resultado)<-niveis
    resultado
}

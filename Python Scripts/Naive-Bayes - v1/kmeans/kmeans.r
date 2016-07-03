euclidean.distance<-function(x, y){
	resp<-sum((x-y)^2)
	resp
}

closest.k<-function(data, centroids){
	euclidean.dist<-c()
	for(i in 1:nrow(centroids)){
		euclidean.dist[i]<-euclidean.distance(data, centroids[i,])
	}

	which.min(euclidean.dist)
}

k.means<-function(dataset, k=2){
	resp<-list()
	cat("Inicializando centroides...\n");
	index<-sample(1:nrow(dataset), k)
	centroids<-dataset[index, ]

	clustering<-rep(0,nrow(dataset))
	stop.crit<-matrix(0, nrow=nrow(centroids), ncol=ncol(centroids))

	while(euclidean.distance(stop.crit, centroids) > 0){

		#organiza os objetos em clusters iniciais
		cat("Centroides escolhidos:\n")
		for(i in 1:k){
			cat("[",i,"] = ", as.double(centroids[i,]), "\n")
		}

		for(i in 1:nrow(dataset)){
			clustering[i]<-closest.k(dataset[i,], centroids)
		}
		stop.crit<-centroids

		resp$data<-dataset
		resp$centroids<-centroids
		resp$clustering<-clustering

		#atualiza centroide...
		for(i in 1:nrow(centroids)){
			centroids[i,]<-colMeans(dataset[which(clustering == i),])
		}
	}


	resp
}


plot.kmeans<-function(resp){
	plot(resp$data)
	for(i in sort(unique(resp$clustering))){
		points(resp$data[which(resp$clustering==i),], col=(i+1))
		points(resp$centroids[i,], pch=19, col=(i+1))
	}
}


k.means.interativo<-function(dataset, k=2, sleep.time=1){
	resp<-list()
	cat("Inicializando centroides...\n");
	Sys.sleep(sleep.time)
	index<-sample(1:nrow(dataset), k)
	centroids<-dataset[index, ]

	clustering<-rep(0,nrow(dataset))
	stop.crit<-matrix(0, nrow=nrow(centroids), ncol=ncol(centroids))

	while(euclidean.distance(stop.crit, centroids) > 0){

		#organiza os objetos em clusters iniciais
		cat("Centroides escolhidos:\n")
		for(i in 1:k){
			cat("[",i,"] = ", as.double(centroids[i,]), "\n")
		}

		for(i in 1:nrow(dataset)){
			clustering[i]<-closest.k(dataset[i,], centroids)
		}
		stop.crit<-centroids

		resp$data<-dataset
		resp$centroids<-centroids
		resp$clustering<-clustering

		plot.kmeans(resp)
		Sys.sleep(sleep.time)

		#atualiza centroide...
		for(i in 1:nrow(centroids)){
			centroids[i,]<-colMeans(dataset[which(clustering == i),])
		}
	}


	resp
}

#redes neurais em R
#funcao de ativacao
ativacao <- function(entrada, threshold) {
	if(entrada < threshold) {
	   return(0)
        }
 	return(1)
}

#perceptron
perceptron <- function(entrada, rotulo, threshold = 0.5, alpha = 0.01, iter = 100) {
	bias <- 0
	erro <- 0

	#vetor de pesos
	pesos = rep(0, ncol(entrada))

	#treina o algoritmo iter vezes
	for(i in 1:iter) {
	   j <- floor(runif(1, min = 1, max = nrow(entrada + 1)))

 	   #somatorio de pesos e instancias
	   u <- sum(pesos * entrada[j,]) + bias

	   #verificar se neuronio foi ativado ou nao
	   f <- ativacao(u, threshold)
	   
	   #atualiza pesos e bias
	   pesos <- pesos + alpha * (rotulo[j] - f) * entrada[j,]
  	   bias <- bias + alpha * (rotulo[j] - f)
	   
	   #calcula erro
	   erro <- (erro + abs(rotulo[j] - f))/i	   
	}	

	modelo <- list()
	modelo$pesos <- pesos
	modelo$bias <- bias
	modelo$erro <- erro

	modelo
}

#funcao de teste da rede neural
teste <- function(testset, modelo, threshold) {
	resp <- ativacao(sum(modelo$pesos * testset) + modelo$bias, threshold)
	resp
}

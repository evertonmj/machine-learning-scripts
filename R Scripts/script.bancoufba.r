##CARREGA DADOS
load("bancoufba.rData")

####CONVERTE REGISTROS NA DE SALARIO PARA MEDIA
salario <- as.vector(banco.ufba$salario)
salario[which(is.na(salario))] <- mean(banco.ufba$salario, na.rm = TRUE)
banco.ufba$salario <- salario 

##CONVERTE REGISTROS NA DE POUPANCA PARA MEDIA
poupanca <- as.vector(banco.ufba$poupanca)
poupanca[which(poupanca < 0)] <- poupanca[which(poupanca < 0)] * -1
poupanca[which(is.na(poupanca))] <- mean(banco.ufba$poupanca, na.rm = TRUE)
banco.ufba$poupanca <- poupanca

####CONVERTE REGISTROS NA DE TOTAL.EMPRESTIMO PARA 0
total_emprestimo <- as.vector(banco.ufba$total.emprestimo)
total_emprestimo[which(is.na(total_emprestimo))] <- 0
banco.ufba$total.emprestimo <- total_emprestimo

#### CONVERTE REGISTROS NA DE FINANCIAMENTO PARA N
financiamento <- as.vector(banco.ufba$financiamento)
financiamento[which(is.na(financiamento))] <- 0
financiamento[which(financiamento=="S")] <- 1
financiamento[which(financiamento=="N")] <- 0
banco.ufba$financiamento <- as.integer(financiamento)

####EFETUA A CONVERSAO DO ATRIBUTO DE SEXO####
sexo <- as.vector(banco.ufba$sexo)
sexo[which(sexo=="M")] <- 0
sexo[which(sexo=="F")] <- 1
banco.ufba$sexo <- as.integer(sexo)

####EFETUA A NORMALIZACAO DE CAMPOS####
#SALARIO
salario.min <- min(banco.ufba$salario)
salario.max <- max(banco.ufba$salario)
norm.salario <- as.integer((banco.ufba$salario - salario.min)/(salario.max - salario.min))
banco.ufba$salario = norm.salario

#POUPANCA
poupanca.min <- min(banco.ufba$poupanca)
poupanca.max <- max(banco.ufba$poupanca)
norm.poupanca <- as.integer((banco.ufba$poupanca - poupanca.min) / (poupanca.max - poupanca.min))
banco.ufba$poupanca <- norm.poupanca

#TOTAL.EMPRESTIMO
total.emprestimo.min <- min(banco.ufba$total.emprestimo)
total.emprestimo.max <- max(banco.ufba$total.emprestimo)
norm.total.emprestimo <- as.integer((banco.ufba$total.emprestimo - total.emprestimo.min)/(total.emprestimo.max - total.emprestimo.min))
banco.ufba$total.emprestimo <- norm.total.emprestimo

####REMOVE ATRIBUTOS NAO RELEVANTES
banco.ufba$cpf <- NULL
banco.ufba$estado <- NULL
banco.ufba$altura <- NULL
banco.ufba$peso <- NULL

####MUDA NOME DE ATRIBUTOS
#names(banco.ufba)[1] = 1
#names(banco.ufba)[2] = 2
#names(banco.ufba)[3] = 3
#names(banco.ufba)[4] = 4
#names(banco.ufba)[5] = 5

####RODA O PCA
banco.ufba.pca <- prcomp(banco.ufba, scale=TRUE, center=TRUE)

adultset <- read.table("adult.data.txt", header=T, sep=",")

for(i in 1:ncol(adultset)) {
   adultset <-  adultset[!is.na(adultset[i]),]
}

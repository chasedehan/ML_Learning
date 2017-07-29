#Q1 for Data Incubator Challenge
library(dplyr)

NandM <- function(N, M, size) {
  x <- data.frame( floor(runif(size, min=1, max = 7)) )
  for(die in 2:N){
    a <- floor(runif(size, min=1, max = 7)  )
    x <- cbind(x, a)
  }
  names(x) <- 1:N
  #Pull out the rows equal to M
  x$sum <- rowSums(x)
  x <- x %>%
    filter(sum == M) %>%
    select(-sum) 
  #compute the product
  x$product <- apply(x, 1, prod)
  #Pull out the unique values and compute the answers
  return(x %>% unique() %>% summarise(exp_value = mean(product), sd = sd(product)))
}


N <- 8
M <- 24
size <- 5e7
answer <- NandM(N, M, size)
answer

answer2 <- NandM(50, 150, 2e7)
answer3 <- NandM(50, 150, 2e7)
answer4 <- NandM(50, 150, 2e7)
answer5 <- NandM(50, 150, 2e7)
n50 <- rbind(answer2, answer3, answer4, answer5)
mean(n50$exp_value)
mean(n50$sd)

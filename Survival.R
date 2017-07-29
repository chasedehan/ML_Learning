#survival analysis

#guide at blog:  http://justanotherdatablog.blogspot.com/2015/08/survival-analysis-1.html
# part 2: https://www.r-bloggers.com/survival-analysis-2/


#on Home
setwd("/home/chase/Dropbox/R Code/ML")
#on Laptop
setwd("/home/chasedehan/Dropbox/R Code/ML")
#on Work
setwd("C:/Users/cdehan/Dropbox/R Code/ML")

library(survival)

#input data

train <- read.csv("titanic_train.csv", stringsAsFactors=FALSE)
test  <- read.csv("titanic_test.csv",  stringsAsFactors=FALSE)

mini.surv <- survfit(Surv(Survived, Age)~ 1, conf.type="none", data = train)
summary(mini.surv)
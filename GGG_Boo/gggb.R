#Ghouls, Goblins, and Ghosts, Boo!


setwd("/home/chasedehan/Dropbox/R Code/ML/GGG_Boo")

#load libraries
library(tidyverse)
library(caret)

train <- read_csv("train.csv")
test <- read_csv("test.csv")


train$id <- NULL

#ranger model
library(ranger)
rf_model <- ranger(as.factor(type) ~ .,
                   data = train,
                   write.forest = TRUE)
rf_pred <- predict(rf_model, test)$predictions


submit(rf_pred)

#xgboost
library(xgboost)





submit <- function(x) {
  submission <- data.frame(id = test$id, type = x )
  print(head(submission))
  write.csv(submission, file = "ggb_submission.csv", row.names = FALSE)
}

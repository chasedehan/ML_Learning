#allstate2.R
      #Using the stacking ensemble I built on GGG_B process

#Will be using at Level 0:
  # xgbTree, gbm, random forest, bma, Lasso, ridge ---> then maybe some others like: svm, lda, stepwise, etc


setwd("/home/chasedehan/Dropbox/R Code/ML/Allstate")

library(plyr)   
library(tidyverse)
library(caret)
#library(doMC)
#registerDoMC(cores=8)

##################### Prepping Data #############################
prepFeatures <- function(train, test, idName, yName){
  #Comgine the train/test and prep features
  test[, yName] <- NA
  y <- c(train[, yName], test[, yName])
  id <- c(train[, idName], test[, idName])
  pred <- rbind(train, test)
  pred[, idName] <- pred[, yName] <- NULL
  pred <- as.data.frame(model.matrix(~ ., pred)[, -1])
  #Split the data back up 
  train <- list(y = y[!is.na(y)],
                id = id[!is.na(y)],
                predictors = pred[!is.na(y), ])
  test <- list(id = id[is.na(y)],
               predictors = pred[is.na(y), ])
  #return a list
  return(list(train = train, test = test))
}

L0FeatureSet1 <- prepFeatures(as.data.frame(read_csv("train.csv")),
                              as.data.frame(read_csv("test.csv")),
                              "id", "loss")

################ Training the model #######
trainOneFold <- function(this_fold, feature_set, method, grid = NULL) {
  #Make sure that all important parameters are passed as a list of arguments for "other_param"
  # get fold specific cv data
  cv.data <- list()
  cv.data$predictors <- feature_set$train$predictors[this_fold,]
  cv.data$ID <- feature_set$train$id[this_fold]
  cv.data$y <- feature_set$train$y[this_fold]
  # get training data for specific fold
  train.data <- list()
  train.data$predictors <- feature_set$train$predictors[-this_fold,]
  train.data$y <- feature_set$train$y[-this_fold]
  #Fit the model for the fold
  fitted_mdl <- train(x=train.data$predictors,
                      y=train.data$y,
                      method = method,
                      tuneGrid = grid) 
  
  yhat <- predict(fitted_mdl,newdata = cv.data$predictors,type = "raw")
  score <- get(cv_score_function)(cv.data$y,yhat)
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=cv.data$ID,yhat=yhat,y=cv.data$y))
  return(ans)
} 

mae <- function(y, pred)  mean(abs(pred - y))
cv_score_function <- "mae"   
data_folds <- createFolds(L0FeatureSet1$train$y, k=5)

#First go through of data
data_folds <- createFolds(L0FeatureSet1$train$y, k=10)
model_param <- list(method = "ranger")
mdl <- trainOneFold(data_folds$Fold01, L0FeatureSet1, method = "xgbTree")
ggplot(mdl)

grid_tune_param<- expand.grid(nrounds = (2:12) * 25,
                              max_depth = c(3, 5, 7, 9, 11),
                              eta = 0.3,
                              gamma = 0, 
                              colsample_bytree = 0.6,
                              min_child_weight = 1)  

model_param <- list(method = "ranger", tuneGrid = grid_tune_param)
######Then run the model to tune the model
xgb_set <- llply(data_folds,trainOneFold,L0Features, model_param)

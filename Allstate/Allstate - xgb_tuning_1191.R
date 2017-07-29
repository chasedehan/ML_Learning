#Kaggle Functional Set Up

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

#Set the data into a list
L0Features <- prepFeatures(as.data.frame(read_csv("train.csv")),
                              as.data.frame(read_csv("test.csv")),
                              "id", "loss")

################ Training the model #######
trainOneFold <- function(this_fold, feature_set, method, tuning_params = NULL) {
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
                      tuneGrid = tuning_params)
  
  yhat <- predict(fitted_mdl,newdata = cv.data$predictors,type = "raw")
  score <- get(cv_score_function)(cv.data$y,yhat)
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=cv.data$ID,yhat=yhat,y=cv.data$y))
  return(ans)
} 

########### Training the model: general parameters ###############
mae <- function(y, pred)  mean(abs(pred - y))
cv_score_function <- "mae"    #What CV error are we minimizing in trainOneFold?
data_folds <- createFolds(L0Features$train$y, k=5)


#First go through of data using 10% of data
set.seed(567)
ten_percent <- sample(1:length(L0Features$train$y), 0.1 * length(L0Features$train$y))
mdl <- trainOneFold(ten_percent, L0Features, method = "xgbTree")
#Look at the data to see how well it does
ggplot(mdl)

#Select the appropriate tuning parameters for the expanded grid search
grid_tune_param<- expand.grid(nrounds = (3:12) * 30,
                              max_depth = c(3, 4, 5),
                              eta = 0.3,
                              gamma = 0, 
                              colsample_bytree = 0.6,
                              min_child_weight = 1)  

######Then run the model to tune the model
mdl2 <- llply(data_folds,trainOneFold,L0Features, method = "xgbTree", tuning_params = grid_tune_param)
      #Repeat if necessary

#Generating the tuned model parameters
new_grid <-  ldply(mdl2, function(x){ x$fitted_mdl$bestTune} )[, -1]
new_grid <- new_grid %>% 
  colMeans() %>%
  rbind() %>%
  as.data.frame()

#Rerun the model with tuned parameters for CV
mdl2 <- llply(data_folds,trainOneFold,L0Features, method = "xgbTree", tuning_params = new_grid)
#Check how the model performs
cv.score <- mean(ldply(mdl2, function(x){ x$score})[, -1])
cv.score  

#Training model on full dataset
tuned_mdl <- train(x=L0Features$train$predictors,
                  y=L0Features$train$y,
                  method = "xgbTree",
                  tuneGrid = new_grid)
#predict the values with the newly trained parameters
train_yhat <- predict(tuned_mdl, newdata = L0Features$train$predictors, type = "raw") 
test_yhat <- predict(tuned_mdl, newdata = L0Features$test$predictors, type = "raw") 

##########################
###########      And then start the ensembling process of combining the models
###########################

#Lets work now on functioning the ensembling

submit <- function(x) {
  submission <- data.frame(id = L0Features$test$id, loss = x)
  print(head(submission))
  write.csv(submission, file = "allstate_submission.csv", row.names = FALSE)
}
submit(test_yhat)

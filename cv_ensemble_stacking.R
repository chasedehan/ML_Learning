#cv_ensemble_stacking.R
  #This script is building my own folds for a stacking ensemble

setwd("/home/chasedehan/Dropbox/R Code/ML")

library(plyr)   
library(tidyverse)
library(caret)



############### Prepping the feature lists ###############
##Check GGG_B for updates to this

prepL0Features_cat <- function(df){
  id <- df[, response_Id]
  if (class(df[, response_col_name]) != "NULL") {  #this line didnt work for GGG_B
    y <- as.factor(df[, response_col_name])
  } else {
    y <- NULL
  }
  predictors <- select(df, -Survived, -PassengerId)
  return(list(id=id,y=y,predictors=predictors))
}

prepL0Features_cont <- function(df){
  id <- df[, response_Id]
  if (class(df[, response_col_name]) != "NULL") {
    y <- df[, response_col_name]
  } else {
    y <- NULL
  }
  predictors <- select(df, -Survived, -PassengerId)
  return(list(id=id,y=y,predictors=predictors))
}

################ Training the model #######
trainOneFold <- function(this_fold, feature_set, other_param) {
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
  fitted_mdl <- do.call(train,
                        c(list(x=train.data$predictors,y=train.data$y),
                          verbose = 0,
                          other_param) )
  
  yhat <- predict(fitted_mdl,newdata = cv.data$predictors,type = "raw")
  score <- get(cv_score_function)(cv.data$y,yhat)
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=cv.data$ID,yhat=yhat,y=cv.data$y))
  return(ans)
} 


#####################  Setting up the data #######################
#Name these to prep the data
response_Id <- "PassengerId"       #Name of the id in the dataframe
response_col_name <- "Survived"    #Name of the response variable in dataframe
#Build the features
L0FeatureSet1 <- list(train=prepL0Features_cat(train_clean),
                      test=prepL0Features_cat(test_clean))

########### Training the model: general parameters ###############
cv_score_function <- "accuracy"   #What measure are we trying to measure in CV
data_folds <- createFolds(L0FeatureSet1$train$y, k=5)  #creating the folds

############# Each model parameters parameters
      #This is where we add the 
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param_xgb <- list(method = "xgbTree",
                        tuneGrid = grid_tune_param,
                        metric = "Kappa")

######Then run the model
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param_xgb)
# CV Error Estimate
cv_y <- do.call(c,lapply(xgb_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
cv.score <- mean(do.call(c, lapply(xgb_set, function(x){x$score})))
cv.score
accuracy(cv_y,cv_yhat)  #cv.score and accuracy() provide nearly the same as 

#Should build all of this into a list so I can pass all values as a single call.

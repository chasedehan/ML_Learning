#cv_ensemble_stacking.R
  #Keeping the same models at Level0 ( xgbTree, ranger, gbm)
  #Going to tune them and see if we can beat the previous scores

#Previously was using an automatic grid search and each cv model was estimated using potentially different values

library(plyr)   
library(tidyverse)
library(caret)
library(doMC)
registerDoMC(cores=8)

setwd("/home/chasedehan/Dropbox/R Code/ML/GGG_Boo")
train_data <- read_csv("train.csv")
test_data <- read_csv("test.csv")

#Prepping the dataframe
    #combine to make all_data
test_data$type <- NA
all_data <- rbind(train_data, test_data)
type <- as.factor(all_data$type)
all_data <- as.data.frame(model.matrix(~ ., all_data[, -7]))[, -1]
all_data <- cbind(type, all_data)
head(all_data)

  #Split back up
train_data <- all_data %>% filter(!is.na(type))
test_data <- all_data %>% filter(is.na(type))
test_data$type <- NULL

############### Prepping the feature lists ###############
prepL0Features_cat <- function(df){
  id <- df[, response_Id]
  if (class(df$type) != "NULL") {  #WONKY, it works for the other data, hardcoded here
    y <- as.factor(df[, response_col_name])
    df[, response_Id] <- df[, response_col_name] <- NULL
  } else {
    y <- NULL
    df[, response_Id] <- NULL
  }
  return(list(id=id,y=y,predictors=df))
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
response_Id <- "id"       #Name of the id in the dataframe
response_col_name <- 'type'    #Name of the response variable NOT PASSING CORRECTLY RIGHT NOW
#Build the features
L0FeatureSet1 <- list(train=prepL0Features_cat(train_data),
                      test=prepL0Features_cat(test_data))

########### Training the model: general parameters ###############
accuracy <- function(y, yhat) {
  conf <- table(y, yhat)
  acc <- sum(diag(conf)) / sum(conf)
  return(acc)
}
cv_score_function <- "accuracy"   #What measure are we trying to measure in CV
data_folds <- createFolds(L0FeatureSet1$train$y, k=5)  #creating the folds
L0_data <- list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y)

#Could probably compartmentalize the below a little better

################################################
############# XGB Model ########################
################################################
grid_tune_param<- expand.grid(nrounds = (1:20) * 10,
                              max_depth = 1,
                              eta = c(0.2, 0.3, 0.4),
                              gamma = 0, 
                              colsample_bytree = c(0.4, 0.6, 0.8),
                              min_child_weight = 1)  
model_param <- list(method = "xgbTree", tuneGrid = grid_tune_param)
######Then run the model to tune the model
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
#Look over the plots
lapply(xgb_set, function(x){ ggplot(x$fitted_mdl) })

#Selecting one set of parameters for knowledge of "true" CV scores
a <- ldply(xgb_set, function(x){ x$fitted_mdl$bestTune} )[, -1]
c <- colMeans(a)
d <- as.data.frame(rbind(c))

#restate the model parameters
model_param <- list(method = "xgbTree", tuneGrid = d)
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model to see what the current error is
cv.score <- mean(ldply(xgb_set, function(x){ x$score})[, -1])
cv.score  #0.7147239

# final model fit using the parameters determined in the first instance
xgb_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
xgb_test_yhat <- predict(xgb_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     


################################################
############# Ranger Model ########################
################################################
grid_tune_param <-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "ranger", tuneGrid = grid_tune_param)

######Then run the model
rf_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)

lapply(rf_set, function(x){ ggplot(x$fitted_mdl) })

grid_tune_param <- expand.grid( mtry = 3:8)
#restate the model parameters
model_param <- list(method = "ranger", tuneGrid = grid_tune_param)
rf_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)

###### And check the output of the model to see what the current error is
cv.score <- mean(ldply(rf_set, function(x){ x$score})[, -1])
cv.score #0.727728

# final model fit  
rf_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
rf_test_yhat <- predict(rf_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     


################################################
############# GBM Model ########################
################################################
grid_tune_param <-  expand.grid(interaction.depth = 1, 
                               n.trees = (1:20)*10, 
                               shrinkage = c(0.05, 0.1, 0.15),
                               n.minobsinnode = c(15,20)) #use expand.grid() to build the model parameters
model_param <- list(method = "gbm", tuneGrid = grid_tune_param)

######Then run the model
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
lapply(gbm_set, function(x){ ggplot(x$fitted_mdl) })

#Selecting one set of parameters for knowledge of "true" CV scores
a <- ldply(gbm_set, function(x){ x$fitted_mdl$bestTune} )[, -1]
c <- colMeans(a)
d <- as.data.frame(rbind(c))

#restate the model parameters
model_param <- list(method = "gbm", tuneGrid = d)
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)

####### And check the output of the model
cv.score <- mean(ldply(gbm_set, function(x){ x$score})[, -1])
cv.score  #0.7254256

# final model fit  #It looks
gbm_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
gbm_test_yhat <- predict(gbm_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     

################### Level 1 Training Data Prep ########################
gbm_yhat <- unlist(lapply(gbm_set,function(x){x$predictions$yhat}))
xgb_yhat <- unlist(lapply(xgb_set,function(x){x$predictions$yhat}))
rf_yhat <- unlist(lapply(rf_set,function(x){x$predictions$yhat}))
id <- unlist(lapply(gbm_set,function(x){x$predictions$ID}))
y <- unlist(lapply(gbm_set,function(x){x$predictions$y}))
L1_train_data <- data.frame(id, y, gbm_yhat,xgb_yhat,rf_yhat)

L1_train_data <- as.data.frame(sapply(L1_train_data, as.numeric))

#And check the correlations
cor(L1_train_data)

################### Level 1 Test Data Prep ########################
L1_test_data <- data.frame(id = L0FeatureSet1$test$id,
                           gbm_yhat = as.numeric(gbm_test_yhat),
                           xgb_yhat = as.numeric(xgb_test_yhat),
                           rf_yhat = as.numeric(rf_test_yhat))

############# Level 1 - Neural Net Weightings #################
###############################################################
# model specific training parameter
train_ctrl <- trainControl(method="cv",
                           number=5,
                           repeats=1,
                           verboseIter=FALSE)

model_param <- list(method = "nnet", 
                    trControl = train_ctrl, 
                    tuneLength = 10)

# train the model
#center and scale the 
nnet_mdl <- train(y = as.factor(L1_train_data$y), x = L1_train_data[, 3:5], method = "nnet", tuneLength = 15, verbose = F)
nnet_pred <- predict(nnet_mdl, newdata = L1_test_data)
#now need to convert back into the appropriate category
#Seems overly complicated

nnet_clean <- ifelse(nnet_pred == 1, "Ghost", ifelse(nnet_pred == 2, "Ghoul", "Goblin"))
submit(nnet_clean)


############################ Build Submission File ####
submit <- function(x) {
  submission <- data.frame(id = L1_test_data$id, type = x )
  print(head(submission))
  write.csv(submission, file = "ggb_submission.csv", row.names = FALSE)
}


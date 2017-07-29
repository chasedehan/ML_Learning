#cv_ensemble_stacking.R
  #This script submitted received a 0.72779



library(plyr)   
library(tidyverse)
library(caret)


setwd("/home/chasedehan/Dropbox/R Code/ML/GGG_Boo")
#It's weird that I have to convert to a data.frame to make this work, but it has to do with the prep function
train_data <- as.data.frame(read_csv("train.csv"))
test_data <- as.data.frame(read_csv("test.csv"))

#Prepping the categorical variables
test_data <- as.data.frame(model.matrix(~ ., test_data))[, -1]
holder <- as.data.frame(model.matrix(~ ., train_data[, -7]))[, -1]
train_data <- cbind(holder, type = train_data[, 7])
rm(holder)


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
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "xgbTree", tuneGrid = grid_tune_param)
                        
######Then run the model
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(xgb_set, function(x){x$score})))
cv.score  #0.7304144 accuracy
# final model fit
xgb_mdl <- do.call(train, c(L0_data, verbose = 0, model_param_xgb) )
xgb_test_yhat <- predict(xgb_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     

################################################
############# Ranger Model ########################
################################################
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "ranger", tuneGrid = grid_tune_param)

######Then run the model
rf_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(rf_set, function(x){x$score})))
cv.score  #0.7305225 accuracy
# final model fit  #It looks
rf_mdl <- do.call(train, c(L0_data, verbose = 0, model_param_rf) )
rf_test_yhat <- predict(rf_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     

################################################
############# XGB Model ########################
################################################
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "gbm", tuneGrid = grid_tune_param)

######Then run the model
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(rf_set, function(x){x$score})))
cv.score  #0.7305225 accuracy
# final model fit  #It looks
gbm_mdl <- do.call(train, c(L0_data, verbose = 0, model_param_rf) )
gbm_test_yhat <- predict(gbm_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     


###############################################################
############# Kmeans Model - Yes it is #############
###############################################################
    #Don't worry for now, but really a good idea, need to build
kmeans_yhat <- kmeans(L0FeatureSet1$train$predictors, centers = 3)
#then attach to id,
    #Then, merge with L1 dataframe to make sure that IDs line up
  

################### Level 1 Training Data Prep ########################
gbm_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
xgb_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rf_yhat <- do.call(c,lapply(rf_set,function(x){x$predictions$yhat}))
id <- do.call(c,lapply(gbm_set,function(x){x$predictions$ID}))
y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
L1_train_data <- data.frame(id, y, gbm_yhat,xgb_yhat,rf_yhat)

################### Level 1 Test Data Prep ########################
L1_test_data <- data.frame(id = L0FeatureSet1$test$id,
                           gbm_yhat = as.numeric(gbm_test_yhat),
                           xgb_yhat = as.numeric(xgb_test_yhat),
                           rf_yhat = as.numeric(rf_test_yhat) )





############# Level 1 - Neural Net Weightings #################
###############################################################

# model specific training parameter
train_ctrl <- trainControl(method="cv",
                                 number=5,
                                 repeats=1,
                                 verboseIter=FALSE)

model_param <- list(method = "nnet", 
                    trControl = train_ctrl, 
                    tuneLength = 7)

# train the model

l1_nnet_mdl <- do.call(train,c(list(x=L1FeatureSet$train$predictors,y=L1FeatureSet$train$y),
                               CARET.TRAIN.PARMS,
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))
nnet_mdl <- train(y = as.factor(L1_train_data$y), x = L1_train_data[, 3:5], method = "nnet", verbose = F)
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


################################################## Also tried
###Attempted to put in all this "feature engineering" but gave a worse fit - probably started overfitting
all_data <- all_data %>%
  mutate(flesh_soul = rotting_flesh * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_bone = rotting_flesh * bone_length,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         hair_soul = hair_length * has_soul)
#cv_ensemble_stacking.R
  #Previous version of this script scored 0.72779
    #"Feature Engineering didn't do much


library(plyr)   
library(tidyverse)
library(caret)


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
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "xgbTree", tuneGrid = grid_tune_param)
                        
######Then run the model
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(xgb_set, function(x){x$score})))
cv.score  #0.7252252 accuracy
# final model fit
xgb_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
xgb_test_yhat <- predict(xgb_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     
ggplot(xgb_mdl)
################################################
############# Ranger Model ########################
################################################
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "ranger", tuneGrid = grid_tune_param)

######Then run the model
rf_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(rf_set, function(x){x$score})))
cv.score  #0.7117838 accuracy
# final model fit  
rf_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
rf_test_yhat <- predict(rf_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     


################################################
############# GBM Model ########################
################################################
grid_tune_param<-  NULL #use expand.grid() to build the model parameters
model_param <- list(method = "gbm", tuneGrid = grid_tune_param)

######Then run the model
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
ggplot(gbm_set$Fold1$fitted_mdl)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(gbm_set, function(x){x$score})))
cv.score  #0..7441081 accuracy
# final model fit  #It looks
gbm_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
gbm_test_yhat <- predict(gbm_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     
ggplot(gbm_mdl)

################################################
############# kmeans Model ########################
################################################
kmeans_yhat <- kmeans(L0FeatureSet1$train$predictors, centers = 3)
#attach Id so that they line up
kmeans_train_yhat <- data.frame(id = L0FeatureSet1$train$id, kmeans_yhat = kmeans_yhat$cluster)
#Do the same on the test data
kmeans_yhat <- kmeans(L0FeatureSet1$test$predictors, centers = 3)
kmeans_test_yhat <- data.frame(id = L0FeatureSet1$test$id, kmeans_yhat = kmeans_yhat$cluster)
rm(kmeans_yhat)

################################################
############# Neural Net Model ########################
################################################
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "nnet", tuneGrid = grid_tune_param)

######Then run the model
nnet_set <- llply(data_folds,trainOneFold,L0FeatureSet1, model_param)
####### And check the output of the model
cv.score <- mean(do.call(c, lapply(nnet_set, function(x){x$score})))
cv.score  #0.7411892 accuracy
# final model fit  
nnet_mdl <- do.call(train, c(L0_data, verbose = 0, model_param) )
nnet_test_yhat <- predict(nnet_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")                     

############# AdaBagged  Model ########################
#######################################################
#Work on this next - doesn't now work
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "AdaBag", tuneGrid = grid_tune_param)
####### And check the output of the model
mean(do.call(c, 
             lapply(llply(data_folds,trainOneFold,L0FeatureSet1, model_param), 
                    function(x){x$score})))
# final model fit and predictions
adaB_test_yhat <- predict(do.call(train, c(L0_data, verbose = 0, model_param) ), 
                          newdata = L0FeatureSet1$test$predictors, type = "raw")                     


############# SVM Model ########################
#######################################################
#Trying to trim down the parameters, but this doesn't work right now
grid_tune_param<-  NULL  #use expand.grid() to build the model parameters
model_param <- list(method = "svmLinear", tuneGrid = grid_tune_param)
####### And check the output of the model
mean(do.call(c, 
             lapply(llply(data_folds,trainOneFold,L0FeatureSet1, model_param), 
                    function(x){x$score})))
# final model fit and predictions
svmL_test_yhat <- predict(do.call(train, c(L0_data, verbose = 0, model_param) ), 
                          newdata = L0FeatureSet1$test$predictors, type = "raw")                     


############# KNN models ##############################
#######################################################
#for later
#http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/

#choose a k value odd/even opposite of the number of classes to avoid a tie
#Rescale data
#Makes predictions "just in time" from computing how far from the neighbors it is



################### Level 1 Training Data Prep ########################
gbm_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
xgb_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rf_yhat <- do.call(c,lapply(rf_set,function(x){x$predictions$yhat}))
nnet_yhat <- do.call(c,lapply(nnet_set,function(x){x$predictions$yhat}))
id <- do.call(c,lapply(gbm_set,function(x){x$predictions$ID}))
y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
L1_train_data <- data.frame(id, y, gbm_yhat,xgb_yhat,rf_yhat, nnet_yhat)

L1_train_data <- merge(L1_train_data, kmeans_train_yhat, by = "id" )
head(L1_train_data)
#And check the correlations
cor(L1_train_data)

################### Level 1 Test Data Prep ########################
L1_test_data <- data.frame(id = L0FeatureSet1$test$id,
                           gbm_yhat = as.numeric(gbm_test_yhat),
                           xgb_yhat = as.numeric(xgb_test_yhat),
                           rf_yhat = as.numeric(rf_test_yhat),
                           nnet_yhat = as.numeric(nnet_test_yhat) )
L1_test_data <- merge(L1_test_data, kmeans_test_yhat, by = "id")
head(L1_test_data)

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
#center and scale the 
nnet_mdl <- train(y = as.factor(L1_train_data$y), x = L1_train_data[, 3:7], method = "nnet", verbose = F)
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



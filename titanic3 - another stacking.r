#titanic3 - another stacking.R
#Titanic2.R didn't do much for generating high accuracy on the leaderboard
    #Trying here with generating the folds a lo:
    #https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/ensemble-model-stacked-model-example/notebook



setwd("/home/chasedehan/Dropbox/R Code/ML")


library(plyr)
library(tidyverse)
library(rpart)   #recursive partitioning and regression trees
library(caret)
library(dummies)  #for creating the dummies

######################################################################
############## Getting data into the appropriate form ################
######################################################################



train <- read_csv("titanic_train.csv")
test  <- read_csv("titanic_test.csv")

##########Clean up the data ###########################################
#Need to merge the training and test data together, but test doesn't have a "Survived" column, so we create it
test$Survived <- 1000
#merge test and train
all_data <- data.frame(rbind(train, test))

#convert strings from factors to character
all_data$Name <- as.character(all_data$Name)
#apply to every observation in dataframe with sapply on splitting the title out of the name field
all_data$Title <- sapply(all_data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
#want to remove those with small number of observations
all_data <- all_data %>%
  group_by(Title) %>%
  summarise(count = n()) %>%
  mutate(new = ifelse(count < 7, "other", Title)) %>%
  merge(all_data, by = "Title") %>%
  select(-count, -Title) %>%
  ungroup()
names(all_data)[1] <- "Title"

#insert the embark
all_data$Embarked[which(all_data$Embarked == "")] <- "S"
all_data$Embarked[which(is.na(all_data$Embarked))] <- "S"
#mess with the fare
all_data$Fare[which(is.na(all_data$Fare))] <-median(all_data$Fare, na.rm=T)

#Predict the age and reassign the values
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Embarked + Title, data = all_data[!is.na(all_data$Age),], method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, newdata = all_data[is.na(all_data$Age),] )
rm(predicted_age)

#convert the string variables back into factors so randomForest can process them
all_data$Pclass <- as.factor(all_data$Pclass)
all_data$Ticket <- all_data$Cabin <- all_data$Name <-  NULL


Survived <- all_data$Survived
all_data <- as.data.frame(model.matrix(Survived ~ ., data = all_data))
all_data$`(Intercept)` <- NULL
all_data <- cbind(Survived, all_data)

#Create clusters as a predictor
#Set at 4 groups because thats what the scree plot showed worked
km <- kmeans(select(all_data, -Survived, -PassengerId), centers = 4, nstart=20)
#cbind back with the train data
km<-dummy(km$cluster)
all_data <- cbind(all_data,km[,-1])
rm(km)



#split back into test and train data
train_S <- all_data$Survived != 1000
test_clean <- all_data[!train_S,]
train_clean <- all_data[train_S,]
Survived <- train_clean$Survived
rm(train, test, train_S)


##########Build the stacking model###########################################
#and creating a list to pull the values from in the below functions

prepL0FeatureSet1 <- function(df){
  id <- df$PassengerId
  if (class(df$Survived) != "NULL") {
    y <- as.factor(df$Survived)
  } else {
    y <- NULL
  }
  predictors <- select(df, -Survived, -PassengerId)
  return(list(id=id,y=y,predictors=predictors))
}

#two feature sets for different types of models
L0FeatureSet1 <- list(train=prepL0FeatureSet1(train_clean),
                      test=prepL0FeatureSet1(test_clean))
#######Writing an accuracy function

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
  score <- accuracy(cv.data$y,yhat)
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=cv.data$ID,yhat=yhat,y=cv.data$y))
  return(ans)
} 



# set caret training parameters for each model in particular

CARET.TUNE.GRID <-  expand.grid(mtry=2*as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))))


CARET.TRAIN.OTHER.PARMS <- list(method = "xgbTree",
                                #tuneGrid=CARET.TUNE.GRID,
                                metric = "Kappa")



data_folds <- createFolds(train_clean$Survived, k=5)  #will create the same folds for each set of the data


trainOneFold(data_folds$Fold1, L0FeatureSet1, other_param = CARET.TRAIN.OTHER.PARMS)

xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet1, CARET.TRAIN.OTHER.PARMS)
xgb_set$Fold3$fitted_mdl
#trainOneFold(data_folds$Fold1, L0FeatureSet1)
# CV Error Estimate
cv_y <- do.call(c,lapply(xgb_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
cv.score <- mean(do.call(c, lapply(xgb_set, function(x){x$score})))
cv.score
accuracy(cv_y,cv_yhat)  #cv.score and accuracy() provide nearly the same as 

# final model fit
xgb_mdl <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     method = "xgbTree")  )
xgb_mdl$bestTune  #Differs from the models from the cross validation

xgb_test_pred <- predict(xgb_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")
kag_submit(L0FeatureSet1$test$id, xgb_test_pred, id_name = "PassengerId", response_name = "Survived",  "titanic_submission")















CARET.TRAIN.PARMS <- list(method="ranger")   

CARET.TUNE.GRID <-  expand.grid(mtry=2*as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))))

MODEL.SPECIFIC.PARMS <- list(verbose=0,num.trees=500)

CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

rngr_mdl <- do.call(train,
                    c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))
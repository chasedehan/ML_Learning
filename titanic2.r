#titanic2.R
  #This file is formalizing much of what I have learned since starting and I am going to attempt to move near the top of the leaderboard
  #This smaller file allows for more in the way of building large ensembled models to best predict the outcomes

#on Laptop
setwd("/home/chasedehan/Dropbox/R Code/ML")

library(plyr)
library(tidyverse)
library(rpart)   #recursive partitioning and regression trees
library(caret)
library(doParallel) #does this even do anything?
library(dummies)  #for creating the dummies
registerDoParallel(8,cores=8)
getDoParWorkers()

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
a<-dummy(km$cluster)
all_data <- cbind(all_data,a[,-1])




#split back into test and train data
train_S <- all_data$Survived != 1000
test_clean <- all_data[!train_S,]
test_clean$Survived <- NULL
train_clean <- all_data[train_S,]
train_clean$PassengerId <- NULL #We don't need this anymore, but we do in the test data
Survived <- train_clean$Survived

######################################################################
############## Decision Tree Methodology #############################
######################################################################
#Creating the training and test data sets for the data because we don't have labels for the training set
# Set seed of 567 for reproducibility
set.seed(567)
# Store row numbers for training set: index_train
index_train <- sample(1:nrow(train_clean), 2/3 * nrow(train_clean))
# Create training set: training_set
train <- train_clean[index_train, ]
# Create test set: test_set
validate <- train_clean[-index_train,]

######################################################################
############ Basic Process -   ##########################
######################################################################
#split into training/test dataframes 
#Run a bunch of models through CV
  #Select appropriate tuning parameters
#Select "best" models
#Ensemble and weight appropriately


######################################################################
############ Training contral   ##########################
######################################################################
#we specify the 
repeatControl <- trainControl(## 10-fold CV with 10 repeats
  method = "repeatedcv",
  number = 10,
  repeats = 10)

cvControl <- trainControl(## 10-fold CV with 10 repeats
  method = "cv",
  number = 10)

centerScale <- c("center","scale")


######################################################################
############ Fitting the models   ##########################
#####################################################################b
#build data.frame to put all Level 1 predictions into
train_pred <- data.frame(Survived)
test_pred <- data.frame(PassengerId = test_clean$PassengerId)
model_names <- c("gbm", "rf", "knn", "xgb", "nnet", "glm")

#Gradient Boosted Tree
gbm_grid <-  expand.grid(interaction.depth = c(1,3,5), #says test on only depths of 1
                        n.trees = (1:40)*20, #n.trees is equal to 50:1500 by 50
                        shrinkage = 0.1,
                        n.minobsinnode = 15)
gbm_fit <- train(as.factor(Survived) ~ ., data = train_clean, 
                  method = "gbm", 
                  trControl = repeatControl,
                  verbose = FALSE,
                  tuneGrid = gbm_grid)
gbm_fit
ggplot(gbm_fit)
gbm <- predict(gbm_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, gbm)
gbm <- predict(gbm_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, gbm)
rm(gbm)
#for singular submission
gbm <- predict(gbm_fit, newdata = test_clean)

#Random Forest
rf_fit <- train(as.factor(Survived) ~ ., data = train_clean,
                method = "rf",   #using rf because the probabilities weren't working with ranger
                trControl = cvControl)
rf_fit
ggplot(rf_fit)
rf <- predict(rf_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, rf)
rf <- predict(rf_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, rf)
rm(rf)

#K nearest neighbors
knn_fit <- train(as.factor(Survived) ~ ., data = train_clean,
                 method = "knn",
                 preProcess = centerScale,
                 trControl = cvControl,
                 tuneLength = 20) #shows how many to do it
knn_fit
knn <- predict(knn_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, knn)
knn <- predict(knn_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, knn)
rm(knn)

#### xgBoost
xgb_fit <- train(as.factor(Survived) ~ ., data = train_clean,
                 method = "xgbTree",
                 trControl = cvControl)
xgb_fit
ggplot(xgb_fit)
xgb <- predict(xgb_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, xgb)
xgb <- predict(xgb_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, xgb)
rm(xgb)
                   
#neural networks                   
nnet_fit <- train(as.factor(Survived) ~ ., data = train_clean,
                  method = "nnet",
                  preProcess = centerScale,
                  trControl = cvControl)
nnet_fit
nnet <- predict(nnet_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, nnet)
nnet <- predict(nnet_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, nnet)
rm(nnet)
                   
#Logistic regression
glm_fit <- train(as.factor(Survived) ~ ., data = train_clean,
                  method = "glm",
                  trControl = cvControl)
glm_fit
glm <- predict(glm_fit, newdata = train_clean, type = "prob")[2]
train_pred <- cbind(train_pred, glm)
glm <- predict(glm_fit, newdata = test_clean, type = "prob")[2]
test_pred <- cbind(test_pred, glm)
rm(glm)

names(train_pred)[2:7] <- names(test_pred)[2:7] <- model_names
                   
######################################################################
################ Building Level 1 Dataframe ##########################
######################################################################

#function for predicting the observations into a dataframe



######################################################################
########################## Ensembling Predictions ####################
######################################################################
#We already have the first level dataset built 
nnet2_fit <- train(as.factor(Survived) ~ ., data = train_pred,
                  method = "nnet",
                  preProcess = centerScale,
                  trControl = cvControl)
nnet2_fit
level2_pred <- predict(nnet2_fit, newdata = test_pred)  #Only scored 0.75120, which lost to the simpler model

round(rowMeans(test_pred[,-1]))
lapply(test_pred, class)

#using glm for the ensemble

library(glmnet)
cvglm_fit <- cv.glmnet(as.matrix(train_pred[,2:7]), train_pred$Survived)
cvglm_fit
glm_pred <- predict(cvglm_fit, newx = as.matrix(train_pred[,2:7]), s = glm2_fit$lambda.min)
summary(glm_pred)
cutoff <- seq(from = 0, to = 1, by = 0.02)
cutoff
acc_vector <- rep(NA, length(cutoff))
j <- 1
for (i in cutoff) {
  pred_cutoff <- ifelse(glm_pred>i, 1, 0)
  conf <- table(pred_cutoff, train_pred$Survived)
  acc_vector[j] <- sum(diag(conf)) / sum(conf)
  j <- j + 1
}
cbind(cutoff,acc_vector) 
plot(cutoff, acc_vector, type = "l")

cvglm_pred <- predict(cvglm_fit, newx = as.matrix(test_pred[,2:7]), s = cvglm_fit$lambda.min)
glm2_fit <- glm(Survived ~ ., data = train_pred)
glm2_pred <- predict(glm2_fit, newdata = test_pred)

#choose a 0.4 cutoff value
glm2_pred <- ifelse(glm2_pred > 0.4, 1, 0)
cvglm_pred <- ifelse(cvglm_pred > 0.4, 1, 0)
a<-cbind(glm2_pred, cvglm_pred)
######################################################################
############ Automating selection process   ##########################
######################################################################
class_models <- c("ranger", "LogitBoost", "glm", "gbm", "bayesglm")
scale_models <- c("knn", "nnet")
classTrain <- function(model) {
  train(as.factor(Survived) ~ .,
        data = train_clean,
        method = model,
        trControl = cvControl)
}

output <- as.data.frame(matrix(ncol=length(class_models), nrow = nrow(test_clean)))
names(output) <- class_models
for(i in seq_along(1:length(class_models))) {
  model <- classTrain(class_models[i])
  output[,i] <- predict(model, newdata = test_clean)
}
output <- cbind(test_clean$Survived, output)
names(output)[1] <- "Survived"
nnet_fit <- train(as.factor(Survived) ~ ., data = output,
                  method = "nnet",
                  preProcess = centerScale,
                  trControl = cvControl)

#############
######################################################################
################## Build file for submission #########################
######################################################################

submission <- data.frame(PassengerId = test_clean$PassengerId, Survived = gbm)
head(submission)
write.csv(submission, file = "titanic_submission.csv", row.names = FALSE)

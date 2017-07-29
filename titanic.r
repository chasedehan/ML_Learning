#titanic


#on Home
setwd("/home/chase/Dropbox/R Code/ML")
#on Laptop
setwd("/home/chasedehan/Dropbox/R Code/ML")
#on Work
setwd("C:/Users/cdehan/Dropbox/R Code/ML")

library(dplyr)
library(rpart)   #recursive partitioning and regression trees



######################################################################
############## Getting data into the appropriate form ################
######################################################################



train <- read.csv("titanic_train.csv", stringsAsFactors=FALSE)
test  <- read.csv("titanic_test.csv",  stringsAsFactors=FALSE)

##########Clean up the data ###########################################
#Need to merge the training and test data together, but test doesn't have a "Survived" column, so we create it
test$Survived <- NA
#merge test and train
all_data <- data.frame(rbind(train, test))
#We can also split the title out of the name field, because there is more information contained in there
all_data$Name[1]
#convert strings from factors to character
all_data$Name <- as.character(all_data$Name)
#splitting the string apart on the comma and period because all the cells have these
#[[1]] because it actually returns a list and we want the first element
strsplit(all_data$Name[1], split = '[,.]')[[1]]
#we want the title, which comes right after the last name, and is the second element
strsplit(all_data$Name[1], split = '[,.]')[[1]][2]
#apply to every observation in dataframe with sapply
all_data$Title <- sapply(all_data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

#insert the embark
all_data$Embarked[which(all_data$Embarked == "")] <- "S"
#mess with the fare
all_data$Fare[which(is.na(all_data$Fare))] <-median(all_data$Fare, na.rm=T)

#Predict the age and reassign the values
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Embarked + Title, data = all_data[!is.na(all_data$Age),], method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, newdata = all_data[is.na(all_data$Age),] )

#convert the string variables back into factors so randomForest can process them
all_data$Title <- as.factor(all_data$Title)
all_data$Sex <- as.factor(all_data$Sex)
all_data$Embarked <- as.factor(all_data$Embarked)

#split back into test and train data
test_clean <- all_data[is.na(all_data$Survived),]
train_clean <- all_data[!is.na(all_data$Survived),]


#Creating the training and test data sets for the data because we don't have labels for the training set
# Set seed of 567 for reproducibility
set.seed(567)
# Store row numbers for training set: index_train
index_train <- sample(1:nrow(train_clean), 2/3 * nrow(train_clean))
# Create training set: training_set
train <- train_clean[index_train, ]
# Create test set: test_set
test <- train_clean[-index_train,]




######################################################################
###################### k-fold validation #############################
######################################################################
#see far down below for the use of the caret package

#can use cv.glm() for glm models in library(boot), which is the bootstrap library

#Code by hand doesn't quite work yet, but will at some point

#Randomly shuffle the data
train_shuffle<-train_clean[sample(nrow(train_clean)),]

#Create 10 equally size folds, assigning a group value to each value in the vector
    #Can change "breaks=" to however many breaks we want
n_folds <- 10
folds <- cut(seq(1,nrow(train_shuffle)),breaks=n_folds,labels=FALSE)

#Perform 10 fold cross validation
      #Change the 10 here it necessary
cv_tmp <- matrix(NA, nrow = n_folds, ncol = length(train_shuffle))
for(i in 1:n_folds){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- train_shuffle[testIndexes, ]
  trainData <- train_shuffle[-testIndexes, ]
  x<- trainData$Sex  #or whatever the classifiers are
  y <- trainData$Survived
  fitted_model < - lm() #or whatever model
  x <- testData$Sex #same classifiers as above
  y <- testData$Survived
  pred <- predict()
  cv_tmp[k, ] <- sapply(as.list(data.frame(pred)), function(y_hat) mean((y - y_hat)^2))
}
cv <- colMeans(cv_tmp)

######################################################################
############## Decision Tree Methodology #############################
######################################################################

#A basic decision tree methodology
tree <- rpart(Survived ~ Pclass + Sex + Age + Parch + Fare + Embarked, data = train, method = "class")
plot(tree, uniform = T)
text(tree)

#predict values
tree_predict <- predict(tree, newdata = test, type = "class")

#Confusion Matrix - 
conf_tree <- table(test$Survived, tree_predict)
conf_tree
acc_tree <- sum(diag(conf_tree)) / sum(conf_tree)
acc_tree


###Build for submission
submission <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
#Write to csv for upload
write.csv(submission, file = "titanic_submission.csv", row.names = FALSE)


##############3
#fancy plot
library(rattle)   #It has been a while back, but I believe I had to install libgtk2.0-dev   
#enhanced tree plots
library(rpart.plot)
prp(tree)
fancyRpartPlot(tree)
#More color selection for fancy tree plots
library(RColorBrewer)


######################################################################
############ Random Forest Model Datacamp   ##########################
######################################################################

#Handles the overfitting we saw with decision trees.  Basically, the model overfits the decision trees 
    #randomly and has the majority vote "win" the outcome.
library(randomForest)

#Created the forest model object
forest_model <- randomForest(as.factor(Survived) ~ Pclass + Age + Title + Sex + Embarked,
                             data = train, 
                             importance = T,
                             ntree= 1000)
rf_prediction <- predict(forest_model, newdata=test)
#look at the plot to see what is more important
varImpPlot(forest_model)
#Building the submission
conf_rf <- table(test$Survived, rf_prediction)
conf_rf
acc_rf <- sum(diag(conf_rf)) / sum(conf_rf)
acc_rf


#67%, which is not as good as the one above.
      #We need to work on some diagnostics and model fitting
#moved up to 0.78469 when inserted Sex, Title, and Embarked to the model



######################################################################
################ Logistic Regression Model  ##########################
######################################################################
#library(glmnet) has a number of glm applications, including lasso, ridge, etc.  Link here: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

#Logisitic Regression, and comparing logit, probit, cloglog
#Build the glm() model with family="binomial"
log_model <- glm(Survived ~ Pclass + Age + Sex + Embarked, family = binomial(link=logit), data=train)
library(modelr)
train <- add_residuals(train, log_model, var = "resid")
ggplot(train, aes(Survived, resid)) +
  geom_point()
log_probit <- glm(Survived ~ Pclass + Age + Sex + Embarked, family = binomial(link=probit), data=train)
log_clog <- glm(Survived ~ Pclass + Age + Sex + Embarked, family = binomial(link=cloglog), data=train)
      #Had to take Title out of the above model because there were titles in test, not present in train
#Predict probability of survival
log_predict <- predict(log_model, newdata = test, type = "response")
probit_predict <- predict(log_probit, newdata = test, type = "response")
clog_predict <- predict(log_clog, newdata = test, type = "response")
#look at the prediction values because we have to choose a cutoff
range(log_predict)
summary(log_predict)

#setting cutoff value at the mean value 0.37
log_predict <- ifelse(log_predict>0.37, 1, 0)

#then confusion matrix to compare accuracies
conf<- table(test$Survived, log_predict)
acc_logit <- sum(diag(conf)) / nrow(test)
#and doing it for the others
probit_predict <- ifelse(probit_predict>0.37, 1, 0)
conf<- table(test$Survived, probit_predict)
acc_probit <- sum(diag(conf)) / nrow(test)
clog_predict <- ifelse(clog_predict>0.37, 1, 0)
conf<- table(test$Survived, clog_predict)
acc_cloglog <- sum(diag(conf)) / nrow(test)

acc_logit
acc_probit  #Scores highest, but barely
acc_cloglog

#Comparing with ROC curve
library(pROC)
auc(test$Survived, log_predict)
auc(test$Survived, probit_predict) #does best here
auc(test$Survived, clog_predict)

#It appears as though the probit provides the best results from Accuracy and AUC, so we should get better results

#We should also start removing some of the observations to see if we get more informative results
log_remove <- glm(Survived ~ Age + Sex + Embarked, family = binomial(link=probit), data=train)
pred_remove_Pclass <- predict(log_remove, newdata = test, type="response")
log_remove <- glm(Survived ~ Pclass + Sex + Embarked, family = binomial(link=probit), data=train)
pred_remove_Age<- predict(log_remove, newdata = test, type="response")
log_remove <- glm(Survived ~ Pclass + Age + Embarked, family = binomial(link=probit), data=train)
pred_remove_Sex<- predict(log_remove, newdata = test, type="response")
log_remove_embark <- glm(Survived ~ Pclass + Age + Sex, family = binomial(link=probit), data=train)
pred_remove_Embark<- predict(log_remove, newdata = test, type="response")

#check the auc
auc(test$Survived, probit_predict)
auc(test$Survived, pred_remove_Sex)
auc(test$Survived, pred_remove_Embark)
auc(test$Survived, pred_remove_Age) #Really interesting that age removal results in higher AUC
auc(test$Survived, pred_remove_Pclass)

#We remove Embark and test with removing something else
log_remove <- glm(Survived ~ Pclass + Sex, family = binomial(link=probit), data=train)
pred_Emb_age<- predict(log_remove, newdata = test, type="response")
log_remove <- glm(Survived ~ Pclass + Age, family = binomial(link=probit), data=train)
pred_Emb_sex<- predict(log_remove, newdata = test, type="response")
log_remove_emb_class <- glm(Survived ~ Age + Sex, family = binomial(link=probit), data=train)
pred_Emb_class<- predict(log_remove_emb_class, newdata = test, type="response")

auc(test$Survived, pred_remove_Embark)
auc(test$Survived, pred_Emb_sex)
auc(test$Survived, pred_Emb_age)
auc(test$Survived, pred_Emb_class)
#We get the best AUC by removing Embarked and Age

#Use the whole training dataset to get more accurate stats
log_model <- glm(Survived ~ Pclass + Sex + Age, family = binomial(link=probit), data=train_clean)
pred <- predict(log_remove_embark, newdata = test_clean, type="response")
###Build for submission
submission <- data.frame(PassengerId = test_clean$PassengerId)
#Set the cutoff
submission$Survived <- ifelse(pred>0.37, 1, 0)
head(submission)

#Write to csv for upload
write.csv(submission, file = "titanic_submission.csv", row.names = FALSE)

###############  Estimating Performance measures and cutoffs for logistic regressions
library(ROCR)
#https://www.r-bloggers.com/a-small-introduction-to-the-rocr-package/
#Should look into it further

######################################################################
####################### Support Vector Machines  #####################
######################################################################



library(kernlab)
svp <- ksvm(train$Sex,train$Survived,type="C-svc",kernel="vanilladot",C=100,scaled=c())


######################################################################
####################### k-means clustering  ##########################
######################################################################
#This really is not the best method for determining what is going on here
    #We know there are two groups, but this method is figuring out how many groups there are


set.seed(42)

#Have to restrict the data down to just the explainers
a <- train %>%
  mutate(Male = ifelse(Sex == "male", 1, 0)) %>%
  select(Pclass, Male, Age, SibSp, Parch)
#use kmeans() to group into 2 groups
km <- kmeans(a, centers = 2, nstart=20)
km
km_conf <- table(km$cluster, train$Survived)
sum(diag(km_conf)) / sum (km_conf)
#Doesn't provide a great accuracy


#Create a scree plot to see how many groups there are

# Initialise ratio_ss 
ratio_ss <- rep(0, 7)
#Write the for loop depending on k. 
for (k in 1:7) {
  # Apply k-means to a
  km <- kmeans(a, k, nstart = 20) 
  # Save the ratio between of WSS to TSS in kth element of ratio_ss
  ratio_ss[k] <- km$tot.withinss / km$totss
}

# Make a scree plot with type "b" and xlab "k"
plot(ratio_ss, type = "b", xlab = "k")
#Plot shows there are 3 or 4 groups in the data

    #Maybe I take those three or four groups and assign it as a classifier?

#lets try that now on a random forest

######################################################################
################## Clustering as a Classifier ########################
######################################################################

#Set at 4 groups because thats what the scree plot showed worked
km <- kmeans(a, centers = 4, nstart=20)
#cbind back with the train data
train_clustered <- cbind(train,Cluster= km$cluster)

#create the groups in the test data
library(clue)
b <- test %>%
  mutate(Male = ifelse(Sex == "male", 1, 0)) %>%
  select(Pclass, Male, Age, SibSp, Parch)
km_predict <- cl_predict(km, newdata = b, type = "class_ids")
test_clustered <- cbind(test, Cluster = as.vector(km_predict))

#Run the random forest model
#Created the forest model object
forest_model <- randomForest(as.factor(Survived) ~ Pclass + Age + Title + Sex + Embarked + Cluster,
                             data = train_clustered, 
                             importance = T,
                             ntree= 1000)
rf_prediction <- predict(forest_model, newdata=test_clustered)
#look at the plot to see what is more important
varImpPlot(forest_model)
#Building the submission
conf_rf <- table(test$Survived, rf_prediction)
conf_rf
acc_rf_cluster <- sum(diag(conf_rf)) / sum(conf_rf)
#Compare to the base rf model
acc_rf
acc_rf_cluster  #Shit! it actually improves!
      #However was slightly lower on Kaggle - I don't like that it only shows on 1/2 the data

#################################
##########Now predict it with the full data for submission

a <- train_clean %>%
  mutate(Male = ifelse(Sex == "male", 1, 0)) %>%
  select(Pclass, Male, Age, SibSp, Parch)
ratio_ss <- rep(0, 7)
for (k in 1:7) {
  km <- kmeans(a, k, nstart = 20) 
  ratio_ss[k] <- km$tot.withinss / km$totss
}
plot(ratio_ss, type = "b", xlab = "k")
#Set at 4 groups because thats what the scree plot showed worked
km <- kmeans(a, centers = 4, nstart=20)
#cbind back with the train data
train_clustered <- cbind(train_clean,Cluster= km$cluster)

#create the groups in the test data
library(clue)
b <- test_clean %>%
  mutate(Male = ifelse(Sex == "male", 1, 0)) %>%
  select(Pclass, Male, Age, SibSp, Parch)
km_predict <- cl_predict(km, newdata = b, type = "class_ids")
test_clustered <- cbind(test_clean, Cluster = as.vector(km_predict))

#Run the random forest model
forest_model <- randomForest(as.factor(Survived) ~ Pclass + Age + Title + Sex + Embarked + Cluster,
                             data = train_clustered, 
                             importance = T,
                             ntree= 1000)
rf_prediction <- predict(forest_model, newdata=test_clustered)

######################################################################
################ Using the "caret" package  ##########################
######################################################################
library(caret)

#using caret for partial least squares discriminant analysis
rfFit <- train(as.factor(Survived) ~ Pclass + Age + Title + Sex + Embarked,
                data = train,
                method = "rf")
caret_predict <- predict(rfFit, newdata = test)  #6 are classified different from above, maybe not specifying the parameters

rpartFit <- train(Survived ~ Pclass + Age + Title + Sex + Embarked,
                  data = train,
                  method = "rpart")
prediction <- predict(rpartFit, newdata = test)

tree <- rpart(Survived ~ Pclass + Sex + Age + Parch + Fare + Embarked, data = train, method = "class")
plot(tree, uniform = T)
text(tree)

#predict values
tree_predict <- predict(tree, newdata = test, type = "class")
#Confusion Matrix - 
conf_tree <- table(test$Survived, tree_predict)
conf_tree
acc_tree <- sum(diag(conf_tree)) / sum(conf_tree)
acc_tree

######################################################################
################################# BMA for predictions ################
######################################################################
library(BMA)

train_subset <- select(train, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)
bma_model <- bicreg(train_subset,train$Survived)
bma_predictions <- predict(bma_model, newdata = test)
#specify cutoff
summary(bma_predictions$mean) #choose the mean value of 0.37
bma_predict <- ifelse(bma_predictions$mean>0.45, 1, 0) #and it got bumped up to the 0.45
conf_tree <- table(test$Survived, bma_predict)
conf_tree
acc_tree <- sum(diag(conf_tree)) / sum(conf_tree)
acc_tree

#Build for submission now on the full data using the same cutoff point
train_subset <- select(train_clean, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)
bma_model <- bicreg(train_subset,train_clean$Survived)
bma_predictions <- predict(bma_model, newdata = test_clean)
imageplot.bma(bma_model) #can see that it breaks almost all the variables into dummies
summary(bma_predictions$mean) #summary looks basically the same as above, now cutoff the values
bma_predict <- ifelse(bma_predictions$mean>0.45, 1, 0)

#I am really surprised by the effectiveness of BMA. It actually did a pretty decent job of fitting the observations



######################################################################
########################## Ensembling Predictions ####################
######################################################################
#idea came from:  http://rpubs.com/Vincent/Titanic
      #And more applications can be found here: http://mlwave.com/kaggle-ensembling-guide/
      #And library(caretEnsemble): https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
#simple weighting scheme where you average the results and see where it falls
        #although I'm not 100% sure why we subtract the one from the predictions
ensemble <- as.numeric(predict_1) + as.numberic(predict_2)-1 + as.numeric(predict_3)-1
#then average the estimates
ensemble <- sapply(ensemble/3, round) #and here is the output


######################################################################
################## Feature Selection - Boruta ########################
######################################################################
#file:///home/chase/Downloads/v36i11.pdf

#Check the importance of the variables through this Boruta process
    #Creates shadow variables to test against. Important variables should be more important that the most important shadoww
library(Boruta)

borutaTest <- Boruta(Survived ~ ., data=train, doTrace=1, ntree=500) #running the Boruta algorithm
          #Can take quite a bit of time if the dataframe is large
borutaTest
plot(borutaTest2)
attStats(borutaTest)  #Shows the Z-score statistics and the fraction of RF runs that it was more important than the most important shadow run


borutaTest2 <- Boruta(Survived ~ ., data=train, doTrace=1,maxRuns=100)
plot(borutaTest2)
#############
######################################################################
################## Build file for submission #########################
######################################################################

submission <- data.frame(PassengerId = test_clean$PassengerId, Survived = prediction)
head(submission)
write.csv(submission, file = "titanic_submission.csv", row.names = FALSE)

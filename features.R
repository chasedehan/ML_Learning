#Feature Selection.R

#Boruta
#LASSO and ridge
#Stepwise Regressions

#from Machine Learning Mastery
    #http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
library(mlbench)
library(caret)

set.seed(3)

str(train)

############################ Remove highly correlated observations
correlationMatrix <- cor(train[,1:(ncol(train)-1)])
correlationMatrix
#Find those that are highly correlated 
high_cor <- findCorrelation(correlationMatrix, cutoff = 0.75)
high_cor
#now pull out the one that is the most important


############################ Ranking by importance and cutting 
    #Learning Vector Quantization (LVQ) - variable importance

#Build the model
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
lvq_model <- train(loss ~ ., data = train, method = "lvq", preProcess = "scale", trControl = control)
#Variable importance
importance <- varImp(lvq_model, scale = FALSE)
importance
plot(importance)


###### Feature selection using a randomForest
    #The plan is to run through a randomForest model and CV the accuracy
#use random forest selection function
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
#Run RFE algorithm
results <- rfe(train[,1:ncol(train)-1], y, rfeControl = control)
results
predictors(results)
#Plot out the results of the feature selection to see how well the model does with a certain number of variables
plot(results, type=c("g","o"))

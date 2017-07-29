#Predicting sale price of houses - Kaggle

#There is the large possibility of feature engineering and using ensemble methods.
#My approach will be:
    #Pre process the data quite a bit to find/fit some appropriate variables
    #Run through a large number of models (kitchen sink)
        #Cross-validating
    #Compute the RMSE on each model to find the optimal model for each type
    #Use some ensembling methods to weight them differently.

######################################################################
################## TODO with the housing data ################
######################################################################
    #Optimizing the model selection and variable manipulation
    #Stepwise Regression, AIC, BIC, 
    #Chapter 6: ridge regression, lasso, principal components
    #Chapter 7: non-linear regressions
        #polynomial
        #step functions
        #Regression Splines
        #Smoothing Splines
        #Local Regressions
        #Generalized Additive Models



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

train <- read.csv("house_train.csv", stringsAsFactors=FALSE)
test  <- read.csv("house_test.csv",  stringsAsFactors=FALSE)


######################################################################
############## Feature Engineering ###################################
######################################################################

#Need to build out a number of features as the data is incredibly rich
    
#Need to control for the year - trend? dummy?

#imput LotFrontage, LotArea

#combine, manipulate, and separate data
#Need to merge the training and test data together, but test doesn't have a "Survived" column, so we create it
test$SalePrice <- NA
#merge test and train
all_data <- data.frame(rbind(train, test))
#We can also split the title out of the name 
all_data$MSZoning[is.na(all_data$MSZoning)] <- "RL"
all_data$Utilities[is.na(all_data$Utilities)] <- "AllPub"
all_data$BsmtFullBath[is.na(all_data$BsmtFullBath)] <- 0
all_data$BsmtHalfBath[is.na(all_data$BsmtHalfBath)] <- 0
all_data$GarageCars[is.na(all_data$GarageCars)] <- 0
all_data$GarageArea[is.na(all_data$GarageArea)] <- 0
all_data$BsmtFinSF1[is.na(all_data$BsmtFinSF1)] <- 0
all_data$BsmtFinSF2[is.na(all_data$BsmtFinSF2)] <- 0
all_data$BsmtUnfSF[is.na(all_data$BsmtUnfSF)] <- 0
all_data$TotalBsmtSF[is.na(all_data$TotalBsmtSF)] <- 0
all_data$KitchenQual[is.na(all_data$KitchenQual)] <- 0
all_data$SaleType[is.na(all_data$SaleType)] <- "WD"
all_data$Exterior1st[is.na(all_data$Exterior1st)] <- "VinylSd"
all_data$Exterior2nd[is.na(all_data$Exterior2nd)] <- "VinylSd"
#create some dummy variables
all_data <- all_data %>%
  mutate(PavedStreet = ifelse(Street == "Pave", 1, 0),
         PavedAlley = ifelse(is.na(Alley) | Alley != "Pave", 0, 1),
         GrvlAlley = ifelse(is.na(Alley) | Alley != "Grvl", 0, 1),
         HouseAge = YrSold - YearBuilt,
         Basement = ifelse(TotalBsmtSF>0, 1, 0),
         FinBsmt = ifelse(BsmtFinSF1>0, 1, 0),
         AC = ifelse(CentralAir=="Y",1,0),
         TwoStory = ifelse(X2ndFlrSF>0,1,0),
         Baths = BsmtFullBath + 0.5*BsmtHalfBath + FullBath + 0.5*HalfBath,
         Pool = ifelse(PoolArea >0, 1, 0),
         Shed = ifelse(is.na(MiscFeature), 0, ifelse(MiscFeature =="Shed", 1, 0)),
         Fence = ifelse(is.na(Fence), 0, 1),
         Carport = ifelse(is.na(GarageType), 0, ifelse(GarageType == "CarPort", 1, 0)),
         AttGarage = ifelse(is.na(GarageType), 0, ifelse(GarageType == "Attchd" | GarageType == "BuiltIn", 1, 0)),
         DetGarage = ifelse(is.na(GarageType), 0, ifelse(GarageType == "Detchd", 1, 0)),
         MasVnrType = ifelse(is.na(MasVnrType), "None", MasVnrType),
         MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea),
         Electrical = ifelse(is.na(Electrical), "SBrkr", Electrical),
         YrSold = as.factor(YrSold)) %>%
  select(-c(Street, Alley, YearBuilt, MiscFeature, LotFrontage, PoolQC, FireplaceQu, CentralAir),
         -c(BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2),
         -c(GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond), - Functional)

#add, 50 years or greater

#Split the data back up
test<- all_data[is.na(all_data$SalePrice),]
train <- all_data[!is.na(all_data$SalePrice),]
test$SalePrice <- NULL
rm(all_data)

#Imput some values into the test data for simplicity right now
######################################################################
################## Feature Selection - Boruta ########################
######################################################################
#file:///home/chase/Downloads/v36i11.pdf
library(Boruta)

#Run the Boruta test to see what should be important
borutaTest <- Boruta(SalePrice ~ ., data=train, doTrace=1, ntree=500)
    #One reason is that if certain things look important we will spend more time on imputing the values
borutaTest  #A couple tentative
boruta2 <- TentativeRoughFix(borutaTest) #Rough fixed it to bring in only those relevant
plot(borutaTest)
attStats(borutaTest) 
borutaTest$impSource
borutaNames <- getSelectedAttributes(boruta2) #assign the confirmed important variables to vector

borutaNames <- c(2,3,4,5,6,9,10,11,13,14,15,16,17,20,21,22,23,24,26,27,29,30,32,33,34,35,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,61,64,66,67,68,69,70,71,74,75,76)
borutaNames <- match(borutaNames, names(train)) #returns vector column indexes where borutaNames match the names of "train"
train_subset<- select(train, Id, SalePrice, YrSold, borutaNames)
train_subset <- select(train_subset, -KitchenQual, -Exterior1st, -Exterior2nd, -HeatingQC, -Electrical)  
which(names(train) == "SalePrice")
borutaNames_test <- ifelse(borutaNames < which(names(train) == "SalePrice"), borutaNames, borutaNames - 1)
test_subset <- select(test, Id, YrSold, borutaNames_test)
test_subset <- select(test, -KitchenQual, -Exterior1st, -Exterior2nd, -HeatingQC, -Electrical)  
#These are all factors with levels that don't show up in predict, go back through and fix these
train_subset$Electrical <- NULL
######################################################################
################## Linear Regression ########################
######################################################################
str(train_subset)

regModel <- lm(SalePrice ~ ., data=train_subset)
summary(regModel)

lm_predictions <- predict(regModel, newdata = train_subset)
    #has predictions from a rank-deficient fit may be misleading
    #A lot of predictors, most are insignificant, high F statistic
plot(lm_predictions, train_subset$SalePrice)
abline(0,1)
    #On visual inspection, it actually looks like a decent job
    #The line is centered on the lower priced homes, but is skewed upward on the higher priced



######################################################################
################## Ridge and Lasso Regressions #######################
######################################################################
#https://rstudio-pubs-static.s3.amazonaws.com/54576_142b58255f8944c990c5663290d28517.html
library(glmnet)

##############################################    Ridge Regression
x<- model.matrix(log(SalePrice) ~ ., train_subset)
y <- train_subset$SalePrice
#Not sure why I'm doing the below line
#grid <- 10^seq(10,-2, length = 100)
ridge_model <- glmnet(x, y, alpha = 0)              
#The plot shows lambda as it converges toward zero, picking the lambda requires CV
plot(ridge_model, xvar="lambda", label=TRUE)
ridge_pred <- predict(ridge_model, s = 12, newx = x) 
plot(ridge_pred, train_subset$SalePrice)
abline(0,1)
#computing MSE
mean((ridge_pred - y)^2)
#CV to find appropriate lambda
cv_ridge <- cv.glmnet(x, y, alpha=0)
plot(cv_ridge)
best_lam_r <- cv_ridge$lambda.min
best_lam_r

ridge_cvpred <- predict(ridge_model, s = best_lam, newx = x)
plot(ridge_cvpred, train_subset$SalePrice) 
abline(0,1)
mean((ridge_cvpred - y)^2)


##############################################    Lasso Regression
lasso_model <- glmnet(x, y) #change alpha to 1 for the lasso
plot(lasso_model, xvar="lambda", label=TRUE)

cv_lasso <- cv.glmnet(x, y) #default alpha = 1
plot(cv_lasso)
best_lam <- cv_lasso$lambda.min
lasso_pred <- predict(lasso_model, s = log(best_lam), newx  = x)  #the log of best_lam does predict better than otherwise
plot(lasso_pred, train_subset$SalePrice) 
abline(0,1)
mean((lasso_pred - y)^2)

##############################################  Principal Component Analysis

######################################################################
########################## Stacking (averaging estimates) ############
######################################################################

#Create estimates using the average value of fwd stepwise, lasso, and ridge

library(caret)
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)
lm_caret <- train(log(SalePrice) ~ ., data = train_subset,
                    method = "lm")
lm_pred <- exp(predict(ridge_mod2, newdata = train_subset))
plot(lm_pred, train_subset$SalePrice)
abline(0,1)
mean((pred - train_subset$SalePrice)^2)

#Changing the data to "test" and averaging with fwd_pred
lm_pred <- exp(predict(ridge_mod2, newdata = test))
avg_pred <- (fwd_pred + lm_pred) / 2  #This lowered the accuracy of the first stepwise regression submission


######################################################################
########################## Random Forest ###############################
######################################################################
#"treebag" doesn't seem to do so well

#so we will use a random forest - "ranger"
rf_mod <- train(log(SalePrice) ~ ., data = train_subset,
                 method = "ranger")
rf_pred <- exp(predict(rf_mod, newdata = train_subset))
plot(rf_pred, train_subset$SalePrice)
abline(0,1)
mean((rf_pred - train_subset$SalePrice)^2)

rf_pred <- exp(predict(rf_mod, newdata = test))
submit(rf_pred)
    #The RMSE was 0.14184, which was worse than 

#average with fwd_pred
avg_pred <- (fwd_pred + rf_pred) / 2 
submit(avg_pred)

    #Improved by 0.00316 by averaging --> RMSE = 0.13086 

######################################################################
########################## Splines ###################################
######################################################################
library(splines)
spline_mod <- lm(SalePrice ~ bs(HouseAge, df = 3), data = train_subset)
plot(train_subset$SalePrice,predict(spline_mod, newdata = train_subset))
abline(0,1, col="red")
plot(train_subset$SalePrice, train_subset$HouseAge)

######################################################################
########################## Generalized Additive Models ###############
######################################################################
#fitting a natural spline for Generalized Additive Models

gam <- lm(SalePrice ~ ns(HouseAge, 4) + ns(FullBath), data = train_subset)
plot(train_subset$SalePrice,predict(gam, newdata = train_subset))
abline(0, 1, col = "red")
library(gam)
gam2 <- gam(SalePrice ~ s(HouseAge, 4) + s(FullBath), data = train_subset)
plot(train_subset$SalePrice,predict(gam2, newdata = train_subset))

###caret and GAM with splines
gam_mod <- train(SalePrice ~ ., data = train_subset,
                method = "gam")
gam_pred <- exp(predict(gam_mod, newdata = train_subset))
plot(rf_pred, train_subset$SalePrice)
abline(0,1)

##################################################3
###############  Stepwise Regression

#Build the minimum model and the largest
min_model <- lm(log(SalePrice) ~ 1, data = train_subset)
biggest <- formula(lm(log(SalePrice) ~ ., data=train_subset))
biggest
fwd_model <- step(min_model, direction = "forward", scope=biggest)
fwd_model
fwd_pred <- exp(predict(fwd_model, data= train_subset))
graph_exp(fwd_model)
plot(fwd_pred, train_subset$SalePrice)
abline(0,1)
mean((fwd_pred - train_subset$SalePrice) ^2 )
#Looks pretty similar to the data from lm() 
#But tails off on higher prices - trying log transform on this go through
#lets put fwd_pred into Kaggle
fwd_pred <- exp(predict(fwd_model, newdata = test)  )

#Naive approach, no transformations:
#RMSE = 0.16234 for no log, beating the naive lm() model - as it should.
#Transforming with log()
#RMSE = 0.13402, doing really well!
######################################################################
############### caret training models ####################
######################################################################
#using caret
library(caret)
y <- train_subset$SalePrice


set.seed(3)
control <- trainControl(method = "cv", number = 10)
rf_fit <- train(log(SalePrice) ~ ., data = train_subset, method = "ranger")

rf_pred <- graph_exp(rf_fit)

gbm_fit <- train(log(SalePrice) ~ ., data = train_subset, method = "gbm", trControl = control, verbose = FALSE)
gbm_pred <- graph_exp(gbm_fit)

xnn <- data.frame(fwd_pred, rf_pred, gbm_pred)
xnn_rank <- t(apply(xnn,1,rank))
colnames(xnn_rank) <- paste0("rank_",names(xnn))
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=1,
                                 verboseIter=FALSE)
CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                maximize=FALSE,
                                tuneGrid=NULL,
                                tuneLength=7,
                                metric="RMSE",
                                method = "nnet")
MODEL.SPECIFIC.PARMS <- list(verbose=FALSE,linout=TRUE,trace=FALSE) 
l1_nnet_mdl <- do.call(train,c(list(x=xnn,y=y),
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))


nnet_pred <- predict(nnet_fit, newdata = train_subset)

View(xnn)
nnet_fit <- train(y ~ ., data = xnn, method = "nnet", maxit = 1000, verbose = FALSE, linout = TRUE, trace = FALSE )


#step_fit <- train(log(SalePrice) ~ ., data = train_subset, method = "lmStepAIC")
#graph_exp(step_fit)




######################################################################
########################## Ensembling from Kaggle ####################
######################################################################
# https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/ensemble-model-stacked-model-example


# http://machinelearningmastery.com/machine-learning-ensembles-with-r/
library(caret)
library(caretEnsemble)

#Use Gradient Boosting First
set.seed(3)
control <- trainControl(method = "cv", number = 10)
gbm_fit <- train(log(SalePrice) ~ ., data = train_subset, method = "gbm", trControl = control, verbose = FALSE)
attributes(gbm_fit)
summary(gbm_fit$finalModel)

gbm_pred <- exp(predict(gbm_fit, newdata = train_subset))
plot(gbm_pred, train_subset$SalePrice)
abline(0,1)
gbm_pred <- exp(predict(gbm_fit, newdata = test_subset))
submit(gbm_pred)
      #Beats RF model on Kaggle --> 0.13980 (without log())
      #with log(), it become 0.13404

#bagged CART --> "treebag"
tbag_fit <- train(SalePrice ~ ., data = train_subset, method = "treebag", metric = metric, trControl = control)
tbag_pred <- predict(tbag_fit, newdata = train_subset)
plot(tbag_pred, train_subset$SalePrice)
abline(0,1)
tbag_pred <- predict(tbag_fit, newdata = test_subset)
submit(tbag_pred)
      #This one did really bad - 0.21312, but I should have expected that with the way the plot looked

#stepwise in caret
step_fit <- train(log(SalePrice) ~ ., data = train_subset, method = "lmStepAIC", metric = metric)
step_pred <- predict(step_fit, newdata = train_subset)
plot(step_pred, train_subset$SalePrice)
abline(0,1)

#########################################################
#Using caretEnsemble
algorithm_list <- c("gbm", "ranger")
model_list <- caretList(log(SalePrice) ~ ., data = train_subset, methodList = algorithm_list)
      #caretList generates the model fits for each algorithm passed into it
results <- resamples(model_list)
summary(results)

#can generate the predictions off model_list
p <- as.data.frame(exp(predict(model_list)), newdata = train_subset)
#checking for correlations between the models:

modelCor(results)
splom(results)

##Simple Greedy optimizatoin on AUC
greedy_ensemble <- caretEnsemble(model_list, metric = "RMSE")
greedy_pred <- exp(predict(greedy_ensemble, newdata = train_subset))
plot(greedy_pred, train_subset$SalePrice)
abline(0,1, col = "red")
#stack using simple linear model
stack_model <- caretStack(models, method = "glm", metric = "ROC", 
                          trControl = trainControl(
                            method = "boot",
                            number = 10,
                            savePredictions = "final"
                            )
                          )
stack_pred <- predict(stack_model, newdata = train_subset)
plot(stack_pred, train_subset$SalePrice)
abline(0,1)


######################################################################
################## Build file for submission #########################
######################################################################
submit <- function(x) {
  submission <- data.frame(Id = test$Id, SalePrice = x)
  head(submission)
  write.csv(submission, file = "house_submission.csv", row.names = FALSE)
}
  
  
  

######################################################################
################## predict and graph predictions #####################
###################################################################### 
graph_exp <- function(x) {
  lm_pred <- exp(predict(x, newdata = train_subset))
  plot(y, lm_pred)
  abline(0,1,col="red")
  return(lm_pred)
}

######################################################################
################## Calculate RMSE and MAE ############################
######################################################################
#Below is how we compute RMSE, but this doesn't need to be the only metric we use to calculate it.
    #From: https://heuristically.wordpress.com/2013/07/12/calculate-rmse-and-mae-in-r-and-sas/

# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Example data
actual <- c(4, 6, 9, 10, 4, 6, 4, 7, 8, 7)
predicted <- c(5, 6, 8, 10, 4, 8, 4, 9, 8, 9)

# Calculate error
error <- actual - predicted
#for decision tree
error <- log(train$SalePrice) - log(greedy_pred)
error_rf <- log(train$SalePrice) - log(rf_pred)
error_lm <- train$SalePrice - predict(regModel, newdata=train_subset)
# Example of invocation of functions
rmse(error)
rmse(error_rf)
mae(error)
  
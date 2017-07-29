#Allstate.R

#TODO
  #Check for correlations once the variables have been converted to the model.matrix()  
  #I wonder if I should be training it on all the observations rather than cutting out the large losses
          #Maybe create another feature with the probability of being a large loss > 35000
          #The random forest probably would perform better --> try that next
    #I also want to try running a ridge on the x_subset data because of the apparently better predictive power.
    #Work on ensembling the methods a little better

setwd("/home/chasedehan/Dropbox/R Code/ML/Allstate")

library(pryr)
library(tidyverse)

library(modelr)
#library(caret)

######################################################################
########## Checking out the data/ prelim look ########################
######################################################################

#importing the data
train <- read_csv("train.csv")
test  <- read_csv("test.csv")
trainId <- train$id
testId <- test$id
y <- train$loss
train$loss <- train$id <- test$id <- NULL

#Boruta first pass
  #See down at the bottom for running the Boruta, but we did extract the appropriate variables
borutaNames <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,16,23,25,26,27,28,29,30,36,37,38,40,42,43,44,45,49,50,51,52,53,57,60,61,66,72,73,75,76,79,80,81,82,84,87,88,89,90,91,92,94,95,96,97,98,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130)
train <- select(train, borutaNames) 
test  <- select(test, borutaNames)

#Convert the categorical variables to dummies
    #model.matrix spreads the data across the dimensions, creating a design matrix
x <- model.matrix(~ ., train)  #creates a pretty large matrix object ~1.4GB, a lot of overhead, but due to ~900 variables
x <- as.data.frame(x)
rm(train)
testX <- model.matrix(~ ., test)
testX <- as.data.frame(testX)
rm(test)
testX$`(Intercept)` <- x$`(Intercept)` <- NULL  #Have to remove the intercept from the data generate through model.matrix

#Want to take out the categorical dummies with only a few observations
    #we only do it on the training dataset because that is what the model is built on
    #we first sum each column and then delete those variables that show up in less than 0.5%
a <- colSums(x)
a <- as.data.frame(sort(a))
names(a)[1] <- "count"
a <- mutate(a, varname = rownames(a)) #need the rownames because filter() below will chop them off

#Now cut off the observations that don't meet
cutoff <- 0.002*nrow(x) #376.636
    #doing count%%1 != 0 because I don't want to cut the continuous variables
b <- filter(a, count > cutoff | count%%1 != 0) #goes from 845 to 357, keeps all continuous
head(b$varname)


#pull out the appropriate variable name and select() to keep that particular variable
keepVars <- match(b$varname, names(x))
sort(keepVars)
x<- select(x, keepVars)
keepVarsTest <- match(b$varname, names(testX))
sort(keepVarsTest)
testX <- select(testX, keepVarsTest)
rm(a,b)

#narrow down the matrix to include only the variables that are found in both datasets
    #if necessary
all.equal(names(x), names(testX)) #So not necessary in this instance
#x <- x[,names(x) %in% names(testX)]
#testX <- testX[,names(testX) %in% names(x)]


#For this instance, we want to remove those observations from the training data with losses greater than $40k
x <- cbind(trainId, y, x)
x <- filter(x, y < 35000)
y <- x$y
trainId <- x$trainId
x$y <- x$trainId <- NULL

#convert back into a matrix to run through the LASSO
x <- as.matrix(x)
testX <- as.matrix(testX)

######################################################################
########## Linear Model as First Step w/ All Features ################
######################################################################
#x <- cbind(y,x)

#regModel <- lm(y ~ ., data=x)
#plotting out the residuals

#x <- x %>%
#  add_residuals(regModel, "resid")
#ggplot(x, aes(y, resid)) +
#  geom_hex(bins=50)
      #we see large bias in the residuals

library(lmtest) #to test and see what is going on exactly
library(sandwich)
coeftest(regModel, vcov = sandwich)
######################################################################
################## LASSO for predictions and model selection #########
######################################################################
#https://rstudio-pubs-static.s3.amazonaws.com/54576_142b58255f8944c990c5663290d28517.html
#Starting off using glmnet on the estimations, but could also use lm.ridge() from MASS, or the "penalized" package

library(glmnet)  # uses mean-squared errors for model fitting

#Building the LASSO for prediction
lasso_model <- glmnet(x, y, alpha = 1)
plot(lasso_model, xvar="lambda", label=TRUE) # I like 3
lasso_pred3 <- as.numeric(predict(lasso_model, s = 3, newx  = x)  )
plot_pred(lasso_pred3)


#lets check the lasso with log(y)
lasso_model <- glmnet(x, log(y), alpha = 1)
plot(lasso_model, xvar="lambda", label=TRUE)
lasso_pred_log <- as.numeric(exp(predict(lasso_model, s = -5, newx  = x)))
plot_pred(lasso_pred_log)

lasso_pred_test <- as.numeric(exp(predict(lasso_model, s = -5, newx  = testX)))

#cv the lasso --> seem to get almost the same result whether I am using cv or visual inspection
cv_lasso <- cv.glmnet(x, log(y)) #default alpha = 1
plot(cv_lasso)
best_lam <- cv_lasso$lambda.min
best_lam
lasso_pred <- as.numeric(exp(predict(lasso_model, s = best_lam, newx  = x) ) )
lasso_pred2 <- as.numeric(exp(predict(lasso_model, s = log(best_lam), newx  = x) ) )
plot_pred(lasso_pred)
plot_pred(lasso_pred2)
######################################################################
########## Extraction of important variables from the LASSO ##########
######################################################################
#We went through the process of using the LASSO to narrow down the number of var

#extracting coefficients from the model 
a <- coef.glmnet(lasso_model, s=exp(-4))
a <- as.data.frame(cbind(coef = a[,1], varname = rownames(a)))
a <- a[-1,] #deleting the intercept
a <- filter(a, coef != 0)
#Unfortunately we have to convert the matrix back to a dataframe for the next step
x<-as.data.frame(x)
keepVars <- match(as.character(a$varname), names(x))
x_subset <- select(x, keepVars) 
x_subset <- cbind(y, x_subset)
rm(a)

######################################################################
########## Random Forest Model #######################################
######################################################################
#running a RF on the subset data from above
library(ranger)
rf_model <- ranger(y ~ ., data = x_subset, write.forest = TRUE) #Ok, that didn't take too long
    #could add the importance="permutation" to calculate the importance, but computationally takes some time.
rf_pred <- predict(rf_model, data = x_subset)
plot_pred(predictions(rf_pred))


######################################################################
########## Gradient Boosting Attempt #################################
######################################################################
control <- trainControl(method = "cv", number = 10)
gbm_fit <- train(y ~ ., data = x_subset, method = "gbm", trControl = control, verbose = FALSE)
gbm_pred <- predict(gbm_fit, newdata = x_subset)
plot_pred(gbm_pred)

gbm_test_pred <- predict(gbm_fit, newdata = testX)
##################################################3
###############  Stepwise Regression

#Build the minimum model and the largest
min_model <- lm(log(y) ~ 1, data = x_subset)
biggest <- formula(lm(log(y) ~ ., data=x_subset))
biggest
fwd_model <- step(min_model, direction = "forward", scope=biggest)
fwd_model
fwd_pred <- exp(predict(fwd_model, data= x_subset))
plot_pred(fwd_pred)

fwd_test_pred <-  exp(predict(fwd_model, newdata= testX))
#Looks pretty similar to the data from lm() 
#But tails off on higher prices - trying log transform on this go through
#lets put fwd_pred into Kaggle
fwd_pred <- exp(predict(fwd_model, newdata = test)  )
######################################################################
########## Prep for Submission #######################################
######################################################################

#lets just submit the subset RF model first
rf_test_pred <- predictions(predict(rf_model, data = testX))
summary(rf_test_pred)
submit(rf_test_pred)

#average the log_lasso with rf
mean_pred <- (lasso_pred_test + rf_test_pred) / 2
submit(mean_pred)

#average rf, log_lasso, fwd_reg, and gbm
mean_pred <- (lasso_pred_test + rf_test_pred + fwd_test_pred + gbm_test_pred) / 4
submit(mean_pred) #did worse than the average above

submit(lasso_pred_test) #just submit this one

#The lesson I have from here is that the ensembling does work
    #rf_subset scored 1203.93
    #LASSO_log scored 1245.13
        #But averaging the two of them scored:  1186.22!!!

######################################################################
########## Feature Selection - Boruta (too intensive for dataset) ####
######################################################################
#need to sample the data to shorten the run time
sample_size <- floor(0.1 * nrow(train))
set.seed(3)
sample <- sample(seq_len(nrow(train)), size = sample_size)
train_boruta <- train[sample,]
#file:///home/chase/Downloads/v36i11.pdf
library(Boruta)
borutaTest <- Boruta(loss ~ ., data=train_boruta, doTrace=1, maxRuns = 101)
#One reason is that if certain things look important we will spend more time on imputing the values
borutaTest  #A couple tentative

boruta2 <- TentativeRoughFix(borutaTest) #Rough fixed it to bring in only those relevant
plot(borutaTest)
attStats(borutaTest) 
borutaTest$impSource
borutaNames <- getSelectedAttributes(boruta2) #assign the confirmed important variables to vector
borutaNames <- match(borutaNames, names(train)) #returns vector column indexes where borutaNames match the names of "train"

borutaNames <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,16,23,25,26,27,28,29,30,36,37,38,40,42,43,44,45,49,50,51,52,53,57,60,61,66,72,73,75,76,79,80,81,82,84,87,88,89,90,91,92,94,95,96,97,98,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130)

train <- select(train, borutaNames) #leaves only predictors in the dataframe
test  <- select(test, borutaNames)

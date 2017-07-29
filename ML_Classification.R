#Logit, LDA, QDA, and KNN
#Lab work for "An Introduction to Machine Learning" Ch 4. Classification


#on Home
setwd("/home/chase/Dropbox/R Code/ML")
#on Laptop
setwd("/home/chasedehan/Dropbox/R Code/ML")
#on Work
setwd("C:/Users/cdehan/Dropbox/R Code/ML")

library(ISLR)
data <- Smarket
head(data)
summary(data)

#create train/test data
train <- data[data$Year<2005,]
test <- data[data$Year>=2005,]


############# Logistic Regression
glm_model <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, family = "binomial", data = train)
summary(glm_model)
glm_prob <- predict(glm_model, type="response", newdata = test)
glm_pred <- ifelse(glm_prob>0.5, "Up", "Down")
table(glm_pred, test$Direction)
mean(glm_pred==test$Direction) #48%

############## Linear Discriminant Analysis
#using the same data
library(MASS)
lda_model <- lda(Direction ~ Lag1 + Lag2, data=train)
lda_model
#Observations: Down 49.2%, Up 50.8%

lda_pred <- predict(lda_model, test)
lda_pred$class  #Presents the prediction classed at the .5 posterior probability
lda_pred$posterior #provides the posterior probabilities
    #We can use the posterior probability if we wish to only trade on days we are more certain of a change in price
table(lda_pred$class, test$Direction)
mean(lda_pred$class==test$Direction)


################  Quadratic Discriminant Analysis
qda_model <- qda(Direction ~ Lag1 + Lag2, data=train)
qda_model
qda_pred <- predict(qda_model, newdata = test)
table(qda_pred$class, test$Direction)
mean(qda_pred$class == test$Direction)
      #Does a much better job than LDA or logit so might fit the data a little better.


################  K-Nearest Neighbors
library(class)
#forms predictions with one command, rather than fitting a model and then predicting
train_mat <- cbind(train$Lag1, train$Lag2)
test_mat <- cbind(test$Lag1, test$Lag2)

set.seed(1)
#knn() takes 4 arguments, a training matrix containing only the predictors, a test matrix the same, the training response vector, and K neighbors
knn_pred <- knn(train_mat, test_mat, train$Direction, k=1)  
table(knn_pred, test$Direction)
mean(knn_pred == test$Direction)

#Change k to 3
knn_pred <- knn(train_mat, test_mat, train$Direction, k=3)  
table(knn_pred, test$Direction)
mean(knn_pred == test$Direction)
 
#####Applying KNN to Caravan insurance
data <- Caravan
summary(data)
head(data)

#We need to standardize the data for accurate estimates
    #scale() sets the mean value to 0 and stdev of 1
    #exclued column 86 because that is the response variable
standardized <- scale(data[,-86]) 

#split into training or not 
test_ind <- 1:1000
train <- standardized[-test_ind,]
test <- standardized[test_ind,]
train_res <- data$Purchase[-test_ind]
test_res <- data$Purchase[test_ind]

#now run KNN
set.seed(1)
knn_pred <- knn(train, test, train_res, k=1)
mean(test_res ==knn_pred)
      #looks pretty good, but only 6% are no so we could actually do better by just predicting "no" all the time
table(knn_pred, test_res)
#so if determining who we should attempt to sell to we can compute who we should be selling to
9/ (68+9) #--> 11.6%, which is better than the 6% randomly assigned
#the sensitivity of this approach does better as we increase k

#We can also fit a logit to the data for a pretty good decision as to where to invest time/resources
train <- data[-test_ind,]
test <- data[test_ind,]
glm_model <- glm(Purchase ~., data = train, family = "binomial")
glm_model
glm_pred <- predict(glm_model, newdata = test, type="response")
glm_pred <- ifelse(glm_pred > 0.25, "Yes", "No")
table(glm_pred, test$Purchase)
11/(22+11)  #33.3% chance of making a sale!!! Thats a pretty good ratioB

#caret methods etc.
    #http://topepo.github.io/caret/index.html

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

train <- select(train, c(-Name, -Cabin, -Ticket))
head(train)
######################################################################
############## Loading the "caret" package  ##########################
######################################################################
library(caret)

######################################################################
###################### Data Visualizations  ##########################
######################################################################
#we are able to plot out some of the data to visually inspect
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
featurePlot(x= iris[,1:4],
            y = iris$Species,
            plot="ellipse")  #The ellipse doesn't look like it works for categorical variables

#plot choices: pairs, ellipse, density, box, scatter

#These plots don't do much for the titanic dataset, but they do look really nice on the iris dataset
featurePlot(x= train[,5:8],
            y = as.factor(train$Survived),
            plot="pairs",
            auto.key=list(columns=2))

######################################################################
############## Preprocessing - Caret #################################
######################################################################

#Identify and remove highly correlated observations
descrCor <- cor(data)  #must be only numeric vectors in data
highlyCor <- findCorrelation(descrCor, cutoff = 0.75) #remove descriptors with cor() above 0.75
filteredCor <- data[,-highlyCor]

preprocessed <- preProcess(data[,columns],
                           method = c("center", "scale"))  #Plus others

######################################################################
############## caret ##############################
######################################################################

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


#misnamed, but is an adaboost; have to make sure y is a factor
rpartFit <- train(as.factor(Survived) ~ Pclass + Age + Title + Sex + Embarked,
                  data = train,
                  method = "adaboost")
prediction <- predict(rpartFit, newdata = test_clean)

######################################################################
############## Diagnostics with caret ################################
######################################################################

#traditional way
conf_tree <- table(test$Survived, tree_predict)
conf_tree
acc_tree <- sum(diag(conf_tree)) / sum(conf_tree)
acc_tree

#Using caret
confusionMatrix(data = test_set$pred, reference = test_set$obs)
      #Prints the matrix and the statistics on the measure
twoClassSummary() #Summarizes ROC, Sens, Spec for two class outcomes
prSummary()  #AUC, Precision, Recall, 

mnLogLoss() #computes the negative of the multinomial log-likelihood (smaller is better) based on class probabilities
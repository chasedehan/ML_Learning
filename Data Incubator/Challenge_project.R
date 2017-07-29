#Challenge_project.R
setwd("/home/chasedehan/Dropbox/R Code/Lending_Club")


##################Load the libraries
library(tidyverse, lubridate, Boruta)

##################Importing Data
lc_data <- read_csv("LoanStats3a_securev1.csv")

##################First Plot between FICO and subgrade
ggplot(data = lc_data, aes(x = sub_grade, y=fico_range_high)) + geom_boxplot()

################## Cleaning and Feature Engineering
#There are a number of other things that could be placed into effect here, but these appear to be a good first shot
loan_data <- lc_data %>%
  mutate(default = ifelse(loan_status=="Charged Off", 1, 0),
         issue_d = mdy(issue_d),
         earliest_cr_line = mdy(earliest_cr_line),
         time_history = issue_d - earliest_cr_line,
         revol_util = as.numeric(sub("%","", revol_util)), 
         emp_listed = as.numeric(!is.na(emp_title) * 1),
         empty_desc = as.numeric(is.na(desc)), #I'm not sure if this is a pre or post description, need to look into it
         emp_na = ifelse(emp_length == "n/a", 1, 0),
         emp_length = ifelse(emp_length == "< 1 year" | emp_length == "n/a", 0, emp_length),
         emp_length = as.numeric(gsub("\\D", "", emp_length)),
         delinq_ever = as.numeric(!is.na(mths_since_last_delinq)),   #Have they ever had a delinquency?
         home_ownership = ifelse(home_ownership == "NONE", "OTHER", home_ownership)
  ) %>%  #encoding <1 to having 0 time in job to distinguish from 1 year.
  select(default, loan_amnt, empty_desc, emp_listed, emp_na, emp_length, verification_status, 
         home_ownership, annual_inc, purpose, time_history, fico_range_high, inq_last_6mths, 
         open_acc, pub_rec, revol_util, dti, total_acc, delinq_2yrs, delinq_ever)
#Build the dummies, convert back to data.frame and remove the intercept
x <- as.data.frame(model.matrix(~ ., data = loan_data))[, -1]
#Remove the spaces in variable names
names(x) <- gsub(" ", "", names(x))

###########################Boruta Analysis
borutaTest <- Boruta(as.factor(default) ~ ., data=x, doTrace=1, ntree=500)
plot(borutaTest)  
boruta_stats <- attStats(borutaTest)
boruta_stats[order(boruta_stats$meanImp, decreasing = TRUE), ]

#I have soent a considerable amount of time on this project
    #This is just a sample of what I have completed thus far
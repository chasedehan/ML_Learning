#stacking process

#Run model a first time with an automatic grid search
    #Look and see where the grid should be expanded
#Run cross-validated model with the expanded grid
    #Evaluate the parameters
    #Select the appropriate by whatever measure
#Train cross-validated with selected parameters
    #Make predictions on the training for level 1
#Train the test data for Level 1 predictions with selected parameters
    #Make predictions on test data for level 1

#Repeat for each model at Level 0


#Do basically the same thing for the models at Level 1
    #Because these should be 
#allstate helper functions

#quick predict function
pred <- function(model) predict(model, newdata = train)

# Mean Squared Error
mse <- function(y, yhat) mean((yhat - y)^2)
# Root Mean Squared Error
rmse <- function(y, yhat)  sqrt(mean((yhat - y)^2))
# Mean Absolute Error
mae <- function(y, yhat)  mean(abs(yhat - y))
# Accuracy
accuracy <- function(y, yhat) {
  conf <- table(y, yhat)
  acc <- sum(diag(conf)) / sum(conf)
  return(acc)
}


#plot train predictions against the actual data
plot_yhat <- function(y, yhat) {
  plot(yhat, y)
  abline(0, 1, col = "red")
  print(paste("mse",mse(yhat)))
  print(paste("rmse",rmse(yhat)))
  print(paste("mae",mae(yhat)))
}

#Building a submission file for Kaggle
kag_submit <- function(testId, predictions, id_name = "id", response_name = "y", filename = "submission") {
  submission <- data.frame(id = testId, V1 = predictions)
  names(submission) <- c(id_name, response_name)
  print(head(submission))
  write.csv(submission, file = paste(filename, ".csv", sep = ""), row.names = FALSE)
}





#allstate helper functions

#quick predict function
pred <- function(model) predict(model, newdata = train)

# Mean Squared Error
mse <- function(pred) mean((pred - y)^2)
# Root Mean Squared Error
rmse <- function(pred)  sqrt(mean((pred - y)^2))
# Mean Absolute Error
mae <- function(pred)  mean(abs(pred - y))



#plot train predictions against the actual data
plot_pred <- function(pred) {
  plot(pred, y)
  abline(0, 1, col = "red")
  print(paste("mse",mse(pred)))
  print(paste("rmse",rmse(pred)))
  print(paste("mae",mae(pred)))
}

submit <- function(x) {
  submission <- data.frame(id = testId, loss = x)
  print(head(submission))
  write.csv(submission, file = "allstate_submission.csv", row.names = FALSE)
}





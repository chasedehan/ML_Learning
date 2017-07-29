
library(ISLR)
set.seed(1)
train = sample(392, 196)


lm_fit <- lm(mpg~horsepower + displacement, data = Auto, subset=train)
summary(lm_fit)


############  K-fold Cross-Validation

library(boot)
library(glmnet)
glm_fit <- glm(mpg~log(horsepower)  + displacement, data = Auto)
cv_err <- cv.lm(Auto, glm_fit, K=10)
cv_err$delta #[1] is the raw estimate, [2] is the adjusted to compensate for bias
      #This is the MSE of the model

cv_predict <- predict(cv_err)
plot(cv_predict, Auto$mpg, ylim= c(5,35), xlim=c(5,35))
##############Testing the size of the polynomial
set.seed(1)
cv_error_10 <- rep(0,10)
for(i in 1:10){
  glm_fit <- glm(mpg ~ poly(horsepower, i), data=Auto)
  cv_error_10[i] <- cv.glm(Auto, glm_fit, K=10)$delta[1]
}
cv_error_10


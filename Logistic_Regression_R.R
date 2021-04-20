#load the data
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", "present_resid", "property", "age", "other_install", "housing", "n_credits", "job", "n_people", "telephone", "foreign", "response")
# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1
colnames(german_credit)
summary(german_credit)
#split the dataset
index <- sample(nrow(german_credit),nrow(german_credit)*0.90)
credit_train = german_credit[index,]
credit_test = german_credit[-index,]
# logistic regression
german_glm0 <- glm(response~., family= binomial, data=credit_train)
summary(german_glm0)
#variable selection
german_glm_back <- step(german_glm0)
summary(german_glm_back)
AIC(german_glm0)
AIC(german_glm_back)
BIC(german_glm_back)
## variable selection using BIC
german_glm_back_BIC <- step(german_glm0, k=log(nrow(credit_train)))
summary(german_glm_back_BIC)
BIC(german_glm_back_BIC)
## ROC Curve
pred_glm0_train<- predict(german_glm_back_BIC, type="response")
install.packages('ROCR')
library(ROCR)
pred <- prediction(pred_glm0_train, credit_train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
## find the AUC
unlist(slot(performance(pred, "auc"), "y.values"))
## misclassification rate
pred_resp <- predict(german_glm_back_BIC,type="response")
hist(pred_resp)
table(credit_train$response, (pred_resp > 0.5)*1, dnn=c("Truth","Predicted"))
table(credit_train$response, (pred_resp > 0.2)*1, dnn=c("Truth","Predicted"))
##out-of-sample and AUC
#with original data
pred_glm0_test_original <- predict(german_glm0, newdata= credit_test, type = "response")
pred_original <- prediction(pred_glm0_test_original, credit_test$response)
perf_original <- performance(pred_original, "tpr", "fpr")
plot(perf_original, colorize=TRUE)
unlist(slot(performance(pred_original, "auc"), "y.values"))
#with 10% test set
pred_glm0_test<- predict(german_glm_back_BIC, newdata = credit_test, type="response")
pred_2 <- prediction(pred_glm0_test, credit_test$response)
perf_2 <- performance(pred_2, "tpr", "fpr")
plot(perf_2, colorize=TRUE)
unlist(slot(performance(pred_2, "auc"), "y.values"))
# misclassification rate for testing set
pred_resp2 <- predict(german_glm_back_BIC,type="response", newdata = credit_test)
hist(pred_resp2)
table(credit_test$response, (pred_resp2 > 0.5)*1, dnn=c("Truth","Predicted"))
##cross validation - 5 folds
pcut <- 0.5
cost1 <- function(r, pi, pcut){
  mean(((r==0)&(pi>pcut)) | ((r==1)&(pi<pcut)))}

cost2 <- function(r, pi, pcut){
  weight1 <- 2
  weight0 <- 1
  c1 <- (r==1)&(pi<pcut)
  c0 <-(r==0)&(pi>pcut)
  return(mean(weight1*c1+weight0*c0))
}
costfunc  <- function(obs, pred.p){
  + weight1 <- 5
  + weight0 <- 1
  + pcut <- 1/(1+weight1/weight0)
  + c1 <- (obs==1)&(pred.p < pcut) 
  + c0 <- (obs==0)&(pred.p >= pcut)
  + cost <- mean(weight1*c1 + weight0*c0)
  + return(cost)
}
library(boot)
german_glm1<- glm(response~. , family=binomial, data=german_credit); 
cv_result  <- cv.glm(data=german_credit, glmfit=german_glm1, cost=costfunc, K=5) 
cv_result$delta[2]
### Classification Tree
library(rpart)
library(rpart.plot)
index <- sample(nrow(german_credit),nrow(german_credit)*0.80)
credit_train1 = german_credit[index,]
credit_test1 = german_credit[-index,]
#With symmetric cost (default)
credit_rpart0 <- rpart(formula = response ~ ., data = credit_train1, method ="class")
#With asymmetric cost
credit_rpart <- rpart(formula = response~ ., data = credit_train1, method = "class", parms = list(loss=matrix(c(0,5,1,0), nrow =2)))
pred0 <- predict(credit_rpart0, type="class") 
table(credit_train1$response, pred0, dnn = c("True","Pred"))
#intepret: false neg is a lot -> not good, this is because we don't specify asymmetric cost
pred_classification <- predict(credit_rpart, type="class")
table(credit_train1$response, pred_classification, dnn = c("True","Pred"))
#much lower false neg because we specify asymmetric cost -> better model
prp(credit_rpart, extra =1)


library(data.table)   # Read Data
library(plotly)       # Data Visualization
library(DMwR)         # Data Imputation by Knn
library(mice)         # Data Imputation by Regression
library(missForest)   # Data Imputation by Random Forest
library(ROSE)         # Synthetic Data Generation
library(glmnet)       # Lasso Regression
library(plotmo)       # Lasso Regression Visualization
library(xgboost)      # Gradient Boosting Machine
library(caret)        # Cross Validation

PredPerformance <- function(predictedval, actualval){
  confusionMatrix <- table(predictedval,actualval)
  Accuracy <- mean(predictedval==actualval)
  TrueNegative = confusionMatrix[1,1]
  FalseNegative = confusionMatrix[1,2]
  FalsePositive = confusionMatrix[2,1]
  TruePositive = confusionMatrix[2,2]
  
  Positive = FalseNegative + TruePositive
  Negative = FalsePositive + TrueNegative
  
  Sensitivity <- TruePositive/Positive
  Specificity <- 1 - FalsePositive/Negative
  
  predictionPerformance <- matrix(c(Accuracy))
  rownames(predictionPerformance) <- c("Accuracy")
  return(predictionPerformance)
}

setwd("C:/Users/Apu/Downloads")
feature <- fread("secom.data", data.table = F)
label <- fread("secom_labels.data", data.table = F)
data <- cbind(label,feature)
colnames(data) <- c("Class", "Time", paste0(rep("Feature", ncol(feature)), seq(1,ncol(feature))))
data$Class <- factor(data$Class, labels = c("pass", "fail"))
data$Time <-  as.POSIXct(data$Time, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
# View(data["Feature276"])
print(data)

str(data, list.len=8)
summary(data[,1:8])


#variable redundant
# Time #
index_vr1 <- which(colnames(data) == "Time")

# Equal Value #
equal_v <- apply(data, 2, function(x) max(na.omit(x)) == min(na.omit(x)))
index_vr2 <- which(equal_v == T)

#Missing value Imputation
row_NA <- apply(data, 1, function(x) sum(is.na(x))/ncol(data))
col_NA <- apply(data, 2, function(x) sum(is.na(x))/nrow(data))
plot_ly(x = seq(1,nrow(data)), y = row_NA, type = "scatter", mode = "markers") %>% 
layout(title = "Observation Missing Values Percentage",xaxis = list(title = "Observation Index"),yaxis = list(title = "Percentage(%)"))


plot_ly(x = seq(1,ncol(data)), y = col_NA, type = "scatter", mode = "markers") %>%
  layout(title = "Variable Missing Values Percentage",
         xaxis = list(title = "Variable Index"),
         yaxis = list(title = "Percentage(%)"))

index_mr <- which(col_NA > 0.4)
data_c <- data[,-unique(c(index_vr1, index_vr2, index_mr))]
data_I  <- knnImputation(data_c)

# View(data_I["Feature479"])

set.seed(10)
index <- sample(1:nrow(data_I), nrow(data_I)/10)
train <- data_I[-index,]
test <- data_I[index,]

train_rose <- ROSE(Class ~ ., data = train, seed = 1)$data
table(train_rose$Class)


#feature selection

fit_LS <- glmnet(as.matrix(train_rose[,-1]), train_rose[,1], family="binomial", alpha=1)
plot_glmnet(fit_LS, "lambda", label=5)
fit_LS_cv <- cv.glmnet(as.matrix(train_rose[,-1]), as.matrix(as.numeric(train_rose[,1])-1), type.measure="class", family="binomial", alpha=1)
plot(fit_LS_cv)
coef <- coef(fit_LS_cv, s = "lambda.min")
coef_df <- as.data.frame(as.matrix(coef))
index_LS <- rownames(coef_df)[which(coef_df[,1] != 0)][-1]
print(index_LS)
summary(index_LS)


#PCA
library(Boruta)
set.seed(100)
fit_PCA <- Boruta(as.matrix(train_rose[,-1]), train_rose[,1], doTrace =2)
# plot_glmnet(final.feature_PCA, "lambda", label=5) 
# pr.out = prcomp(train_rose[,-1]), scale = TRUE)

print(fit_PCA)
final.fit_PCA <- TentativeRoughFix(fit_PCA)
print(final.fit_PCA)

final.feature_PCA <- getSelectedAttributes(final.fit_PCA, withTentative = F)

print(final.feature_PCA)
biplot(final.feature_PCA)
# biplot(final.feature_PCA, scale = 0)


#Model Fitting
fit_LR <- glm(Class ~ ., data=train_rose[,c("Class",index_LS)], family = "binomial")
table_LR <- round(summary(fit_LR)$coefficient, 4)
table_LR[order(table_LR[,4])[1:20],]
pred_LR <- factor(ifelse(predict(fit_LR, test, type = "response") > 0.5, "fail", "pass"), levels = c("pass", "fail"))
table(test$Class, pred_LR)
roc.curve(test$Class, predict(fit_LR, test))

#PCA model fitting
fit_LR <- glm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)], family = binomial)
summary(fit_LR)
test_pred <- predict(fit_LR, test)
# test_pred
test_pred_final <- ifelse(test_pred > 0.5, "fail","pass")
logErrorTable <- PredPerformance(test_pred_final, test$Class)
logErrorTable
colnames(logErrorTable) <- "logistic"
print(logErrorTable)
roc.curve(test$Class, predict(fit_LR, test))



#Tree model
library(tree)
library(boot)
library(randomForest)
tree.fit <- tree(Class ~ .,  data=train_rose[,c("Class",final.feature_PCA)])
summary(tree.fit)
plot(tree.fit)
text(tree.fit,pretty = FALSE)

#LDA
library(MASS)
library(e1071)
fit_Lda <- lda(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)], family = "binomial")
summary(fit_Lda)
pred_Lda <- predict(fit_Lda, test, type ='response')
pred_Lda
pred_Lda_Final <- pred_Lda$class
pred_Lda_Final
ldaerror <- PredPerformance(pred_Lda_Final, test$Class)
colnames(ldaerror) <- "LDA"
print(ldaerror)

#SVC
library(e1071)
# i <- -1:2
# costs <- 10^i
# gammas <- seq(0.5,5,by = 0.5)
# degrees <- i[5:6]
svc <- svm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)],kernel = 'linear', gamma = 0.5, cost = 1)
svcPreds <- predict(svc, test)
svcErrorTable <- PredPerformance(svcPreds, test$Class)
colnames(svcErrorTable) <- "SVC"
print(svcErrorTable)

svc <- svm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)],kernel = 'linear', gamma = 1, cost = 100)
svcPreds <- predict(svc, test)
bagErrorTable <- PredPerformance(svcPreds, test$Class)
colnames(bagErrorTable) <- "Bagging"
print(bagErrorTable)

svc <- svm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)],kernel = 'linear', gamma = 1, cost = 100)
svcPreds <- predict(svc, test)
rfErrorTable <- PredPerformance(svcPreds, test$Class)
colnames(rfErrorTable) <- "Random Forest"
print(rfErrorTable)


#SVM Radial

svmRadial <- svm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)],kernel = 'radial', gamma = 0.5, cost = 10, scale = FALSE)
svmRadialPreds <- predict(svmRadial, test)
svmRadialErrorTable <- PredPerformance(svmRadialPreds, test$Class)
colnames(svmRadialErrorTable) <- "SVM Radial"
print(svmRadialErrorTable)


#SVM Poly

svmPoly <- svm(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)],kernel = 'polynomial', degree = 1, cost = 10, scale = FALSE)
svmPolyPreds <- predict(svmPoly, test)
svmPolyErrorTable <- PredPerformance(svmPolyPreds, test$Class)
colnames(svmPolyErrorTable) <- "SVM Poly"
print(svmPolyErrorTable)



# set.seed(1)
# p <- ncol(train_rose[,c("Class",final.feature_PCA)]) - 1
# p
# set.seed(1)
# bag <- randomForest(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)], ntree = 200, mtry = p, importance = TRUE)
# bagPreds <- predict(bag, test)
# bagErrorTable <- PredPerformance(bagPreds, test$Class)
# colnames(bagErrorTable) <- "Bagging"
# print(bagErrorTable)
# 
# 
# p <- ncol(train_rose[,c("Class",final.feature_PCA)]) - 1
# set.seed(1)
# rf <- randomForest(Class ~ ., data=train_rose[,c("Class",final.feature_PCA)], ntree = 100, mtry = sqrt(p), importance = TRUE)
# rfPreds <- predict(rf, test)
# rfErrorTable <- PredPerformance(rfPreds, test$Class)
# colnames(rfErrorTable) <- "Random Forest"
# print(rfErrorTable)


library(caret)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
knn_fit <- train(Class ~ ., data = train_rose[,c("Class",final.feature_PCA)], method = "knn",trControl=trctrl,preProcess = c("center", "scale"),tuneLength = 10)
knn_fit

test_pred <- predict(knn_fit, newdata = test)
knnErrorTable <- PredPerformance(test_pred, test$Class)
colnames(knnErrorTable) <- "KNN"
print(knnErrorTable)

SummaryErrorTable <- cbind(knnErrorTable, rfErrorTable, bagErrorTable, svmPolyErrorTable, svmRadialErrorTable, 
                           svcErrorTable, ldaerror, logErrorTable)
print(SummaryErrorTable)





 
---
title: PML report
author: "by Vishnu"
output:
  html_document:
    fig_height: 9
    fig_width: 9
---


## Preprocessing of Data 
```{r, cache = T}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
 
### Data Reading
we can read the two csv files into two data frames(test and train).
```{r, cache = T}
trainRaw <- read.csv("./data/p-training.csv")
testRaw <- read.csv("./data/p-testing.csv")
dim(trainRaw)
```

```
## [1] 19622   160
```	
```
dim(testRaw)
``` 
```
## [1]  20 160
```

### Data Cleaning
Claening gets rid of missing valus of some observations and meaning less values.
```{r, cache = T}
sum(complete.cases(trainRaw))
```
```
## [1] 417
```
Removing observations with no values
```{r, cache = T}
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```  
getting rid of pointless values
```{r, cache = T}
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```

### Data Slicing	
 Splitting data into standard 70 and 30.
```{r, cache = T}
set.seed(22328)
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

## Modeling of Data
predictive model for activity recognition using Random forest algorithm with 5fold cross validation.	
```{r, cache = T}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```
```
## Random Forest 
## 
## 13725 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.990827  0.988396  0.001675093  0.00212138
##   27    0.991119  0.988764  0.001755971  0.00222277
##   52    0.984057  0.979829  0.003497420  0.00442506
## 
```
estimating performance of model on validation sets.
```{r, cache = T}
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    1
##          B    5 1129    5    0    0
##          C    0    1 1020    5    0
##          D    0    0   13  949    2
##          E    0    0    1    6 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915	          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9982   0.9817   0.9885   0.9972
## Specificity            0.9995   0.9979   0.9988   0.9970   0.9985
## Pos Pred Value         0.9988   0.9912   0.9942   0.9844   0.9935
## Neg Pred Value         0.9988   0.9996   0.9961   0.9978   0.9994
## Prevalence             0.2850   0.1922   0.1766   0.1631   0.1832
## Detection Rate         0.2841   0.1918   0.1733   0.1613   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9983   0.9981   0.9902   0.9927   0.9979
```

```{r, cache = T}
accu <- postResample(predictRf, testData$classe)
accu
```
```
##  Accuracy     Kappa 
## 0.993203 0.991402
'''
'''
ooseee <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
ooseee
```
'''
## [1] 0.00679694
'''

estimated accuracy of the model = 99.57% 
the estimated out-of-sample error = 0.58%.

## Test data prediction 
```{r, cache = T}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
``` 

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
``` 

## Figures
1.Visualization of correalation matrix
```{r, cache = T}
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```
![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)
2.Visualization of decision tree
```{r, cache = T}
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) 
![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)
```

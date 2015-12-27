# Prediction and Analysis on Weight Lifting Exercise Dataset
Liang Dong  
27 December 2015  

# Executive Summary

This report focus on data from Weight Lifting Exercise. Among the 6 classes of weight lifting, only Class A is the correct training method, while others are training method with common mistakes. The report use other varaibles in the dataset to predict the class of weight lifting.

More information are available from Weight Lifting Exercises Dataset section in following website: http://groupware.les.inf.puc-rio.br/har

# Data Preprocessing

Load the required ggplot2 and caret library for prediction.


```r
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

Download the dataset and preprocess the data. Some of the data are missing, so it will be marked as NA for


```r
pml_train_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
pml_test_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("pml-training.csv")) {
    download.file(pml_train_url, destfile="pml-training.csv", method="curl")
}
if (!file.exists("pml-testing.csv")) {
download.file(pml_test_url, destfile="pml-testing.csv", method="curl")
}

pml_training <- read.csv("pml-training.csv", na.strings=c("NA",""))
pml_testing <- read.csv("pml-testing.csv", na.strings=c("NA",""))
```

The dimension of the training set is 19622 rows with 160 columns, the testing set is 20 rows with the same columns numbers. 


```r
dim(pml_training)
```

```
## [1] 19622   160
```

```r
dim(pml_testing)
```

```
## [1]  20 160
```

First of all, the training data has to be further splitted into training and testing set, which the testing set is used for characterising and cross-validating the performance of the prediction other than the final prediction set. Also, To achieve the reproducibility, the seed should be set to guarantee that goal.

70% percents of the training data is used for traning, while the remaining 30% is used for testing.


```r
set.seed(20151227)

inTrain <- createDataPartition(pml_training$classe, p=0.70, list=FALSE)
training <- pml_training[inTrain, ]
testing <- pml_training[-inTrain, ]
```

From the perspective weight lifting domain knowledge, the first several columns which are literally the row numbers, username, timestamps, should be excluded from the features set. 


```r
training <- training[, c(-1:-5)]
testing <- testing[, c(-1:-5)]
```

Also the near zero variable columns and mostly NA columns should be also removed from data set. The columns which kept are NA not exceeding 90%.


```r
nzv_train <- nearZeroVar(training)
training <- training[, -nzv_train]
testing<- testing[, -nzv_train]

NA90p <- sapply(training, function(x) mean(is.na(x))) < 0.90
training <- training[, NA90p==TRUE]
testing <- testing[, NA90p==TRUE]
```

# Random Forest

The random forest is the tried as the first algorithm. The result is shown as below, the error rate on training data is 99.74%.


```r
fit <- randomForest(classe ~ ., data = training)
fit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.3%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B    6 2650    2    0    0 0.0030097818
## C    0    9 2386    1    0 0.0041736227
## D    0    0   13 2237    2 0.0066607460
## E    0    0    0    7 2518 0.0027722772
```

# Cross Validation

To cross validate the result, the testing set is predicted and compared the original result. The accuracy is 99.75%, which is pretty good and consistent with the training set result.


```r
testing_prediction <- predict(fit, newdata=testing)
confusionMatrix(testing$classe, testing_prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1133    1    0    0
##          C    0    3 1022    1    0
##          D    0    0   10  954    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9943, 0.9977)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9974   0.9894   0.9969   1.0000
## Specificity            1.0000   0.9987   0.9992   0.9980   0.9996
## Pos Pred Value         1.0000   0.9947   0.9961   0.9896   0.9982
## Neg Pred Value         0.9988   0.9994   0.9977   0.9994   1.0000
## Prevalence             0.2853   0.1930   0.1755   0.1626   0.1835
## Detection Rate         0.2845   0.1925   0.1737   0.1621   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9980   0.9943   0.9974   0.9998
```

# Testset Prediction

It is time to predit the test set. Transform the test set into the same data format as the training set to fit the prediction model. The last columns are also removed because it is the problem id.


```r
pml_testing <- pml_testing[, c(-1:-5)]
pml_testing <- pml_testing[, -nzv_train]
pml_testing <- pml_testing[, NA90p==TRUE]

final_predict <- predict(fit, newdata=pml_testing)
```

Use the code given in the project submission to write the final prediction to file and submit.



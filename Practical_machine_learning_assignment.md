# Practical machine learning assignment



# Overview

In this assignment, we will use the training dataset to develop a model. The goal is to identify the type of movement given data from accelerometers on the belt, forearm, arm, and dumbell. In order to obtain an accurate model, we consider the method of cross validation by splitting the training dataset into two subsets: training and testing. 

The constructed model will be apply to predict the type of exercise over the testing dataset. Note that, all the preprocess used for the training dataset at the model development stage will be added to the testing set first.



# Data Processing

### 1. Load the data

All the blank space in the dataset will be replaced by 'NA'.


```r
TRA <- read.csv('pml-training.csv', na.strings = c('NA', ''))
```


### 2. Preprocess the data

The dataset contains lots of NAs. We use the following chunk to determine the degree of missing values in each column.


```r
k <- NULL
for (i in 1:dim(TRA)[2]) 
{
    k <- c(k, sum(!is.na(TRA[, i])))
}
k
```

```
##   [1] 19622 19622 19622 19622 19622 19622 19622 19622 19622 19622 19622
##  [12]   406   406   406   406   406   406   406   406   406   406   406
##  [23]   406   406   406   406   406   406   406   406   406   406   406
##  [34]   406   406   406 19622 19622 19622 19622 19622 19622 19622 19622
##  [45] 19622 19622 19622 19622 19622   406   406   406   406   406   406
##  [56]   406   406   406   406 19622 19622 19622 19622 19622 19622 19622
##  [67] 19622 19622   406   406   406   406   406   406   406   406   406
##  [78]   406   406   406   406   406   406 19622 19622 19622   406   406
##  [89]   406   406   406   406   406   406   406   406   406   406   406
## [100]   406   406 19622   406   406   406   406   406   406   406   406
## [111]   406   406 19622 19622 19622 19622 19622 19622 19622 19622 19622
## [122] 19622 19622 19622   406   406   406   406   406   406   406   406
## [133]   406   406   406   406   406   406   406 19622   406   406   406
## [144]   406   406   406   406   406   406   406 19622 19622 19622 19622
## [155] 19622 19622 19622 19622 19622 19622
```

As can be seen from the array, many columns contain only 2% (i.e. 406/19622) observed values. We will remove all these columns. 'id' is defined as the index of all the useful columns.


```r
id <- NULL
for (i in 1:length(k))
{
  if (k[i] == 19622) {id <- c(id, i)}
}
```

Further, we remove the first seven columns which contain the irrelevant information for our model development.


```r
id <- id[-1 : -7]
length(id)
```

```
## [1] 53
```

Now, the total number of variables is 52 and the last column is the type of exercise which we would like to predict.


### 3. Prepare datasets for cross validation

The ratio of observations in training dataset and validating dataset is 7 to 3.


```r
Tra <- TRA[, id]
set.seed(20170507)
library(caret)
inTrain = createDataPartition(Tra$classe, p = 0.7)[[1]]
training = Tra[inTrain, ]
validating = Tra[-inTrain, ]
```



# Machine learning algorithm development

In this problem, we need a method for **classification** since the expected result is one specific type of exercise. Recognising we have a large group of variables, the random forests could be a good model which constructs a multitude of decision trees and outputs the class that is the mode of the classes (classification) of the individual trees.

### 1. Set the seed and predict 'classe' with all the other variables using a 'randomForest' model


```r
library(randomForest)
set.seed(20170507)
rf_tra <- randomForest(classe ~ ., data = training)
rf_tra$confusion
```

```
##      A    B    C    D    E class.error
## A 3902    2    0    0    2 0.001024066
## B   14 2641    3    0    0 0.006395786
## C    0   18 2376    2    0 0.008347245
## D    0    0   27 2223    2 0.012877442
## E    0    0    1    7 2517 0.003168317
```

The low class.error in each class indicates that the model can efficiently classify the exercise based on the recorded data from device.


### 2. Use 'confusionMatrix' to determine the resulting accuracy on the validating set 


```r
rf_val_pre <- predict(rf_tra, validating)
confusionMatrix(validating$classe, rf_val_pre)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    3 1134    2    0    0
##          C    0    8 1016    2    0
##          D    0    0    7  955    2
##          E    0    0    1    1 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9954         
##                  95% CI : (0.9933, 0.997)
##     No Information Rate : 0.2848         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9942         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9921   0.9903   0.9969   0.9982
## Specificity            0.9998   0.9989   0.9979   0.9982   0.9996
## Pos Pred Value         0.9994   0.9956   0.9903   0.9907   0.9982
## Neg Pred Value         0.9993   0.9981   0.9979   0.9994   0.9996
## Prevalence             0.2848   0.1942   0.1743   0.1628   0.1839
## Detection Rate         0.2843   0.1927   0.1726   0.1623   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9990   0.9955   0.9941   0.9975   0.9989
```

**The prediction accuracy is 99.54% when considering the validating cases. Hence, the fitted random forests model is achieved.**



# Apply the algorithm to the testing set

### 1. Load and preprocess the data using the same approach


```r
TEST <- read.csv('pml-testing.csv', na.strings = c('NA', ''))
testing <- TEST[, id]
```



### 2. Predict the 'classe' using the 'rf_tra' model


```r
predict(rf_tra, testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


### 3. Finish the Quiz with the above result

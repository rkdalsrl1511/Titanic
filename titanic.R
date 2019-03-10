library(tidyverse)
library(rpart) # �ǻ��������
library(randomForest) # ����������Ʈ
library(caret) # ȥ����Ŀ� �ʿ��� ��Ű��
library(MLmetrics) # F1 ������ �ʿ��� ��Ű��
library(xgboost)

# �۾����� �����ϱ�
setwd('C:/Users/Leehuimin/Desktop/���α׷��� ���/R/Kaggle/Titanic')
getwd()


# �����ͼ� �ҷ�����
Titanic.train <- read.csv(file = 'train.csv',
                          header = TRUE)

# testset���� ��ǥ������ ���� ( survived )
Titanic.test <- read.csv(file = 'test.csv',
                         header = TRUE)


# ������ ���� �ľ��ϱ�
str(Titanic.train)
nrow(Titanic.train)
View(Titanic.train)

# NA�� �� �ľ��ϱ�
sapply(Titanic.train, function(x) sum(is.na(x)))

# ----------------------------------------------------------------
# 1. EDA

# ����ó�� ���⼺
# passengerId : index ( ���� )
# survived : �������� ( ��ǥ���� ). factor�� ��ȯ�ϱ�
# Pclass : Ƽ�� Ŭ���� 1,2,3���� �Ǿ�����. factor�� ��ȯ�ϱ�
# Name : �̸� ( ���� )
# sex : �ǵ� �ʿ� ����
# age : NA�� ó���ϱ�
# sibsp : ���� �� ������� ž�� �� �� �� ����. ( factor or int )
# parch : �θ� �� ������ ž�� �� �� �� ����. ( factor or int )
# Ticket : Ƽ�� ��ȣ. factor�̱� �ѵ�, ���� ���� �ٿ����ҵ�.
# Fare : ����. ���� �״�� ����
# Cabin : factor�̱� �ѵ�, ���� ����. ����, ������ ""�� �͵��� ����
# Embarked : Ư���� ������ ������, ������ ""�� �͵��� ����

# PassengerId, Name �����
Titanic.train <- Titanic.train %>% dplyr::select(-PassengerId,
                                                 -Name)
str(Titanic.train)

# survived, Pclass factor�� ��ȯ�ϱ�
Titanic.train$Survived <- as.factor(Titanic.train$Survived)
Titanic.train$Survived <- relevel(Titanic.train$Survived,
                                  ref = '1')
levels(Titanic.train$Survived)

Titanic.train$Pclass <- as.factor(Titanic.train$Pclass)
levels(Titanic.train$Pclass)


# sibsp ó���ϱ� : factor�� ��ȯ�ϱ�
table(Titanic.train$SibSp)
Titanic.train$SibSp <- as.factor(Titanic.train$SibSp)

# parch ó���ϱ� : factor�� ��ȯ�ϱ�
table(Titanic.train$Parch)
Titanic.train$Parch <- as.factor(Titanic.train$Parch)

# embarked ó���ϱ� : ���� �̸� �ٲٱ�
levels(Titanic.train$Embarked)
levels(Titanic.train$Embarked) <- c('unknown','C','Q','S')

# �߿��� ����
# ticket, cabin ó���ϱ�
# age ó���ϱ� : �̰͵� ���������� ó���ϴ� ���� �� ���� �� ����.

# ���(1) : �ϴ� ���������� ��ȯ�ϰ�, survived ������ ���� ����
# ���(2) : ���������� ��ȯ�ϰ�, survived ������ ���� ���� ��, trainset���� ������, testset���� ���� �����͸� ó���ؾ��Ѵ�. �̶� �� �������� ���� ����, �������� ���� ���� ��� �״�� �ξ���ϰ�, ���� ��� �׳� unknown���� ��ģ��. �ϴ� trainset�� validationset������ 1���� ����� ����Ѵ�.

# �ǹ���: NA������ survived�� 0�� �� ���� ���� �׷�ȭ�ϰ�, 1�ΰ� ���� ����� �׷�ȭ�ϸ� �� ������? �� ������?

table(Titanic.train$Age)
table(Titanic.train$Ticket)
table(Titanic.train$Cabin)

# �ϴ� age�� NA�� ó���ϱ�
# age = NA���� survived ��
table(Titanic.train[is.na(Titanic.train$Age), 'Survived'])

# NA 1�� 0 �и��ؼ� �׷�ȭ�ϱ�
Titanic.train$Age <- as.character(Titanic.train$Age)

# 0�� �͵��� �������� �����
Titanic.train[is.na(Titanic.train$Age) & Titanic.train$Survived == 0,
              'Age'] <- 'unknown0'

# 1�� �͵��� �������� �����
Titanic.train[is.na(Titanic.train$Age) & Titanic.train$Survived == 1,
              'Age'] <- 'unknown1'

# factor�� ��ȯ
Titanic.train$Age <- as.factor(Titanic.train$Age)

# �� ��ȯ �Ǿ����� Ȯ���ϱ�
by(Titanic.train$Survived,
   Titanic.train$Age,
   table)


# ���� 3���� ������, ���� 10�� ���Ϸ� ��ҽ�Ű��
change.name <- c('Fare','Age','Ticket','Cabin')

for(i in 1:length(change.name)){
  
  # charactor�� ��ȯ
  Titanic.train[, change.name[i]] <- as.character(Titanic.train[, change.name[i]])
  
  # �����鿡 ���� survived�� Ȯ���ϱ� ���� ���� detector
  detector <- by(Titanic.train$Survived, 
                 Titanic.train[, change.name[i]], 
                 table)
  
  # ������ ���� ������ �����س��� ����
  detector.vector <- c()
  
  # �� ������ ���� ������ ���ؼ� �����Ѵ���, �� ������ ���� �̸��� �����Ѵ�. ����, ���������� factor�� ��ȯ�Ѵ�. 
  for(k in 1:length(detector)){
  
    # ����
    detector.prop <- detector[[k]][1] / (detector[[k]][1] + detector[[k]][2]) * 100
      
    # rbind�� �����ϱ�
    detector.vector <- rbind(detector.vector, detector.prop)
    
  }
  
  # ����
  detector.factor <- cut(detector.vector,
                         breaks = seq(from = 0,
                                      to = 100,
                                      by = 10),
                         right = FALSE)
  
  
  for(k in 1:length(detector)){
    
    
    # ������ 100�� ���
    if(is.na(detector.factor[k])){
      
      Titanic.train[Titanic.train[,change.name[i]] == names(detector)[k],change.name[i]] <- '[ukn]'
      
      # �� ��
    }else{
      
      Titanic.train[Titanic.train[,change.name[i]] == names(detector)[k],change.name[i]] <- as.character(detector.factor[k])
      
    }
    
  }
  
  # factor�� ��ȯ
  Titanic.train[, change.name[i]] <- as.factor(Titanic.train[, change.name[i]])
  
}

# ����� ��ȯ�Ͽ����� Ȯ���ϱ�
for(i in 1:length(change.name)){
  
  cat('������ �̸� : ', change.name[i], '\n')
  cat('������ �� : ',nlevels(Titanic.train[, change.name[i]]), '\n')
  
}

# trainset�� validationset���� ������
set.seed(123)

index <- sample(1:2,
                size = nrow(Titanic.train),
                prob = c(0.7,0.3),
                replace = TRUE)


Titanic.train.t <- Titanic.train[index == 1, ]
Titanic.train.v <- Titanic.train[index == 2, ]

nrow(Titanic.train.t)
nrow(Titanic.train.v)


# ���� ���� ------------------------------------------------------------

# �ǻ�������� �����ϱ�

fitTree <- rpart(Survived ~.,
                 data = Titanic.train.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

print(fitTree)

pred <- predict(fitTree,
                newdata = Titanic.train.v,
                type = 'class')

real <- Titanic.train.v$Survived


# ���� ��ġ Ȯ���ϱ�
confusionMatrix(pred,real, positive = '1')
F1_Score(pred,real)

# --------------------------------------------------------------------

# ����������Ʈ ���� �����ϱ�

fitRFC <- randomForest(x = Titanic.train.t[,-1],
                       y = Titanic.train.t[, 1],
                       xtest = Titanic.train.v[,-1],
                       ytest = Titanic.train.v[, 1],
                       ntree = 700,
                       mtry = 2,
                       importance = TRUE,
                       do.trace = 50,
                       keep.forest = TRUE)


pred <- fitRFC$test$predicted
real <- Titanic.train.v$Survived

# ���� ��ġ Ȯ���ϱ�
confusionMatrix(pred,real, positive = '1')
F1_Score(pred,real)

# -------------------------------------------------------------------

# xgboost ���� �����ϱ�
# ��� ������ �ڷḦ ���̺����� ��ȯ�ϰ�, ������ �����ϵ��� �ϰڴ�.

# ���̺���ȭ �� ������
Titanic.train.dummy <- Titanic.train
Survived <- Titanic.train.dummy[, 1] # ��ǥ����
Survived <- ifelse(Survived == 0,0,1)
str(Survived)
Titanic.train.dummy <- Titanic.train.dummy[,-1] # ��ǥ������ ������ ������
str(Titanic.train.dummy)

# ���̺��� ����� �Լ� ( ���̺������� data.frame���� ��ȯ )
MakingDummy <- function(data.variable, data.name){
  
  result <- data.frame(index = 1:length(data.variable))
  
  for(i in 1:nlevels(data.variable)){
    
    newdummy <- ifelse(data.variable ==
                         levels(data.variable)[i],
                       1,
                       0)
    result <- cbind(result, newdummy)
    newname <- str_c(data.name, i, sep = '.')
    colnames(result)[i+1] <- newname
    
  }
  
  result <- result[,-1]
  
  return(result)
  
}

# ���̺���ȭ �� ������ �̸���
factor.name <- colnames(Titanic.train.dummy)
factor.name


for(i in 1:length(factor.name)){
  
  Titanic.train.dummy <- cbind(Titanic.train.dummy,
                               MakingDummy(Titanic.train.dummy[,factor.name[i]], factor.name[i]))
  
}

# �ʿ���� ������ �����ϰ�, ��ǥ���� �����ϱ�
Titanic.train.dummy <- Titanic.train.dummy[,-c(1:9)]
Titanic.train.dummy <- cbind(Titanic.train.dummy, Survived)
ncol(Titanic.train.dummy)
str(Titanic.train.dummy)

# trainset�� validationset���� ������
Titanic.train.dummy.t <- Titanic.train.dummy[index == 1, ]
Titanic.train.dummy.v <- Titanic.train.dummy[index == 2, ]

# xgboost �����ϱ� ( ����Ʈ������ )
dtrain <- xgb.DMatrix(data = as.matrix(Titanic.train.dummy.t[ ,-57]),
                      label= as.matrix(Titanic.train.dummy.t[ , 57]))

# �Ķ����
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds ã��
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost ���� �����ϱ�
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(Titanic.train.dummy.v[, -57]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- Titanic.train.dummy.v[, 57] %>% as.factor() %>% relevel(ref = '1')

# ȥ�����
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)


# ---------------------------------------------------------------
# ---------------------------------------------------------------

# testset���� �����ؼ�, �����غ���

# train�� test�� ������ ������ ������, 
Titanic.master <- rbind(Titanic.train[,-2], Titanic.test)
Titanic.object <- Titanic.train[,2] %>% as.factor()
Titanic.object <- relevel(Titanic.object, ref = '1')

# NA �ľ��ϱ�
sapply(Titanic.master, function(x) sum(is.na(x)))

# PassengerId, Name �����
Titanic.master <- Titanic.master %>% dplyr::select(-PassengerId,
                                                   -Name)
str(Titanic.master)

# Pclass factor�� ��ȯ�ϱ�
Titanic.master$Pclass <- as.factor(Titanic.master$Pclass)
levels(Titanic.master$Pclass)

# sibsp ó���ϱ� : factor�� ��ȯ�ϱ�
table(Titanic.train$SibSp)
Titanic.master$SibSp <- as.factor(Titanic.master$SibSp)
levels(Titanic.master$SibSp)

# parch ó���ϱ� : factor�� ��ȯ�ϱ�
table(Titanic.master$Parch)
Titanic.master$Parch <- as.factor(Titanic.master$Parch)
levels(Titanic.master$Parch)

# embarked ó���ϱ� : ���� �̸� �ٲٱ�
levels(Titanic.master$Embarked) <- c('unknown','C','Q','S')
levels(Titanic.master$Embarked)

# age ó���ϱ� : factor�� ��ȯ
# ���⼭�� ���� �����ϴ�.
# �ϴ� survived�� ���� NA������ ���� ���� �ִ�. ������, train���� ������, test���� �ִ� NA��? survived�� ���� ���� ��Ʊ� ������ �ٸ� ����� �����ؾ� �Ѵ�. ���� unknown���� ���� ����ۿ� ���� �� ����.

# trainset �α��� ��Ȳ Ȯ���ϱ� 
by(Titanic.object,
   Titanic.master[1:nrow(Titanic.train), 'Age'],
   table)

# NA 1�� 0 �и��ؼ� �׷�ȭ�ϱ�
Titanic.master$Age <- as.character(Titanic.master$Age)

# 0�� �͵��� �������� �����

Titanic.temp <- Titanic.master[1:length(Titanic.object), 'Age']
Titanic.temp[is.na(Titanic.temp) & Titanic.object == 0] <- 'unknown0'
Titanic.master[1:length(Titanic.object), 'Age'] <- Titanic.temp

# 1�� �͵��� �������� �����
Titanic.temp <- Titanic.master[1:length(Titanic.object), 'Age']
Titanic.temp[is.na(Titanic.temp) & Titanic.object == 1] <- 'unknown1'
Titanic.master[1:length(Titanic.object), 'Age'] <- Titanic.temp

# ���� testset �α��� 86���� ��������
# unknownTest ��� ���������
Titanic.master[is.na(Titanic.master$Age), 'Age'] <- 'unknownTest'

# factor�� ��ȯ
Titanic.master$Age <- as.factor(Titanic.master$Age)

# ����� �������� Ȯ���غ���
table(Titanic.master$Age)

# Fare�� na�� ó���ϱ� ( �׳� �������� )
table(Titanic.master$Fare)
Titanic.master[is.na(Titanic.master$Fare), 'Fare'] <- 14.4542
Titanic.master$Fare <- as.factor(Titanic.master$Fare)

# ��ȯ �� �Ǿ����� Ȯ���ϱ�
str(Titanic.master)
str(Titanic.object)


# -----------------------------------------------------------------

# ���� �� 10�� ���Ϸ� ����ϱ� ( ���� ����� �۾� )

# �ϴ� �󵵸� ���ؼ� ��ҽ�ų �� �ִٸ�, �ִ��� �󵵸� Ȱ���ؼ� ����Ѵ�.
# �󵵸� ���ؼ� ������� ���ϴ� �͵��� ��¿ �� ���� ���Ĺ�����.

# ���� 3���� ������, ���� 10�� ���Ϸ� ��ҽ�Ű��
change.name <- colnames(Titanic.master)
change.name

# ���� ��� �۾�
for(i in 1:length(change.name)){
  
  # �����鿡 ���� survived�� Ȯ���ϱ� ���� ���� detector
  detector <- by(Titanic.object, 
                 Titanic.master[1:length(Titanic.object), change.name[i]], 
                 table)
  
  # charactor�� ��ȯ
  Titanic.master[, change.name[i]] <- as.character(Titanic.master[, change.name[i]])
  
  # ������ ���� ������ �����س��� ����
  detector.vector <- c()
  
  # �� ������ ���� ������ ���ؼ� �����Ѵ���, �� ������ ���� �̸��� �����Ѵ�. ����, ���������� factor�� ��ȯ�Ѵ�. 
  for(k in 1:length(detector)){
    
    # trainset���� �������� �ʴ� ������ ���
    if(is.null(detector[[k]])){
      
      detector.prop <- 110
      detector.vector <- rbind(detector.vector, detector.prop)
      
      # trainset�� �����Ǵ� ������ ���
    }else{
      
      detector.prop <- 100 * detector[[k]][1] / (detector[[k]][1] + detector[[k]][2])
      
      detector.vector <- rbind(detector.vector, detector.prop)
      
    }
    
  }
  
  # ����
  detector.factor <- cut(detector.vector,
                         breaks = seq(from = 0,
                                      to = 110,
                                      by = 10),
                         right = FALSE)
  
  
  for(k in 1:length(detector)){
    
    
    # ������ 100�� ���
    if(is.na(detector.factor[k])){
      
      Titanic.master[Titanic.master[,change.name[i]] == names(detector)[k],change.name[i]] <- '[ukn]'
      
      # �� ��
    }else{
      
      Titanic.master[Titanic.master[,change.name[i]] == names(detector)[k],change.name[i]] <- as.character(detector.factor[k])
      
    }
    
  }
  
  # factor�� ��ȯ
  Titanic.master[, change.name[i]] <- as.factor(Titanic.master[, change.name[i]])
  
}


# ���� ��� Ȯ���ϱ�
for(i in 1:length(change.name)){
  
  cat('������ �̸� : ', change.name[i], '\n')
  cat('������ �� : ',nlevels(Titanic.master[, change.name[i]]), '\n')
  
}

str(Titanic.master)
str(Titanic.object)

# -------------------------------------------------------------------

# xgboost ���� �����ϱ� ���� ������ ��ó��
# ��� ������ �ڷḦ ���̺����� ��ȯ�ϰ�, ������ �����ϵ��� �ϰڴ�.

# ���̺���ȭ �� ������
Titanic.master.dummy <- Titanic.master
Survived <- Titanic.object # ��ǥ����
Survived <- ifelse(Survived == 0,0,1)
str(Titanic.master.dummy)
str(Survived)

# ���̺��� ����� �Լ� ( ���̺������� data.frame���� ��ȯ )
MakingDummy <- function(data.variable, data.name){
  
  result <- data.frame(index = 1:length(data.variable))
  
  for(i in 1:nlevels(data.variable)){
    
    newdummy <- ifelse(data.variable ==
                         levels(data.variable)[i],
                       1,
                       0)
    result <- cbind(result, newdummy)
    newname <- str_c(data.name, i, sep = '.')
    colnames(result)[i+1] <- newname
    
  }
  
  result <- result[,-1]
  
  return(result)
  
}

# ���̺���ȭ �� ������ �̸���
factor.name <- colnames(Titanic.master.dummy)
factor.name


for(i in 1:length(factor.name)){
  
  Titanic.master.dummy <- cbind(Titanic.master.dummy,
                                MakingDummy(Titanic.master.dummy[,factor.name[i]], factor.name[i]))
  
}

# �ʿ���� ������ �����ϰ�, ��ǥ���� �����ϱ�
Titanic.master.dummy <- Titanic.master.dummy[,-c(1:9)]
str(Titanic.master.dummy)


# -----------------------------------------------------------------

# xgboost �����ϱ�

dtrain <- xgb.DMatrix(data = as.matrix(Titanic.master.dummy[1:length(Titanic.object), ]),
                      label= as.matrix(Titanic.object))

# �Ķ����
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds ã��
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost ���� �����ϱ�
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


# �����ϱ�
pred <- predict(newxgb, as.matrix(Titanic.master.dummy[-c(1:length(Titanic.object)), ]))
pred <- ifelse(pred > 0.5, 1, 0)


# -----------------------------------------------------------------------

# �������� ���� / ������ �����ϱ�

# csv �ҷ�����
submission <- read.csv(file = 'gender_submission.csv',
                       header = TRUE)

# ���� �ľ��ϱ�
nrow(submission)
str(submission)

# ������ Survived �����ϱ�
submission <- submission[,-2]

# pred�� ���ο� Survived�� �߰��ϱ�
submission <- data.frame(PassengerId = submission,
                         Survived = pred)

# �� �����ߴ��� Ȯ���ϱ�
str(submission)

# �����ϱ�
write.csv(submission,
          file = 'submission.csv',
          row.names = FALSE)




# ��������, �� �������� �������ھ� 0.72727 �μ�, 8971���� �Ͽ���.
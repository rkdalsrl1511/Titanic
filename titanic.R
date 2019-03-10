library(tidyverse)
library(rpart) # 의사결정나무
library(randomForest) # 랜덤포레스트
library(caret) # 혼동행렬에 필요한 패키지
library(MLmetrics) # F1 점수에 필요한 패키지
library(xgboost)

# 작업공간 설정하기
setwd('C:/Users/Leehuimin/Desktop/프로그래밍 언어/R/Kaggle/Titanic')
getwd()


# 데이터셋 불러오기
Titanic.train <- read.csv(file = 'train.csv',
                          header = TRUE)

# testset에는 목표변수가 없음 ( survived )
Titanic.test <- read.csv(file = 'test.csv',
                         header = TRUE)


# 데이터 구조 파악하기
str(Titanic.train)
nrow(Titanic.train)
View(Titanic.train)

# NA의 수 파악하기
sapply(Titanic.train, function(x) sum(is.na(x)))

# ----------------------------------------------------------------
# 1. EDA

# 변수처리 방향성
# passengerId : index ( 삭제 )
# survived : 생존여부 ( 목표변수 ). factor로 전환하기
# Pclass : 티켓 클레스 1,2,3으로 되어있음. factor로 전환하기
# Name : 이름 ( 삭제 )
# sex : 건들 필요 없음
# age : NA값 처리하기
# sibsp : 형제 및 배우자의 탑승 수 인 것 같음. ( factor or int )
# parch : 부모 및 아이의 탑승 수 인 것 같음. ( factor or int )
# Ticket : 티켓 번호. factor이긴 한데, 레벨 많음 줄여야할듯.
# Fare : 운임. 형식 그대로 유지
# Cabin : factor이긴 한데, 레벨 많음. 또한, 레벨이 ""인 것들이 있음
# Embarked : 특별한 문제는 없으나, 레벨이 ""인 것들이 있음

# PassengerId, Name 지우기
Titanic.train <- Titanic.train %>% dplyr::select(-PassengerId,
                                                 -Name)
str(Titanic.train)

# survived, Pclass factor로 전환하기
Titanic.train$Survived <- as.factor(Titanic.train$Survived)
Titanic.train$Survived <- relevel(Titanic.train$Survived,
                                  ref = '1')
levels(Titanic.train$Survived)

Titanic.train$Pclass <- as.factor(Titanic.train$Pclass)
levels(Titanic.train$Pclass)


# sibsp 처리하기 : factor로 변환하기
table(Titanic.train$SibSp)
Titanic.train$SibSp <- as.factor(Titanic.train$SibSp)

# parch 처리하기 : factor로 변환하기
table(Titanic.train$Parch)
Titanic.train$Parch <- as.factor(Titanic.train$Parch)

# embarked 처리하기 : 레벨 이름 바꾸기
levels(Titanic.train$Embarked)
levels(Titanic.train$Embarked) <- c('unknown','C','Q','S')

# 중요한 문제
# ticket, cabin 처리하기
# age 처리하기 : 이것도 범주형으로 처리하는 것이 더 좋을 것 같다.

# 방법(1) : 일단 범주형으로 전환하고, survived 비율에 따라서 묶기
# 방법(2) : 범주형으로 전환하고, survived 비율에 따라서 묶은 후, trainset에는 있지만, testset에는 없는 데이터를 처리해야한다. 이때 이 데이터의 수를 보고, 데이터의 수가 많은 경우 그대로 두어야하고, 적은 경우 그냥 unknown으로 퉁친다. 일단 trainset과 validationset에서는 1번의 방법을 사용한다.

# 의문점: NA값에서 survived가 0인 것 따로 떼어내어서 그룹화하고, 1인것 따로 떼어내서 그룹화하면 더 좋을까? 안 좋을까?

table(Titanic.train$Age)
table(Titanic.train$Ticket)
table(Titanic.train$Cabin)

# 일단 age의 NA값 처리하기
# age = NA들의 survived 빈도
table(Titanic.train[is.na(Titanic.train$Age), 'Survived'])

# NA 1과 0 분리해서 그룹화하기
Titanic.train$Age <- as.character(Titanic.train$Age)

# 0인 것들을 집합으로 만들기
Titanic.train[is.na(Titanic.train$Age) & Titanic.train$Survived == 0,
              'Age'] <- 'unknown0'

# 1인 것들을 집합으로 만들기
Titanic.train[is.na(Titanic.train$Age) & Titanic.train$Survived == 1,
              'Age'] <- 'unknown1'

# factor로 변환
Titanic.train$Age <- as.factor(Titanic.train$Age)

# 잘 변환 되었는지 확인하기
by(Titanic.train$Survived,
   Titanic.train$Age,
   table)


# 이제 3가지 변수들, 레벨 10개 이하로 축소시키기
change.name <- c('Fare','Age','Ticket','Cabin')

for(i in 1:length(change.name)){
  
  # charactor로 변환
  Titanic.train[, change.name[i]] <- as.character(Titanic.train[, change.name[i]])
  
  # 레벨들에 따른 survived를 확인하기 위한 변수 detector
  detector <- by(Titanic.train$Survived, 
                 Titanic.train[, change.name[i]], 
                 table)
  
  # 레벨에 따른 빈도율을 저장해놓을 벡터
  detector.vector <- c()
  
  # 각 레벨에 따른 빈도율을 구해서 저장한다음, 그 빈도율에 따라서 이름을 변경한다. 그후, 마지막에는 factor로 변환한다. 
  for(k in 1:length(detector)){
  
    # 빈도율
    detector.prop <- detector[[k]][1] / (detector[[k]][1] + detector[[k]][2]) * 100
      
    # rbind로 저장하기
    detector.vector <- rbind(detector.vector, detector.prop)
    
  }
  
  # 구간
  detector.factor <- cut(detector.vector,
                         breaks = seq(from = 0,
                                      to = 100,
                                      by = 10),
                         right = FALSE)
  
  
  for(k in 1:length(detector)){
    
    
    # 빈도율이 100인 경우
    if(is.na(detector.factor[k])){
      
      Titanic.train[Titanic.train[,change.name[i]] == names(detector)[k],change.name[i]] <- '[ukn]'
      
      # 이 외
    }else{
      
      Titanic.train[Titanic.train[,change.name[i]] == names(detector)[k],change.name[i]] <- as.character(detector.factor[k])
      
    }
    
  }
  
  # factor로 변환
  Titanic.train[, change.name[i]] <- as.factor(Titanic.train[, change.name[i]])
  
}

# 제대로 변환하였는지 확인하기
for(i in 1:length(change.name)){
  
  cat('변수의 이름 : ', change.name[i], '\n')
  cat('레벨의 수 : ',nlevels(Titanic.train[, change.name[i]]), '\n')
  
}

# trainset과 validationset으로 나누기
set.seed(123)

index <- sample(1:2,
                size = nrow(Titanic.train),
                prob = c(0.7,0.3),
                replace = TRUE)


Titanic.train.t <- Titanic.train[index == 1, ]
Titanic.train.v <- Titanic.train[index == 2, ]

nrow(Titanic.train.t)
nrow(Titanic.train.v)


# 모형 적합 ------------------------------------------------------------

# 의사결정나무 적합하기

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


# 예측 수치 확인하기
confusionMatrix(pred,real, positive = '1')
F1_Score(pred,real)

# --------------------------------------------------------------------

# 랜덤포레스트 모형 적합하기

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

# 예측 수치 확인하기
confusionMatrix(pred,real, positive = '1')
F1_Score(pred,real)

# -------------------------------------------------------------------

# xgboost 모형 적합하기
# 모든 범주형 자료를 더미변수로 전환하고, 모형에 적합하도록 하겠다.

# 더미변수화 할 데이터
Titanic.train.dummy <- Titanic.train
Survived <- Titanic.train.dummy[, 1] # 목표변수
Survived <- ifelse(Survived == 0,0,1)
str(Survived)
Titanic.train.dummy <- Titanic.train.dummy[,-1] # 목표변수를 제외한 데이터
str(Titanic.train.dummy)

# 더미변수 만드는 함수 ( 더미변수들을 data.frame으로 반환 )
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

# 더미변수화 할 변수의 이름들
factor.name <- colnames(Titanic.train.dummy)
factor.name


for(i in 1:length(factor.name)){
  
  Titanic.train.dummy <- cbind(Titanic.train.dummy,
                               MakingDummy(Titanic.train.dummy[,factor.name[i]], factor.name[i]))
  
}

# 필요없는 데이터 제거하고, 목표변수 통합하기
Titanic.train.dummy <- Titanic.train.dummy[,-c(1:9)]
Titanic.train.dummy <- cbind(Titanic.train.dummy, Survived)
ncol(Titanic.train.dummy)
str(Titanic.train.dummy)

# trainset과 validationset으로 나누기
Titanic.train.dummy.t <- Titanic.train.dummy[index == 1, ]
Titanic.train.dummy.v <- Titanic.train.dummy[index == 2, ]

# xgboost 적합하기 ( 디폴트값으로 )
dtrain <- xgb.DMatrix(data = as.matrix(Titanic.train.dummy.t[ ,-57]),
                      label= as.matrix(Titanic.train.dummy.t[ , 57]))

# 파라미터
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds 찾기
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost 모형 적합하기
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(Titanic.train.dummy.v[, -57]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- Titanic.train.dummy.v[, 57] %>% as.factor() %>% relevel(ref = '1')

# 혼동행렬
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)


# ---------------------------------------------------------------
# ---------------------------------------------------------------

# testset으로 예측해서, 제출해보기

# train과 test를 병합한 마스터 데이터, 
Titanic.master <- rbind(Titanic.train[,-2], Titanic.test)
Titanic.object <- Titanic.train[,2] %>% as.factor()
Titanic.object <- relevel(Titanic.object, ref = '1')

# NA 파악하기
sapply(Titanic.master, function(x) sum(is.na(x)))

# PassengerId, Name 지우기
Titanic.master <- Titanic.master %>% dplyr::select(-PassengerId,
                                                   -Name)
str(Titanic.master)

# Pclass factor로 전환하기
Titanic.master$Pclass <- as.factor(Titanic.master$Pclass)
levels(Titanic.master$Pclass)

# sibsp 처리하기 : factor로 변환하기
table(Titanic.train$SibSp)
Titanic.master$SibSp <- as.factor(Titanic.master$SibSp)
levels(Titanic.master$SibSp)

# parch 처리하기 : factor로 변환하기
table(Titanic.master$Parch)
Titanic.master$Parch <- as.factor(Titanic.master$Parch)
levels(Titanic.master$Parch)

# embarked 처리하기 : 레벨 이름 바꾸기
levels(Titanic.master$Embarked) <- c('unknown','C','Q','S')
levels(Titanic.master$Embarked)

# age 처리하기 : factor로 변환
# 여기서는 조금 복잡하다.
# 일단 survived에 따라서 NA값들을 묶을 수는 있다. 하지만, train에는 없지만, test에는 있는 NA는? survived에 따라서 묶기 어렵기 때문에 다른 방법을 생각해야 한다. 따로 unknown으로 묶는 방법밖에 없는 것 같다.

# trainset 부근의 현황 확인하기 
by(Titanic.object,
   Titanic.master[1:nrow(Titanic.train), 'Age'],
   table)

# NA 1과 0 분리해서 그룹화하기
Titanic.master$Age <- as.character(Titanic.master$Age)

# 0인 것들을 집합으로 만들기

Titanic.temp <- Titanic.master[1:length(Titanic.object), 'Age']
Titanic.temp[is.na(Titanic.temp) & Titanic.object == 0] <- 'unknown0'
Titanic.master[1:length(Titanic.object), 'Age'] <- Titanic.temp

# 1인 것들을 집합으로 만들기
Titanic.temp <- Titanic.master[1:length(Titanic.object), 'Age']
Titanic.temp[is.na(Titanic.temp) & Titanic.object == 1] <- 'unknown1'
Titanic.master[1:length(Titanic.object), 'Age'] <- Titanic.temp

# 이제 testset 부근의 86개가 남아있음
# unknownTest 라고 묶어버리자
Titanic.master[is.na(Titanic.master$Age), 'Age'] <- 'unknownTest'

# factor로 변환
Titanic.master$Age <- as.factor(Titanic.master$Age)

# 제대로 묶었는지 확인해보기
table(Titanic.master$Age)

# Fare의 na값 처리하기 ( 그냥 중위수로 )
table(Titanic.master$Fare)
Titanic.master[is.na(Titanic.master$Fare), 'Fare'] <- 14.4542
Titanic.master$Fare <- as.factor(Titanic.master$Fare)

# 변환 잘 되었는지 확인하기
str(Titanic.master)
str(Titanic.object)


# -----------------------------------------------------------------

# 레벨 수 10개 이하로 축소하기 ( 가장 어려운 작업 )

# 일단 빈도를 통해서 축소시킬 수 있다면, 최대한 빈도를 활용해서 축소한다.
# 빈도를 통해서 축소하지 못하는 것들은 어쩔 수 없이 합쳐버린다.

# 이제 3가지 변수들, 레벨 10개 이하로 축소시키기
change.name <- colnames(Titanic.master)
change.name

# 레벨 축소 작업
for(i in 1:length(change.name)){
  
  # 레벨들에 따른 survived를 확인하기 위한 변수 detector
  detector <- by(Titanic.object, 
                 Titanic.master[1:length(Titanic.object), change.name[i]], 
                 table)
  
  # charactor로 변환
  Titanic.master[, change.name[i]] <- as.character(Titanic.master[, change.name[i]])
  
  # 레벨에 따른 빈도율을 저장해놓을 벡터
  detector.vector <- c()
  
  # 각 레벨에 따른 빈도율을 구해서 저장한다음, 그 빈도율에 따라서 이름을 변경한다. 그후, 마지막에는 factor로 변환한다. 
  for(k in 1:length(detector)){
    
    # trainset에서 관측되지 않는 레벨의 경우
    if(is.null(detector[[k]])){
      
      detector.prop <- 110
      detector.vector <- rbind(detector.vector, detector.prop)
      
      # trainset에 관측되는 레벨의 경우
    }else{
      
      detector.prop <- 100 * detector[[k]][1] / (detector[[k]][1] + detector[[k]][2])
      
      detector.vector <- rbind(detector.vector, detector.prop)
      
    }
    
  }
  
  # 구간
  detector.factor <- cut(detector.vector,
                         breaks = seq(from = 0,
                                      to = 110,
                                      by = 10),
                         right = FALSE)
  
  
  for(k in 1:length(detector)){
    
    
    # 빈도율이 100인 경우
    if(is.na(detector.factor[k])){
      
      Titanic.master[Titanic.master[,change.name[i]] == names(detector)[k],change.name[i]] <- '[ukn]'
      
      # 이 외
    }else{
      
      Titanic.master[Titanic.master[,change.name[i]] == names(detector)[k],change.name[i]] <- as.character(detector.factor[k])
      
    }
    
  }
  
  # factor로 변환
  Titanic.master[, change.name[i]] <- as.factor(Titanic.master[, change.name[i]])
  
}


# 가공 결과 확인하기
for(i in 1:length(change.name)){
  
  cat('변수의 이름 : ', change.name[i], '\n')
  cat('레벨의 수 : ',nlevels(Titanic.master[, change.name[i]]), '\n')
  
}

str(Titanic.master)
str(Titanic.object)

# -------------------------------------------------------------------

# xgboost 모형 적합하기 위한 데이터 전처리
# 모든 범주형 자료를 더미변수로 전환하고, 모형에 적합하도록 하겠다.

# 더미변수화 할 데이터
Titanic.master.dummy <- Titanic.master
Survived <- Titanic.object # 목표변수
Survived <- ifelse(Survived == 0,0,1)
str(Titanic.master.dummy)
str(Survived)

# 더미변수 만드는 함수 ( 더미변수들을 data.frame으로 반환 )
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

# 더미변수화 할 변수의 이름들
factor.name <- colnames(Titanic.master.dummy)
factor.name


for(i in 1:length(factor.name)){
  
  Titanic.master.dummy <- cbind(Titanic.master.dummy,
                                MakingDummy(Titanic.master.dummy[,factor.name[i]], factor.name[i]))
  
}

# 필요없는 데이터 제거하고, 목표변수 통합하기
Titanic.master.dummy <- Titanic.master.dummy[,-c(1:9)]
str(Titanic.master.dummy)


# -----------------------------------------------------------------

# xgboost 적합하기

dtrain <- xgb.DMatrix(data = as.matrix(Titanic.master.dummy[1:length(Titanic.object), ]),
                      label= as.matrix(Titanic.object))

# 파라미터
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds 찾기
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost 모형 적합하기
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


# 예측하기
pred <- predict(newxgb, as.matrix(Titanic.master.dummy[-c(1:length(Titanic.object)), ]))
pred <- ifelse(pred > 0.5, 1, 0)


# -----------------------------------------------------------------------

# 제출파일 정리 / 예측값 저장하기

# csv 불러오기
submission <- read.csv(file = 'gender_submission.csv',
                       header = TRUE)

# 구조 파악하기
nrow(submission)
str(submission)

# 기존의 Survived 삭제하기
submission <- submission[,-2]

# pred를 새로운 Survived로 추가하기
submission <- data.frame(PassengerId = submission,
                         Survived = pred)

# 잘 정리했는지 확인하기
str(submission)

# 저장하기
write.csv(submission,
          file = 'submission.csv',
          row.names = FALSE)




# 여담으로, 이 예측값은 예측스코어 0.72727 로서, 8971위를 하였다.
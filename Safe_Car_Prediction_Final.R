
# Objective: To build models that will predict whether a driver will initiate an auto insurance claim in the next year

# Input: car_insurance_claim.csv (Dataset is also avaialbe on Kaggle but didn't have any Kernal discussion when I worked on this project)

# Workflow: data exploratory analysis and visualization - > 
#           data-preprocessing -> 
#           splitting data to train and test datasets -> 
#           buiding 9 classification models ->
#           results visualization    

# How to run the code: 
# 1) Put the input dataset ("car_insurance_claim.csv") and R code ("data_preprocess_models.R") under the same directory of you PC
# 2) In RSutdio, setting the working directory to the directory that you save the dataset and R code
# 3) Run the R code


if (!require('corrplot')) install.packages('corrplot')
library(corrplot)
library(dplyr) # missing replacement

if (!require('Matrix')) install.packages('Matrix')
library(Matrix)


library(class)
library(lattice)
library(ggplot2)
library(caret)


df_car <- read.csv('car_insurance_claim.csv',na.strings="") # without na.strings="", OCCUPATION would not be shown
str(df_car) # 10302 * 27
############ Data Exploration; Assign Factor
#removing the id and birth date
df0 <- df_car[,-c(1,3)] # 10302 * 25
sum(is.na(df0)) #3004
# check missing values
Misstab <- data.frame() # create missing values table
for (i in (1:ncol(df0))){
  Misstab[1,i] = sum(is.na(df0[,i]))
  colnames(Misstab)[i] = colnames(df0)[i]
  row.names(Misstab) = "count"
}
Misstab # missing values dist AGE:7, YOJ:548, INCOME:570, HOME_VAL:575, OCCUPATION 665, CAR_AGE:639
# assign factor
str(df0) # check the classes
num = c() # store the # of unique variable
for (i in (1:ncol(df0))){
  num[i] = length(unique(df0[,i]))
}
nums = which(num > 2) # if # of unique variable > 2 --> num; <=2 assign as factors
df0[-nums] <- lapply( df0[-nums], factor)
summary(df0) # shows the summary statistitics  
df0 = df0[-which(df0$CAR_AGE==-3),] # Remove one abnormal row CAR_AGE = -3 ## 10301* 25

############ Corrplot including KIDSDRIV; HOMEKIDS; CLM_FREQ
cornums = which(sapply(df0, is.numeric))
num_cols <- df0[cornums]
M = cor(na.omit(num_cols))
windows(width = 12, height = 12)
col <- colorRampPalette(c("red","white", "blue"))

corrplot(M, method="color", col=col(150),  
         type="upper",order = "hclust",
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45 #Text label color and rotation
)

############ General Freq Plots, KIDSDRIV; HOMEKIDS; CLM_FREQ assign as factors 
###### IMPORTANT NOTE
# nums = which(num > 6) # if # of unique variable > 6 --> num; <=6 assign as factors BUT
# It would make later cols to NearZeroVars,
# "KIDSDRIV.2,KIDSDRIV.3,KIDSDRIV.4,HOMEKIDS.4,HOMEKIDS.5,CLM_FREQ.4,CLM_FREQ.5"
# Therefore, we keep them as numeric by # of unique variable > 2
# nums = which(num > 2) # if # of unique variable > 2 --> num; <=2 assign as factors
# df0[-nums] <- lapply( df0[-nums], factor)
###### IMPORTANT
summary(df0) # check the summary 
# Plot
windows(width = 12, height = 12)
par(mfrow=c(5,5))
# Visually display all variables: numeric = yellow; categorical = red; Response = green
nm <- names(df0)
for (i in seq_along(nm)){
  if(is.numeric(df0[,i])==FALSE){
    if(names(df0)[i]=="CLAIM_FLAG"){
      plot(df0[,i],xlab = '',main = nm[i],col = "green")
    }else{
    plot(df0[,i],xlab = '',main = nm[i],col = "red")
    }
  }else{
    hist(df0[,i],xlab = '',main = nm[i],col = "yellow")
  }
}

############ Assign Factors with unique values = 2 to 0,1 ; Keep Y as one Factor

Fact_ind = which(num == 2) # will be assign as 0; 1
# Others will be applied one-hot encoding: EDUCATION, 

# explore col's levels with 2 unique values to determine how to assign values to 0,1
df_Fact_name = data.frame() # store the colname and their levels
for (i in (1:length(Fact_ind))){
  df_Fact_name[1,i] = levels(df0[,Fact_ind[i]])[1]
  df_Fact_name[2,i] = levels(df0[,Fact_ind[i]])[2]
  colnames(df_Fact_name)[i] = colnames(df0)[Fact_ind[i]]
}
df_Fact_name # check the stored colnames and their levels
### transform 2 factors to 0, 1 based on their values
levels(df0[,Fact_ind[1]]) = c(0,1) # No->0, Yes->1, PARENT1
levels(df0[,Fact_ind[2]]) = c(1,0) # No->0, Yes->1, MSTATUS
levels(df0[,Fact_ind[3]]) = c(1,0) # F->0,  M->1  , GENDER
levels(df0[,Fact_ind[4]]) = c(1,0) # Commercial->0, Private->1, CAR_USE
levels(df0[,Fact_ind[5]]) = c(0,1) # No->0, Yes->1, RED_CAR
levels(df0[,Fact_ind[6]]) = c(0,1) # No->0, Yes->1, REVOKED
levels(df0[,Fact_ind[8]]) = c(1,0) # Rural->0, Urban->1, URBANICITY
### transform 2 factors (0,1) to numeric for later one-hot encoding
for (i in (1:length(Fact_ind))){
  if(names(df0)[Fact_ind[i]]!="CLAIM_FLAG"){
    df0[,Fact_ind[i]] = as.numeric(levels(df0[,Fact_ind[i]]))[df0[,Fact_ind[i]]]
    }
}
levels(df0$CLAIM_FLAG) = c("No","Yes") # 0->No, 1->Yes, Y,CLAIM_FLAG
str(df0[Fact_ind])# check all are transformed to numeric and Y to No, Yes


############ One-hot encoding for without Y after split to train and test DUE TO missing categorical data

# df00_outY = df0[,-which(names(df0)=="CLAIM_FLAG")] # df00take out Y 10301* 24
# dmy <- dummyVars("~.",data= df00_outY)
# OH_data <- data.frame(predict(dmy,newdata= df00_outY)) # 10301* 40
# OH_data$CLAIM_FLAG = df0[,which(names(df0)=="CLAIM_FLAG")] # adding Y back so NOW 10301* 41

############ Split to Train and Test 8:2 Before NearZeroVar removed

nearZeroVar(df0) # original data 0

### Split to Train and Test by createDataPartition
set.seed(200)
temp =  createDataPartition(df0$CLAIM_FLAG, p = .8, list= FALSE)
train <- df0[temp,]    # 8241* 25
dim(train) # 8242 * 25
sum(train$CLAIM_FLAG == "Yes") #2196
sum(train$CLAIM_FLAG == "No") #6045
test <- df0[-temp,]    # 2060   25
dim(test) # 2060 * 25
sum(test$CLAIM_FLAG == "Yes") # 549
sum(test$CLAIM_FLAG == "No") #1511
sum(df0$CLAIM_FLAG == "Yes") #2745
sum(df0$CLAIM_FLAG == "No") #7556

############ Dealing Missing values to Median Train and Test !!!Separately!!!
### For Train
# check missing values
sum(is.na(train)) #2433 All 3004 
Misstab_train <- data.frame()
for (i in (1:ncol(train))){
  Misstab_train[1,i] = sum(is.na(train[,i]))
  colnames(Misstab_train)[i] = colnames(train)[i]
  row.names(Misstab_train) = "count"
}
Misstab_train # missing values dist # AGE 7, YOJ 439, INCOME 476, HOME_VAL 455, OCCUPATION 539, CAR_AGE 517
misindex_train = which(Misstab_train > 0) # col index of having missing values # 2, 4, 5, 7, 11, 23
mislist_train = c(colnames(train[misindex_train])) # store the colnumn having missing values
M_train = train
# we assign median for numeric to these missing values (in order to have same str "int" and actual data point)
# we assign median for categorical to these missing values 
#install.packages("aCRM") 
if (!require('aCRM')) install.packages('aCRM')
library("aCRM") # imputeMissings mode and median
M_train = imputeMissings(train)
# ?imputeMissings 
# Character vectors and factors are imputed with the mode. 
# Numeric and integer vectors are imputed with the median.
summary(M_train) # check the summary 
sum(is.na(M_train)) #result is 0 

############ SMOTE Train set should be BEFORE one-hot to avoid duplicate value in the same category
library(lattice)
if (!require('grid')) install.packages('grid')
library(grid)
if (!require('DMwR')) install.packages('DMwR')
library(DMwR)
# SMOTE will have data out of values 0,1; therefore, we assign Factor to keep values first then transform them back
num_SMOTE = c() # store the # of unique variable
for (i in (1:ncol(M_train))){
  num_SMOTE[i] = length(unique(M_train[,i]))
}
nums_SMOTE = which(num_SMOTE > 6)  #if # of unique variable > 6 --> num; <=6 assign as factors
s_train = M_train
s_train[-nums_SMOTE] <- lapply( s_train[-nums_SMOTE], factor)
str(s_train)
s_train <- SMOTE(CLAIM_FLAG ~ .,s_train,perc.over = 100,perc.under = 200)
dim(s_train) #8784 25
sum(s_train$CLAIM_FLAG == "Yes") #4392
sum(s_train$CLAIM_FLAG == "No") #4392
# transform back to numeric
Back_ind = which(num_SMOTE < 7)
Back_ind2 = Back_ind[which((names(s_train)[Back_ind]!="EDUCATION")&(names(s_train)[Back_ind]!="CAR_TYPE"))]
for (i in (1:length(Back_ind2))){
  if((names(s_train)[Back_ind2[i]]!="CLAIM_FLAG")){
    s_train[,Back_ind2[i]] = as.numeric(levels(s_train[,Back_ind2[i]]))[s_train[,Back_ind2[i]]]
  }
}
str(s_train)
#### One-hot encoding for Train

M_train_outY = s_train[,-which(names(s_train)=="CLAIM_FLAG")] # df00take out Y 8784* 24
dmy_tr <- dummyVars("~.",data= M_train_outY)
OH_data_tr <- data.frame(predict(dmy_tr,newdata= M_train_outY)) # 8784* 40
OH_data_tr$CLAIM_FLAG = s_train[,which(names(s_train)=="CLAIM_FLAG")] # adding Y back so NOW 8784* 41

#### Remove nearZeroVar for OH_data
nZ = nearZeroVar(OH_data_tr) # 16
N_nZ = names(OH_data_tr)[nZ] # "OCCUPATION.Doctor"
table(OH_data_tr$OCCUPATION.Doctor==1,OH_data_tr$CLAIM_FLAG) # OCCUPATION.Doctor= TRUE--> NO:168; YES:53
# Remove column "OCCUPATION.Doctor"
# Remove All rows OCCUPATION.Doctor = 1 DUE to one-hot encoding to avoid row having all zero in OCCUPATION
nZ_OH_data_tr = OH_data_tr[-which(OH_data_tr[,nZ]>0),] #8563 * 41
nZ_OH_data_tr = nZ_OH_data_tr[-nZ] # 8563* 40


### For Test
# check missing values
sum(is.na(test)) #571 All 3004
Misstab_test <- data.frame()
for (i in (1:ncol(test))){
  Misstab_test[1,i] = sum(is.na(test[,i]))
  colnames(Misstab_test)[i] = colnames(test)[i]
  row.names(Misstab_test) = "count"
}
Misstab_test # missing values dist # YOJ 109, INCOME 94, HOME_VAL 120, OCCUPATION 126, CAR_AGE 122
misindex_test = which(Misstab_test > 0) # col index of having missing values # 4, 5, 7, 11, 23
# Character vectors and factors are imputed with the mode. 
# Numeric and integer vectors are imputed with the median.
M_test = imputeMissings(test)
summary(M_test) # check the summary 
sum(is.na(M_test)) #0

#### One-hot encoding for Test
M_test_outY = M_test[,-which(names(M_test)=="CLAIM_FLAG")] # M_test out Y 2060* 24
dmy_ts <- dummyVars("~.",data= M_test_outY)
OH_data_ts <- data.frame(predict(dmy_ts,newdata= M_test_outY)) # 2060* 40
OH_data_ts$CLAIM_FLAG = M_test[,which(names(M_test)=="CLAIM_FLAG")] # adding Y back so NOW 2060* 41
?dummyVars
#### Remove OCCUPATION.Doctor for test
nZ_OH_data_ts = OH_data_ts[-which(OH_data_ts[,nZ]>0),] #1996 * 41
nZ_OH_data_ts = nZ_OH_data_ts[-nZ] # 1996* 40

############ SMOTE Train set should be BEFORE one-hot
# library(lattice)
# library(grid)
# library(DMwR)
# # SMOTE will have data out of values 0,1; therefore, we assign Factor to keep values first then transform them back
# num_SMOTE = c() # store the # of unique variable
# for (i in (1:ncol(nZ_OH_data_tr))){
#   num_SMOTE[i] = length(unique(nZ_OH_data_tr[,i]))
# }
# nums_SMOTE = which(num_SMOTE > 6)  #if # of unique variable > 6 --> num; <=6 assign as factors
# s_train = nZ_OH_data_tr
# s_train[-nums_SMOTE] <- lapply( s_train[-nums_SMOTE], factor)
# summary(s_train)
# s_train <- SMOTE(CLAIM_FLAG ~ .,s_train,perc.over = 100,perc.under = 200)
# dim(s_train) #8784 40
# sum(s_train$CLAIM_FLAG == "Yes") #4392
# sum(s_train$CLAIM_FLAG == "No") #4392
# # transform back to numeric
# Back_ind = which(num_SMOTE < 7)
# for (i in (1:length(Back_ind))){
#   if(names(s_train)[Back_ind[i]]!="CLAIM_FLAG"){
#     s_train[,Back_ind[i]] = as.numeric(levels(s_train[,Back_ind[i]]))[s_train[,Back_ind[i]]]
#   }
# }
# summary(s_train)

############ Remove column, CLM_AMT ########################################
nZ_OH_data_tr = nZ_OH_data_tr[,-which(names(nZ_OH_data_tr)=="CLM_AMT")] #8563 *39
nZ_OH_data_ts = nZ_OH_data_ts[,-which(names(nZ_OH_data_ts)=="CLM_AMT")] #1996 *39


### Creat Train,Test X and Y

set.seed(100)
nZ_OH_data_tr <- nZ_OH_data_tr[sample(nrow(nZ_OH_data_tr)),]  # shuffle train dataset
trainX <- nZ_OH_data_tr[, -which(colnames(nZ_OH_data_tr)=="CLAIM_FLAG")]
trainY<- nZ_OH_data_tr[,which(colnames(nZ_OH_data_tr)=="CLAIM_FLAG")]
testX <- nZ_OH_data_ts[, -which(colnames(nZ_OH_data_ts)=="CLAIM_FLAG")]
testY<- nZ_OH_data_ts[,which(colnames(nZ_OH_data_ts)=="CLAIM_FLAG")]

dim(testX)


### write CSV
#write.csv(nZ_OH_data_tr,file='SMOTE_Train_Remove_OCCUPATION.Doctor.csv')
#write.csv(nZ_OH_data_ts,file='Test_Remove_OCCUPATION.Doctor.csv')


############### Models ####################################################
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     number = 10)

if (!require('PRROC')) install.packages('PRROC')
library(PRROC)
library(pROC) 


######## Naive Bayes model ################
if (!require('klaR')) install.packages('klaR')
library(klaR)

set.seed(100)
NbFit <- train(x = trainX,
               y = trainY,
               method = "nb",metric="ROC",
               preProc = c("center","scale"),
               trControl = ctrl)

NbFit
NbImp<-varImp(NbFit)
plot(NbImp, top = 10, main = "Naive Bayes Top 10 Important Variables")

Nbpred <- predict(NbFit, testX)
Nbpred

Nbpr <- postResample(pred = Nbpred, obs = testY)
Nbpr

confusionMatrix(Nbpred, testY) 
Nbpred1 <- ifelse(Nbpred=="Yes", 1, 0)
PRROC_obj_nb <- roc.curve(scores.class0 = testY, weights.class0=Nbpred1,
                          curve=TRUE)
plot(PRROC_obj_nb)

 
#PLS model #####################################
library(pls) 
set.seed(100)
plsFit <- train(x = trainX,
                y = trainY,
                method = "pls",tuneGrid = expand.grid(ncomp = 1:10),
                metric="ROC",
                preProc = c("center","scale"),
                trControl = ctrl)
plsFit
plot(plsFit, main = "PLS tuning plot for number of components")

plsImp<-varImp(plsFit)
plot(plsImp, top = 10, main = "PLS Top 10 Important Variables")

plspred <- predict(plsFit, testX)
plspred

plspr <- postResample(pred = plspred, obs = testY)
plspr

confusionMatrix(plspred, testY) 

plspred1 <- ifelse(plspred=="Yes", 1, 0)

PRROC_obj_pls <- roc.curve(scores.class0 = testY, weights.class0=plspred1,
                       curve=TRUE)
plot(PRROC_obj_pls)


#Mars#####################################################################
library(earth)

marsGrid = expand.grid(.degree=1:2, .nprune=2:38)
set.seed(100)
marsFit <- train(x = trainX, 
                 y = trainY,
                 method="earth", 
                 preProc=c("center", "scale"),
                 metric="ROC",trControl = ctrl)
marsFit 
plot(marsFit, main = "MARS tuning plot for number of terms")

marsImp<-varImp(marsFit)
plot(marsImp, top = 10, main = "MARS Top 10 Important Variables")

marspred <- predict(marsFit, testX)
marspred

marspr <- postResample(pred = marspred, obs = testY)
marspr 

confusionMatrix(marspred, testY) 

marspred1 <- ifelse(marspred=="Yes", 1, 0)

PRROC_obj_mars <- roc.curve(scores.class0 = testY, weights.class0=marspred1,curve=TRUE)
plot(PRROC_obj_mars)

## XGboost ##################################################################
library(gbm)

set.seed(100)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*30, 
                        shrinkage = c(0.001, 0.005, 0.01),
                        n.minobsinnode = 20)

xgbmodel=train(x = trainX, y = trainY, 
               method="gbm", preProc=c("center","scale"),
               metric="ROC", verbose=FALSE, 
               trControl = ctrl, tuneGrid=gbmGrid)
xgbmodel
plot(xgbmodel, main = "XGBoost tuning plot")


xgbImp<-varImp(xgbmodel)
plot(xgbImp, top = 10, main = "XGBoost Top 10 Important Variables")


xgbpred <- predict(xgbmodel, testX)
xgbpred

xgbpr <- postResample(pred = xgbpred, obs = testY)
xgbpr 

confusionMatrix(xgbpred, testY) 

xgbpred1 <- ifelse(xgbpred=="Yes", 1, 0)

PRROC_obj_xgb <- roc.curve(scores.class0 = testY, weights.class0=xgbpred1,curve=TRUE)
plot(PRROC_obj_xgb)


## Neural Networks ##################################################
library(nnet)

nnetGrid <- expand.grid(.size = 1:8,.decay = c(0, .1, .2, .5, 1))
maxSize <- max(nnetGrid$.size)
numWts <- 1*(maxSize * (dim(trainX)[2] + 1) + maxSize + 1)

set.seed(100)
nnetFit <- train(x = trainX,
                 y = trainY,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)

nnetFit
plot(nnetFit, main = "Neural Networks tuning plot")
 
nnetImp<-varImp(nnetFit)
plot(nnetImp, top = 10, main = "Neural Networks Top 10 Important Variables")

nnetpred <- predict(nnetFit, testX)
nnetpred

nnetpr <- postResample(pred = nnetpred, obs = testY)
nnetpr 

confusionMatrix(nnetpred, testY) 

nnetpred1 <- ifelse(nnetpred=="Yes", 1, 0)
PRROC_obj_nnet <- roc.curve(scores.class0 = testY, weights.class0=nnetpred1,curve=TRUE)
plot(PRROC_obj_nnet)

# Logistic Regression ################################
library(glmnet) 
set.seed(100)
lrFit <- train(x = trainX,
               y = trainY,
               method = "glm",
               metric = 'ROC',
               preProc = c("center","scale"),
               trControl = ctrl)
lrFit
lrImp <- varImp(lrFit)
plot(lrImp,top=10,main='Logistic Regression Top 10 Important Variables')

lrpred <- predict(lrFit, testX)
lrpred

lrpr <- postResample(pred = lrpred, obs = testY)
lrpr

confusionMatrix(lrpred,testY)

lrpred1 <- ifelse(lrpred =="Yes", 1, 0)
PRROC_obj_lr <- roc.curve(scores.class0 = testY, weights.class0=lrpred1,
                          curve=TRUE)
plot(PRROC_obj_lr)


# knn ##############################################################
library(kknn) 
set.seed(100)
knnFit <- train(x = trainX,
                y = trainY,
                method = "knn",metric="ROC",
                preProc = c("center","scale"),
                tuneGrid = data.frame(k = c(2*(0:12)+1)),
                trControl = ctrl)
knnFit 

#summary(knnFit)
plot(knnFit,main='KNN Tuning Plot for Number of Neighbors')

knnImp <- varImp(knnFit)
plot(knnImp,top=10,main='KNN Top 10 Important Variables')

knnpred <- predict(knnFit, testX)
knnpred

knnpr <- postResample(pred = knnpred, obs = testY)
knnpr 

confusionMatrix(knnpred,testY)

knnpred <- ifelse(knnpred=="Yes", 1, 0)
PRROC_obj_knn <- roc.curve(scores.class0 = testY, weights.class0=knnpred,
                           curve=TRUE)
plot(PRROC_obj_knn)


# SVM ######################################
if (!require('tictoc')) install.packages('tictoc')
install.packages('tictoc')
library(tictoc)
tic()
library(kernlab)
set.seed(100)

svmFit <- train(x = trainX, y = trainY,method="svmRadial", preProc=c("center", "scale"),metric="ROC",trControl = ctrl,tuneLength=10)
svmFit

plot(svmFit,main='SVM Tuning Plot')

svmImp <- varImp(svmFit)
plot(svmImp, top = 10, main = "SVM Top 10 Important Variables")


svmpred <- predict(svmFit, testX)
svmpred

svmspr <- postResample(pred = svmpred, obs = testY)
svmspr 
confusionMatrix(svmpred,testY)

svmpred1 <- ifelse(svmpred=="Yes", 1, 0)
PRROC_obj_svm <- roc.curve(scores.class0 = testY, weights.class0=svmpred1,curve=TRUE)
plot(PRROC_obj_svm)
toc()


# Random Forest ###################################
if (!require('randomForest')) install.packages('randomForest')
library(randomForest)
tic()
set.seed(100)

ctrl_rf <- trainControl(method = "cv",
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        number = 10,
                        search='grid')


tunegrid <- expand.grid(.mtry=c(1:15))

rfFit <- train(x = trainX, y = trainY,method="rf", preProc=c("center", "scale"),tuneGrid = tunegrid,metric="ROC",trControl = ctrl_rf)
rfFit

plot(rfFit,main='Random Forest Tuning Plot')


rfImp <- varImp(rfFit)
plot(rfImp, top = 10, main = "Random Forest Top 10 Important Variables")

rfpred <- predict(rfFit, testX)
rfpred
rfpr <- postResample(pred = rfpred, obs = testY)
rfpr 

confusionMatrix(rfpred,testY)

rfpred <- ifelse(rfpred=="Yes", 1, 0)
PRROC_obj_rf <- roc.curve(scores.class0 = testY, weights.class0=rfpred,curve=TRUE)
plot(PRROC_obj_rf)

toc()


# Results visualization ##################################
library(ggplot2)
#Plot-Accuracy&Kappa
method<-c("XGBoost","Neural Networks","MARS","Naive Bayes","SVM","Random Forest","Logistic Regression","K-Nearest Neighbours","PLS-DA")
Index<-rep(c('Accuracy','Kappa'),each=9)
value<-c(0.782,0.779,0.769,0.767,0.759,0.758,0.754,0.754,0.751,0.457,0.451,0.429,0.311,0.365,0.433,0.413,0.365,0.405)
df <- data.frame(method = method, Index = Index, value = value)
ggplot(df,aes(x=method,y=value,colour=Index,group=Index,fill=Index)) +
  geom_line(size =0.8)+geom_point(size=1.5)+geom_text(aes(label = value, vjust = 1.1, hjust = 0.5, angle = 0), show.legend = FALSE)+ ggtitle("Accuracy&Kappa")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_color_manual(values=c('#002060','#C00000'))

#plot-AUC&Sensitivity&Specificity
Method<-c("XGBoost","Neural Networks","MARS","Naive Bayes","SVM","Random Forest","Logistic Regression","K-Nearest Neighbours","PLS-DA")
Index<-rep(c('AUC','Sensitivity','Specificity'),each=9)
Value<-c(0.726,0.723, 0.710,0.714,0.701,0.706,0.698,0.687,0.695,0.844,0.840,0.828,0.929,0.811,0.788,0.795,0.844,0.796,0.618,0.617,0.609,0.336,0.618,0.677,0.642,0.511,0.631 )
df <- data.frame(method = method, Index = Index, value = Value)
ggplot(df, aes(x=method, y=Value, fill=Index)) + geom_bar(stat="identity", position="dodge")+geom_text(aes(label = Value), vjust = 1.5, colour = "black", position = position_dodge(.9), size = 5)



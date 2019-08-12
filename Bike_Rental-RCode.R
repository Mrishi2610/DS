rm(list=ls())
getwd()

#setting working directory
setwd("/Users/rishi/Desktop/All/edWisor/Project1")
#installing required packages
install.packages("dplyr", "ggplot2", "MLmetrics")
library("MASS", "dplyr")
#importing the bike rental dataset
dfmain <- read.csv("day.csv")
str(dfmain)

#Feature Engineering (Converting and adding the required variables)
dfmain$season <- as.factor(dfmain$season)
levels(dfmain$season) <- c("spring", "summer", "fall", "winter")
dfmain$yr <- as.factor(dfmain$yr)
levels(dfmain$yr) <- c(2011, 2012)
dfmain$mnth <- as.factor(dfmain$mnth)
levels(dfmain$mnth) <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
dfmain$holiday <- as.factor(dfmain$holiday)
levels(dfmain$holiday) <- c("Not Holiday", "Holiday")
dfmain$weekday <- as.factor(dfmain$weekday)
levels(dfmain$weekday) <- c("Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat")
dfmain$workingday <- as.factor(dfmain$workingday)
levels(dfmain$workingday) <- c("Holiday", "Workingday")
dfmain$weathersit <- as.factor(dfmain$weathersit)
levels(dfmain$weathersit) <- c("Clear", "Cloudy", "Rainy")

dfmain$new_temp <- (dfmain$temp * 47) - 8 #(t-t_min)/(t_max-t_min), t_min=-8, t_max=+39
dfmain$new_atemp <- (dfmain$atemp * 66) - 16 #(t-t_min)/(t_max-t_min), t_min=-16, t_max=+50
dfmain$new_hum <- (dfmain$hum * 100)
dfmain$new_windspeed <- (dfmain$windspeed * 67)

str(dfmain)

#checking for missing values
missingval <- data.frame(apply(dfmain, 2, function(x){sum(is.na(x))}))
names(missingval)[1] <- "Missing Values"
missingval$Percent <- (missingval$`Missing Values`/nrow(dfmain)) *100
missingval
#no any missing values found


#Outlier Analysis
#selecting only numeric variables
dfmain_2 <-dfmain #copy of the dataset
exclude_var <- names(dfmain) %in% c("temp", "atemp", "hum", "windspeed")
dfmain <- dfmain[!exclude_var]  #Removing the unnormalised variables

numeric_index = sapply(dfmain,is.numeric) #checking for numeric variables
numeric_data = dfmain[,numeric_index]
cnames = colnames(numeric_data)


#Plotting Boxplots for every variable
for (i in 1:length(cnames))
{boxplot(dfmain[,i]) }


#Removing the Outliers
val = dfmain$hum[dfmain$hum %in% boxplot.stats(dfmain$hum)$out]
dfmain = dfmain[which(!dfmain$hum %in% val),]
val = dfmain$windspeed[dfmain$windspeed %in% boxplot.stats(dfmain$windspeed)$out]
dfmain = dfmain[which(!dfmain$windspeed %in% val),]


#FEATURE SELECTION
#Correlation test for Numeric Variables
corrgram(dfmain[,numeric_index], order = T,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#chiSq Test for Categorical Variables
factor_index = sapply(dfmain,is.factor)
factor_data = dfmain[,factor_index]
factor_data <- factor_data[-c(1)] #Dropping date column


#loop for Chi-Square test of independence to check multicollinearity between categorical variables
for (i in 1:7)
{
  for (j in 1:7) 
  {
    if (j<=i)
      {next}
    else {
      print(names(factor_data)[i])
      print(names(factor_data)[j])
      print(chisq.test(table(factor_data[,i],factor_data[,j])))
        }
  }
}


#dropping the unwanted variables
dfmain <- select(dfmain,-c("new_atemp", "holiday", "workingday", "casual", "registered"))


#Implementing Linear Regression Model on the dataset
#Dividing the dataset into train and test data
numeric_data <- numeric_data[,-5]
set.seed(123)
train_index = sample(1:nrow(numeric_data), 0.8 * nrow(numeric_data))
train = numeric_data[train_index,]
test = numeric_data[-train_index,]


#Applying Linear Regression Model
#checking for multicollearity first
library(usdm)
vif(numeric_data[,-3])
vifcor(numeric_data[,-3], th = 0.9)

#running Regression Model
lm_model = lm(cnt ~., data = train)

summary(lm_model) #Summary of the model
predictions_LR = predict(lm_model, test[,-3]) #Predicting on Test case
MAPE(test[,-3], predictions_LR) #Calculating MAPE
#Multiple Linear Regression implemented successfully with appreciable accuracy


#Import data file, examine structure
#Set the working directory. 
setwd("E:\\Spring2021\\MA5701\\Final_Project")

library(ggplot2)
library(corrplot)
library(tidyverse)
library(dplyr)
library(purrr)
library(mlbench)
library(corrplot)
library(e1071)
library(naniar)
library(visdat)
library(DataExplorer)
library(VIM)
library(caret)
library(AppliedPredictiveModeling)


#Part 1: Exploratory Analysis. 

df <- read.table('auto-mpg.data')
colnames(df) <- c('MPG','cylinders','displacement', 'horsepower','weight','acceleration',
                  'model_year','origin','car_name')
df$cylinders <- as.factor(df$cylinders)
df$origin <- as.factor(df$origin)
df$model_year <- as.factor(df$model_year)
df$horsepower <- as.numeric(df$horsepower)


# Graphical Display
hist(df$MPG, xlab='MPG',
     main='Histogram of MPG',probability=TRUE)
lines(density(df$V1), col='SteelBlue')

#pair scaterplot
pairs(~MPG+displacement+horsepower+weight+acceleration,data = df)

# paraller boxplot
ggplot(data = df, aes(x = cylinders, y = MPG)) + 
  geom_boxplot( ) +
  labs(title = "Fuel Consumption by Cylinders",
       x = "Cylinders", 
       y = "Fuel (mpg)")

ggplot(data = df, aes(x = origin, y = MPG)) + 
  geom_boxplot( ) +
  labs(title = "Fuel Consumption by Origin",
       x = "Origin", 
       y = "Fuel (mpg)")

summary(df)

pred_con <- c('displacement', 'horsepower','weight','acceleration')
df.Pred.Con <- df[,pred_con]
corrB <- cor(df.Pred.Con)
corrplot(corrB, order = "hclust",tl.pos='n')

# Skeww
swk <- c('displacement', 'horsepower','weight','acceleration','MPG')
df.swk <- df[,swk]
skewValues <- apply(df.swk, 2, skewness)
skewValues <- round(skewValues,2)

#Normal
shapiro.test(df.swk$displacement)
shapiro.test(df.swk$horsepower)
shapiro.test(df.swk$acceleration)
shapiro.test(df.swk$MPG)

####################################################################################################################
####################################################################################################################

#Part 2: Regression Analysis. 

#Reading the Table. 
auto_df <- read.table('auto-mpg.data')

#Adding the Column Names:
cols <- c("mpg","cylinders","displacement",'horsepower','weight','acceleration','model_year','origin','car_name')
colnames(auto_df) <- cols

auto_df$cylinders <- as.factor(df$cylinders)
auto_df$origin <- as.factor(df$origin)
auto_df$model_year <- as.factor(df$model_year)
auto_df$horsepower <- as.numeric(df$horsepower)

#Removing weight from the model since weight and displacement are highly correlated. 
#removing the column car name. 
myvars <- c("mpg","cylinders","displacement",'horsepower','acceleration','model_year','origin')
df_without_weight <- auto_df[myvars]

#Applying Yeo-Johnson transformation to normalize the data
#Yeojohnsontransformation
yeo_transformer <- preProcess(df_without_weight, method=c("YeoJohnson"))
df_without_weight_transformed <- predict(yeo_transformer, df_without_weight)

#Refitting the improved model.
final_model <- lm(formula = mpg ~ acceleration + origin + displacement + horsepower + model_year, 
                  data = df_without_weight)
summary(final_model)
anova(final_model)

#Residual vs Fitted plot to check change in the spread of residuals over a range of values. 
#Residual plot. 
res <- resid(final_model)
plot(fitted(final_model),res, main='Residuals vs fitted values')
abline(0,0)

#residual plot for acceleration.
plot(df_without_weight_transformed$acceleration,res, ylab="Residuals", xlab="acceleration",  main="residuals vs acceleration") 
abline(0, 0) 

#residual plot for origin
plot(df_without_weight_transformed$acceleration,res, ylab="Residuals", xlab="origin",  main="residuals vs origin") 
abline(0, 0) 

#residual plot for displacement
plot(df_without_weight_transformed$acceleration,res, ylab="Residuals", xlab="displacement",  main="residuals vs displacement") 
abline(0, 0) 

#residual plot for horsepower
plot(df_without_weight_transformed$horsepower,res, ylab="Residuals", xlab="horsepower",  main="residuals vs horsepower") 
abline(0, 0) 

#residual plot for model_year
plot(df_without_weight_transformed$model_year,res, ylab="Residuals", xlab="model_year",  main="residuals vs model_year") 
abline(0, 0) 


#QQ plot to check if residuals follow normal distribution.
qqnorm(res)
qqline(res)

#To check if residuals are normally distributed
plot(density(res),main = "Distribution of Residuals", ylab= "density")


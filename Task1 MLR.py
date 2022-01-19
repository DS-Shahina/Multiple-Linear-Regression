# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Multiple Linear Regression/50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Convert Categorical Variable into Dummy variables
df = pd.get_dummies(df, columns = ["State"], drop_first = True)

df.columns = "RandDSpend", "Administration","MarketingSpend","Profit","State_Florida","State_NewYork" #renaming so that no sapces is there otherwise error.

df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# R&DSpend
plt.bar(height = df["RandDSpend"], x = np.arange(1, 51, 1))
plt.hist(df["RandDSpend"]) #histogram
plt.boxplot(df["RandDSpend"]) #boxplot

# Administration
plt.bar(height = df["Administration"], x = np.arange(1, 51, 1))
plt.hist(df["Administration"]) #histogram
plt.boxplot(df["Administration"]) #boxplot

# MarketingSpend
plt.bar(height = df["MarketingSpend"], x = np.arange(1, 51, 1))
plt.hist(df["MarketingSpend"]) #histogram
plt.boxplot(df["MarketingSpend"]) #boxplot

# Profit
plt.bar(height = df["Profit"], x = np.arange(1, 51, 1))
plt.hist(df["Profit"]) #histogram
plt.boxplot(df["Profit"]) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=df['RandDSpend'], y=df['Profit']) #both univariate and bivariate visualization.
sns.jointplot(x=df['Administration'], y=df['Profit']) 
sns.jointplot(x=df['MarketingSpend'], y=df['Profit']) 
sns.jointplot(x=df['Profit'], y=df['Profit']) 

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(df['RandDSpend']) #countplot() method is used to Show the counts of observations in each categorical bin using bars.
sns.countplot(df['Administration'])
sns.countplot(df['MarketingSpend'])
sns.countplot(df['Profit'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(df.Profit, dist = "norm", plot = pylab) # data is normally distributed
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
                             
# Correlation matrix 
a = df.corr() #R&DSpend-MarketingSpend(Collinearity problem) With respect to Profit all of them are positively correlated.

# we see there exists High collinearity between input variables especially between
# R&DSpend-MarketingSpend] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
#here ignoring the collinearity problem    
# We try to eliminate reasons of those varibales being insignificant.try to look into various scenario:
#1st scenario is, Is this because of the relation between y and x, we apply simple linear regression between, y and x1, y and x2.....so on.
#If it showing that there is no problem, we proceed further for influential observation.
ml1 = smf.ols('Profit ~ RandDSpend + Administration + MarketingSpend + State_Florida + State_NewYork', data = df).fit() # regression model

# Summary
ml1.summary()
# p-values for Administration, MarketingSpend, State_Florida, State_NewYork  are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) # It is comming up with showing you the observation which is deviating from the rest of the observations w.r.t. the residuals(error).
# the residuals we are trying to capture, we are trying to see what is that record which has the data skewed from the rest of the observations.
# Studentized Residuals = Residual/standard deviation of residuals
# index 45,46,48 is showing high influence so we can exclude that entire row
df_new= df.drop(df.index[[45,46,48]],axis=0) # it is dropping the record 45,46,48

#again we build model
# Preparing model                  
ml_new = smf.ols('Profit ~ RandDSpend + Administration + MarketingSpend + State_Florida + State_NewYork', data = df_new).fit()    

# Summary
ml_new.summary() 

# This model gives best R-squared:0.950  = it is greater than 0.8 => strong correlation => goodness of fit
# Adj. R-squared:0.944
# Equation is, y = 5.541e+04+0.8111(RandDSpend)-0.0629(Administration)+0.0199(MarketingSpend)+608.6022(State_Florida)+1151.1433(State_NewYork) -> B0+B1x+B2x+B3x+B4X+B5X

# Prediction
pred = ml_new.predict(df)

# Q-Q plot
res = ml_new.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = df["Profit"], lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml_new)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ RandDSpend + Administration + MarketingSpend + State_Florida + State_NewYork", data = df_train).fit()

# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse #14102.686337895531


# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #7216.059795616378


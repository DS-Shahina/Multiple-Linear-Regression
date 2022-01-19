# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
toyota1 = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Multiple Linear Regression/ToyotaCorolla.csv", encoding= 'unicode_escape')
toyota1.columns
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

toyota = toyota1[['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]

toyota.describe()
toyota.columns

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns

#Histogram
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.histplot(data=toyota, x=predictor)
    
#boxplot    
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.boxplot(data=toyota, x=predictor) 

#barplot    
plt.bar(height = toyota["Price"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Age_08_04"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["KM"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["HP"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["cc"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Doors"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Gears"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Quarterly_Tax"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Weight"], x = np.arange(1, 1437, 1))


# Jointplot
import seaborn as sns
sns.jointplot(x=toyota['Age_08_04'], y=toyota['Price']) #both univariate and bivariate visualization.
sns.jointplot(x=toyota['KM'], y=toyota['Price'])
sns.jointplot(x=toyota['HP'], y=toyota['Price'])
sns.jointplot(x=toyota['cc'], y=toyota['Price'])
sns.jointplot(x=toyota['Doors'], y=toyota['Price'])
sns.jointplot(x=toyota['Gears'], y=toyota['Price'])
sns.jointplot(x=toyota['Quarterly_Tax'], y=toyota['Price'])
sns.jointplot(x=toyota['Weight'], y=toyota['Price'])

# Countplot
plt.figure(1, figsize=(16, 10))
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.countplot(data=toyota, x=predictor) #countplot() method is used to Show the counts of observations in each categorical bin using bars.

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(toyota.Price, dist = "norm", plot = pylab) # data is NOT normally distributed
plt.show()

stats.probplot(np.log(toyota['Price']),dist="norm",plot=pylab) #best transformation, Now data is normally distributed.

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(toyota.iloc[:, :])
                             
# Correlation matrix 
a = toyota.corr() 
# we see there exists High collinearity between input variables 
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
#here ignoring the collinearity problem    
# We try to eliminate reasons of those varibales being insignificant.try to look into various scenario:
#1st scenario is, Is this because of the relation between y and x, we apply simple linear regression between, y and x1, y and x2.....so on.
#If it showing that there is no problem, we proceed further for influential observation.
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit() # regression model

# Summary
ml1.summary()
# p-values for cc, Doors are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) # It is comming up with showing you the observation which is deviating from the rest of the observations w.r.t. the residuals(error).
# the residuals we are trying to capture, we are trying to see what is that record which has the data skewed from the rest of the observations.
# Studentized Residuals = Residual/standard deviation of residuals
# index 80,221,960 is showing high influence so we can exclude that entire row

toyota_new = toyota.drop(toyota.index[[80,221,960]]) # it is dropping the record 80,221,960.

#again we build model
# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota_new).fit()    

# Summary
ml_new.summary() # p-value is less than 0.05.
#coefficients are statistically significant

# Before dropping the record: R-squared:0.864, After dropping the record:R-squared:0.885
#Before dropping the record: Adj. R-squared:0.863,After dropping the record:Adj. R-squared:0.885
#R-squared and Adj. R-squared: Increased After removing records

"ml_new =  Final Model"

# Prediction - ml_new
pred = ml_new.predict(toyota)

# Q-Q plot
res = ml_new.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = toyota.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml_new)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
toyota_train, toyota_test = train_test_split(toyota, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = toyota_train).fit()

# prediction on test data set 
test_pred = model_train.predict(toyota_test)

# test residual values 
test_resid = test_pred - toyota_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse #1328.4383667274078


# train_data prediction
train_pred = model_train.predict(toyota_train)

# train residual values 
train_resid  = train_pred - toyota_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #1342.9698259162574

# Training Error and Test Error is somewhat equal then we can say it is right fit.
# So this model can be accepted

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cars = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Multiple Linear Regression/Cars.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cars.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# HP
plt.bar(height = cars.HP, x = np.arange(1, 82, 1))
plt.hist(cars.HP) #histogram
plt.boxplot(cars.HP) #boxplot

# MPG
plt.bar(height = cars.MPG, x = np.arange(1, 82, 1))
plt.hist(cars.MPG) #histogram
plt.boxplot(cars.MPG) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['HP'], y=cars['MPG']) #both univariate and bivariate visualization.

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['HP']) #countplot() method is used to Show the counts of observations in each categorical bin using bars.

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.MPG, dist = "norm", plot = pylab) # data is normally distributed
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr() #HP-SP(Collinearity problem), VOL-WT(Collinearity problem), With respect to MPG all of them are negatively correlated.

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
#here ignoring the collinearity problem    
# We try to eliminate reasons of those varibales being insignificant.try to look into various scenario:
#1st scenario is, Is this because of the relation between y and x, we apply simple linear regression between, y and x1, y and x2.....so on.
#If it showing that there is no problem, we proceed further for influential observation.
ml1 = smf.ols('MPG ~ WT + VOL + SP + HP', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05 these two variables are statistically insignificant.

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) # It is comming up with showing you the observation which is deviating from the rest of the observations w.r.t. the residuals(error).
# the residuals we are trying to capture, we are trying to see what is that record which has the data skewed from the rest of the observations.
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[76]]) # it is dropping the record 76.

#again we build model
# Preparing model                  
ml_new = smf.ols('MPG ~ WT + VOL + HP + SP', data = cars_new).fit()    

# Summary
ml_new.summary() # p-value is not less than 0.05, still it is not statistically significant, problem is not resolved.

# Now, we proceed further and try to remove the 1 of the coliner observation.
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP ~ WT + VOL + SP', data = cars).fit().rsquared  # take x1 and see relation with all the variables
vif_hp = 1/(1 - rsq_hp) 
#R^2 = 0.9498157963084058 (Very very strongly correlated)
#vif = 19.92658897499852
rsq_wt = smf.ols('WT ~ HP + VOL + SP', data = cars).fit().rsquared  # take x2 and see relation with all the variables
vif_wt = 1/(1 - rsq_wt)
#R^2 = 0.9984363610296332 (Very very strongly correlated)
#VIF = 639.5338175572624
rsq_vol = smf.ols('VOL ~ WT + SP + HP', data = cars).fit().rsquared  # take x3 and see relation with all the variables
vif_vol = 1/(1 - rsq_vol) 
#r^2 = 0.9984345797174133 (Very very strongly correlated)
#VIF = 638.8060836592878
rsq_sp = smf.ols('SP ~ WT + VOL + HP', data = cars).fit().rsquared  # take x4 and see relation with all the variables
vif_sp = 1/(1 - rsq_sp) 
#R^2 = 0.9500190896665341 (Very very strongly correlated)
#VIF = 20.00763878305008
# Storing vif values in a data frame
d1 = {'Variables':['HP', 'WT', 'VOL', 'SP'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('MPG ~ VOL + SP + HP', data = cars).fit()
final_ml.summary()# Now, coefficients are statistically significant,problem is resolved.

# with WT: R-squared:0.771, Without WT:R-squared:0.770
#with WT: Adj. R-squared:0.758, Without WT:Adj. R-squared:0.761
#R-squared Decreased by 0.1%, Adj. R-squared: Increased by 0.1%

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.MPG, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("MPG ~ HP + SP + VOL", data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.MPG
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.MPG
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

# Training Error and Test Error is somewhat equal then we can say it is right fit.
#So this model can be accepted


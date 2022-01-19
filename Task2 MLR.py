# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
toyota = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Multiple Linear Regression/Computer_Data.csv")
toyota.columns
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

#delete unwanted columns
del toyota['Unnamed: 0']

# Convert Categorical Variable into Dummy variables
toyota = pd.get_dummies(toyota, columns = ["cd","multi","premium"], drop_first = True)

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
plt.bar(height = toyota["price"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["speed"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["hd"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["ram"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["screen"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["ads"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["trend"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["cd_yes"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["multi_yes"], x = np.arange(1, 6260, 1))
plt.bar(height = toyota["premium_yes"], x = np.arange(1, 6260, 1))

# Jointplot
import seaborn as sns
sns.jointplot(x=toyota['speed'], y=toyota['price']) #both univariate and bivariate visualization.
sns.jointplot(x=toyota['hd'], y=toyota['price'])
sns.jointplot(x=toyota['ram'], y=toyota['price'])
sns.jointplot(x=toyota['screen'], y=toyota['price'])
sns.jointplot(x=toyota['ads'], y=toyota['price'])
sns.jointplot(x=toyota['trend'], y=toyota['price'])
sns.jointplot(x=toyota['cd_yes'], y=toyota['price'])
sns.jointplot(x=toyota['multi_yes'], y=toyota['price'])
sns.jointplot(x=toyota['premium_yes'], y=toyota['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.countplot(data=toyota, x=predictor) #countplot() method is used to Show the counts of observations in each categorical bin using bars.

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(toyota.price, dist = "norm", plot = pylab) # data is normally distributed
plt.show()

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
ml1 = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit() # regression model

# Summary
ml1.summary()
# all p-values are less than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) # It is comming up with showing you the observation which is deviating from the rest of the observations w.r.t. the residuals(error).
# the residuals we are trying to capture, we are trying to see what is that record which has the data skewed from the rest of the observations.
# Studentized Residuals = Residual/standard deviation of residuals
# index 3783,4477,5960 is showing high influence so we can exclude that entire row

toyota_new = toyota.drop(toyota.index[[3783,4477,5960]]) # it is dropping the record 3783,4477,5960.

#again we build model
# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota_new).fit()    

# Summary
ml_new.summary() # p-value is less than 0.05.
#coefficients are statistically significant

# Now, we proceed further and try to remove the 1 of the coliner observation.
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
"price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes"
rsq_speed = smf.ols('speed ~ hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x1 and see relation with all the variables
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x2 and see relation with all the variables
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ speed + hd + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x3 and see relation with all the variables
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ ram + speed + hd + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_screen = 1/(1 - rsq_screen) 

rsq_ads = smf.ols('ads ~ screen + ram + speed + hd + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ ads + screen + ram + speed + hd + cd_yes + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_trend = 1/(1 - rsq_trend) 

rsq_cd_yes = smf.ols('cd_yes ~ ads + screen + ram + speed + hd + trend + multi_yes + premium_yes', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_cd_yes = 1/(1 - rsq_cd_yes) 

rsq_multi_yes = smf.ols('multi_yes ~ cd_yes + ads + screen + ram + speed + hd + trend + premium_yes', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_multi_yes = 1/(1 - rsq_multi_yes) 

rsq_premium_yes = smf.ols('premium_yes ~ multi_yes + cd_yes + ads + screen + ram + speed + hd + trend', data = toyota).fit().rsquared  # take x4 and see relation with all the variables
vif_premium_yes = 1/(1 - rsq_premium_yes) 

# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen', 'ads', 'trend', 'cd_yes','multi_yes', 'premium_yes'], 'VIF':[vif_speed, vif_hd, vif_ram, vif_screen,vif_ads,vif_trend,vif_cd_yes,vif_multi_yes,vif_premium_yes]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = toyota).fit()
final_ml.summary()# coefficients are statistically significant,but R^2 And Adjusted r^2 is reduced so, ml_new is the good model.

# with hd: R-squared:0.775, Without hd:R-squared:0.747
#with hd: Adj. R-squared:0.775, Without hd:Adj. R-squared:0.746
#R-squared and Adj. R-squared: Decreased

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
sns.residplot(x = pred, y = toyota.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml_new)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
toyota_train, toyota_test = train_test_split(toyota, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes", data = toyota_train).fit()

# prediction on test data set 
test_pred = model_train.predict(toyota_test)

# test residual values 
test_resid = test_pred - toyota_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse #278.52949548805515


# train_data prediction
train_pred = model_train.predict(toyota_train)

# train residual values 
train_resid  = train_pred - toyota_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #274.3714301901709

# Training Error and Test Error is somewhat equal then we can say it is right fit.
#So this model can be accepted

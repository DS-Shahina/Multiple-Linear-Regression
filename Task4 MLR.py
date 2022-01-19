# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
ap = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Multiple Linear Regression/Avacado_Price.csv")
ap.columns
ap.columns = 'AveragePrice', 'Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3','Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type', 'year','region' #renaming so that no sapces is there otherwise error.

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

import seaborn as sns
import matplotlib.pyplot as plt

byRegion=ap.groupby('region').mean()
byRegion.sort_values(by=['AveragePrice'], ascending=False, inplace=True)
plt.figure(figsize=(17,8),dpi=100)
sns.barplot(x = byRegion.index,y=byRegion["AveragePrice"],data = byRegion,palette='rocket')
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price According to Region')


#The barplot shows the average price of avocado at various regions in a ascending order. 
#Clearly Hartford Springfield, SanFrancisco, NewYork are the regions with the highest avocado prices.

# Converting Categorical Variable into numeric.
from sklearn import preprocessing 
 
label_encoder = preprocessing.LabelEncoder() 
ap['type']= label_encoder.fit_transform(ap['type']) 
ap

ap.shape

ap.type.value_counts()

#Featuring Engineering- Handle Categorical Features Many Categories(Count/Frequency Encoding)
len(ap["region"].unique())

# Let's obtain the counts for each one of the labels in variable "region"
# Let's capture this in a dictionary that we can use to re-map the labels

ap.region.value_counts().to_dict()

# And now let's replace each label in "region" by its count
# first we make a dictionary that maps each label to the counts

ap_frequency_map = ap.region.value_counts().to_dict()

# and now we replace "region" lables in the dataset ap
ap.region = ap.region.map(ap_frequency_map)

ap.head()

ap.describe()
ap.columns

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns

#Histogram
for i, predictor in enumerate(ap):
    plt.figure(i)
    sns.histplot(data=ap, x=predictor)
    
#boxplot    
for i, predictor in enumerate(ap):
    plt.figure(i)
    sns.boxplot(data=ap, x=predictor) 

#barplot    
plt.bar(height = ap["AveragePrice"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["Total_Volume"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["tot_ava1"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["tot_ava2"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["tot_ava3"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["Total_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["Small_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["Large_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["XLarge Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["type"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["year"], x = np.arange(1, 18250, 1))
plt.bar(height = ap["region"], x = np.arange(1, 18250, 1))


# Jointplot
import seaborn as sns
sns.jointplot(x=ap['Total_Volume'], y=ap['AveragePrice']) #both univariate and bivariate visualization.
sns.jointplot(x=ap['tot_ava1'], y=ap['AveragePrice'])
sns.jointplot(x=ap['tot_ava2'], y=ap['AveragePrice'])
sns.jointplot(x=ap['tot_ava3'], y=ap['AveragePrice'])
sns.jointplot(x=ap['Total_Bags'], y=ap['AveragePrice'])
sns.jointplot(x=ap['Small_Bags'], y=ap['AveragePrice'])
sns.jointplot(x=ap['Large_Bags'], y=ap['AveragePrice'])
sns.jointplot(x=ap['XLarge Bags'], y=ap['AveragePrice'])
sns.jointplot(x=ap['type'], y=ap['AveragePrice'])
sns.jointplot(x=ap['year'], y=ap['AveragePrice'])
sns.jointplot(x=ap['region'], y=ap['AveragePrice'])

# Countplot
plt.figure(1, figsize=(16, 10))
for i, predictor in enumerate(ap):
    plt.figure(i)
    sns.countplot(data=ap, x=predictor) #countplot() method is used to Show the counts of observations in each categorical bin using bars.

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(ap.AveragePrice, dist = "norm", plot = pylab) # data is NOT normally distributed
plt.show()

stats.probplot(np.log(ap['AveragePrice']),dist="norm",plot=pylab) #best transformation, Now data is normally distributed.

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(ap.iloc[0:1000,:])
                             
# Correlation matrix 
a = ap.corr() 
# we see there exists High collinearity between input variables 
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
#here ignoring the collinearity problem    
# We try to eliminate reasons of those varibales being insignificant.try to look into various scenario:
#1st scenario is, Is this because of the relation between y and x, we apply simple linear regression between, y and x1, y and x2.....so on.
#If it showing that there is no problem, we proceed further for influential observation.
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type + year + region', data = ap).fit() # regression model

# Summary
ml1.summary()
# P-values of all variables are more than 0.05 except type
# R-squared:                       0.408
# Adj. R-squared:                  0.408
# Here we can clearly see that r-squared is also very low
# So this model is not accppetable

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) ##System not supporting this much calculations, getting crashed in between

#Applying transformations => CubeRoot(x)

model2=smf.ols('AveragePrice~np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data=ap).fit()
model2.summary()

#now we can see that
# R-squared:                       0.502
# Adj. R-squared:                  0.502
# R-squared has improved
# p-value of all the variables are now below than 0.05 

# So checking for collinearity to remove variable using VIF

rsq_region = smf.ols('np.cbrt(region) ~ np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)', data = ap).fit().rsquared  
vif_region = 1/(1 - rsq_region)  
vif_region #  1.0220383403449582

rsq_year = smf.ols('np.cbrt(year)~np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)', data = ap).fit().rsquared  
vif_year = 1/(1 - rsq_year)  
vif_year # 1.3220141296382264

rsq_type = smf.ols('np.cbrt(type)~np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)', data = ap).fit().rsquared  
vif_type = 1/(1 - rsq_type)  
vif_type # 1.8060113621900487

rsq_XLarge_Bags = smf.ols('np.cbrt(XLarge_Bags)~np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)', data = ap).fit().rsquared  
vif_XLarge_Bags = 1/(1 - rsq_XLarge_Bags)  
vif_XLarge_Bags # 2.674814560074918

rsq_Large_Bags = smf.ols('np.cbrt(Large_Bags)~np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)', data = ap).fit().rsquared  
vif_Large_Bags = 1/(1 - rsq_Large_Bags)  
vif_Large_Bags # 12.815223224015886

rsq_Small_Bags = smf.ols('np.cbrt(Small_Bags)~np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)', data = ap).fit().rsquared  
vif_Small_Bags = 1/(1 - rsq_Small_Bags)  
vif_Small_Bags # 97.05464049893034

rsq_Total_Bags = smf.ols('np.cbrt(Total_Bags)~np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)', data = ap).fit().rsquared  
vif_Total_Bags = 1/(1 - rsq_Total_Bags)  
vif_Total_Bags # 183.66105535482146

rsq_tot_ava3 = smf.ols('np.cbrt(tot_ava3)~np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)', data = ap).fit().rsquared  
vif_tot_ava3 = 1/(1 - rsq_tot_ava3)  
vif_tot_ava3 # 4.610304163574555

rsq_tot_ava2 = smf.ols('np.cbrt(tot_ava2)~np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)', data = ap).fit().rsquared  
vif_tot_ava2 = 1/(1 - rsq_tot_ava2)  
vif_tot_ava2 # 58.198269554686426

rsq_tot_ava1= smf.ols('np.cbrt(tot_ava1)~np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)', data = ap).fit().rsquared  
vif_tot_ava1 = 1/(1 - rsq_tot_ava1)  
vif_tot_ava1 # 35.159023904070295

rsq_Total_Volume = smf.ols('np.cbrt(Total_Volume)~np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data = ap).fit().rsquared  
vif_Total_Volume = 1/(1 - rsq_Total_Volume)  
vif_Total_Volume # 306.81074681924065

# Here we can clearly see that VIF of Total_Volume is very high
# So we will remove this variable from our calculations

ap.columns
# Storing vif values in a data frame
d1 = {'Variables':['Total_Volume','tot_ava1','tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags',' XLarge_Bags',' type','year','region'],
      'VIF':[vif_Total_Volume, vif_tot_ava1, vif_tot_ava2, vif_tot_ava3, vif_Total_Bags, vif_Small_Bags, vif_Large_Bags, vif_XLarge_Bags, vif_type,vif_year,vif_region]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# So following variables have very high vif Total_Volume, Total_Bags
# Removing these variables from our calculation

ap1=ap.drop(['Total_Volume','Total_Bags'], axis=1)
ap1.columns

#creating the model again
model3=smf.ols('AveragePrice~np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data=ap1).fit()
model3.summary()

# coefficients are statistically significant,but R^2 And Adjusted r^2 is reduced so, model2 is the good model.

# with Total_Volume, Total_Bags: R-squared:0.502, Without Total_Volume, Total_Bags:R-squared:0.458
#with Total_Volume, Total_Bags: Adj. R-squared:0.502, Without Total_Volume, Total_Bags:Adj. R-squared:0.458
#R-squared and Adj. R-squared: Decreased

"model2 =  Final Model"

# Prediction - ml_new
pred = model2.predict(ap)

# Q-Q plot
res = model2.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = ap.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(model2) #System not supporting this much calculations, getting crashed in between

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
ap_train, ap_test = train_test_split(ap, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice~np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)", data = ap_train).fit()

# prediction on test data set 
test_pred = model_train.predict(ap_test)

# test residual values 
test_resid = test_pred - ap_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse #0.28373173966683357


# train_data prediction
train_pred = model_train.predict(ap_train)

# train residual values 
train_resid  = train_pred - ap_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #0.2849206746453047

# Training Error and Test Error is approximately equal then we can say it is right fit.
#So this model can be accepted


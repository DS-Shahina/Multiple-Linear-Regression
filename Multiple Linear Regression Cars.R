# Load the Cars dataset
Cars <- read.csv(file.choose())
View(Cars)
 
attach(Cars)
 
# Normal distribution
qqnorm(HP)
qqline(HP)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Cars)

# Scatter plot
plot(HP, MPG) # Plot relation ships between each X with Y
plot(VOL, MPG)

# Or make a combined plot
pairs(Cars)   # Scatter plot for all pairs of variables
plot(Cars)

cor(HP, MPG)
cor(Cars) # correlation matrix

# The Linear Model of interest
model.car <- lm(MPG ~ VOL + HP + SP + WT, data = Cars) # lm(Y ~ X)
summary(model.car)

model.carV <- lm(MPG ~ VOL)
summary(model.carV)

model.carW <- lm(MPG ~ WT)
summary(model.carW)

model.carVW <- lm(MPG ~ VOL + WT)
summary(model.carVW)

#### Scatter plot matrix with Correlations inserted in graph
# install.packages("GGally")
library(GGally)
ggpairs(Cars)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(Cars)

cor2pcor(cor(Cars))

# Diagnostic Plots
install.packages(car)
library(car)

plot(model.car)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.car, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.car, id.n = 3) # Index Plots of the influence measures
influencePlot(model.car, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.car1 <- lm(MPG ~ VOL + HP + SP + WT, data = Cars[-77, ])
summary(model.car1)


### Variance Inflation Factors
vif(model.car)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
VIFWT <- lm(WT ~ VOL + HP + SP)
VIFVOL <- lm(VOL ~ WT + HP + SP)
VIFHP <- lm(HP ~ VOL + WT + SP)
VIFSP <- lm(SP ~ VOL + HP + WT)

summary(VIFWT)
summary(VIFVOL)
summary(VIFHP)
summary(VIFSP)

# VIF of SP
1/(1-0.95)

#### Added Variable Plots ######
avPlots(model.car, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without WT
model.final <- lm(MPG ~ VOL + HP + SP, data = Cars)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <- lm(MPG ~ VOL + HP + SP, data = Cars[-77, ])
summary(model.final1)

# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

# Subset selection
# 1. Best Subset Selection
# 2. Forward Stepwise Selection
# 3. Backward Stepwise Selection / Backward Elimination

install.packages("leaps")
library(leaps)
lm_best <- regsubsets(MPG ~ ., data = Cars, nvmax = 15)
summary(lm_best)
?regsubsets
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

lm_forward <- regsubsets(MPG ~ ., data = Cars, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(Cars)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- Cars[-train, ]

# Model Training
model <- lm(MPG ~ VOL + HP + SP, Cars[train, ])
summary(model)


pred <- predict(model, newdata = test)
actual <- test$MPG
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.car)


## House Prices Analysis Report

### Strayn Wang

This report has 3 sections: Understanding the Problem, Data Exploration, Machine Learning.


### Understanding the Problem
Essentially, the price changes can be divided into two parts: changes from the vehicle of the housing market and the quality of the house.

The market-level analysis is hard to be done since the window of House Prices Data is only 2 years. You can skip to the next paragraph if you are more interested in this specific data problem. Here I list the potential market-level analysis:  
1) Substitutes Analysis, e.g. trends of House Renting market can influence the House Buying market prices. 
2) Rivalry Analysis, e.g. competitors' new products can change the rivalry dynamic and influence the prices on the market.
3) New Entrants Analysis, e.g. newcomers can bring new technology that changes the game or consume more raw materials that influence costs and prices. 
4) Bargaining Power of Buyers Analysis, e.g. demographical changes over the area (growth of population, etc.) may change supplier/buyer ratio; also, customer's personal background, for example, her banking credit histories sometimes can influence the prices she gets. 

We can do a product-level analysis for this single data set, which focuses on the Bargaining Power of Suppliers. 
The deal-breaking factors for house buyers are:
1. Property: Amenity, Age, structure, area size, decoration, luxury.
2. Policy: tax rate, interest rate, economic conditions.
3. Location: Neighborhood Safety, Transportation, Commercials (groceries, markets, restaurants), Schools (benefit for new parents), Hospitals (benefit for the old).
(More details can be found in this [quora answer](https://www.quora.com/What-factors-affect-real-estate-prices).)  

Our dataset has rich information for "property", such as:
    overall: 'grade'
    structures: 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view'
    area size: 'sqft_living','sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 
    age: 'condition','date', 'yr_built', 'yr_renovated'

Policies seem a hard one, but actually, they are the same in one state, therefore, state name can carry information about it. However, since sales records in our data are only in Washington, this can be ignored.

Location carries rich deal-breaking information and we get the latitude and longitude, it is promising to spend sometime recover it.   
I use Google Reverse Geocoding API to recover **Street, Neighborhood, County, State, Country** information from longitude and altitude.  
I then use Google Place API to search & count **Transportation, Supermarkets, Restaurants** in 10 minutes walking distance and *Schools and Hospitals** in 30 minutes walking distance.  

The imputed data can be found in ```./tmp/data_imputed.csv```.   
The API query code can be found in ```./lib/geo_recovering.py```.

### Data Exploration
#### 1. How prices flow with time (Time Series Analysis)
Whether it is reasonable to predict future by model depends on whether the price pattern is constant over time. I plot the daily mean price for each city (locality in the plot):  
![](./visualize/time_series_price_by_locality.png) 

Not too many peaks can be found in 8/32 (Seattle, Carnation, Lake Forest Park, Burien, Issaquah, Normandy Park, Bellevue, Mercer Island), we can say the price is stable in the 2-year window. It means we can create a model ignoring absolute time,  which means we can transform columns like yr_built/yr_renovated to relative time, e.g. age/age_renovated or simply delete them.

#### 2. How prices stretch by values (Distribution Analysis)
I print the mean price for different cities in Washington. 
![](./visualize/reginoal_price.png)
Seattle, the only city I ever heard of in Washington, whose house average prices rank 16. And [Medina](https://en.wikipedia.org/wiki/Medina,_Washington), the most expensive city in Washington has itself surrounding by the pretty Lake Washington.  

More details can be found in the following density plot:  
![](./visualize/density_price_by_locality.png)

The number on the plot indicate the count of data in that region. Most of the sales go to Seattle. Secondly, a majority of the regional prices have long tails. Normally, if the long tails take up big proportion in a variable, we may benefit from logarithm transformation of the variable since it rebalances the data by reducing the skewness.

#### 3. How price correlates to indicators (Correlation Analysis)  
To show this I made the scatter plots between each indicator to the logarithm of price:    
![](./visualize/scatter_all_by_log_price.png)

Out of all columns, the grade, size of the living room (sqft_living) has the strongest correlation on the plot (this is further shown in importance analysis).

Code of visualizing can be found in ```./visualize/visualize.py```. 

### Machine Learning

#### 1. Data preparation
I take [Issaquah](https://en.wikipedia.org/wiki/Issaquah,_Washington), a city with 761 records whose average price is 644439.05 to be the test set.

I split the rest data into 8/2, where the training set has 16663 records and validation set has 4166 records.

Code of preparing training/validation/testing dataset is in ```./lib/split_train_val_test.py```.

#### 2. Feature Engineering
I extract the following columns out of the original data:
    age: age of the house, age = date - yr_built
    age_renovated: time since last renovate, age_renovated = date - yr_renovated
    sqft = sqft_above+sqft_basement+sqft_living+sqft_lot
    count_markets: the count of supermarkets in a radius of 800. 
    count_restaurants: the count of restaurants in a radius of 800. 
    count_stations: the count of transition stations in a radius of 800. 
    count_schools: the count of schools in a radius of 2400. 
    count_clinics: the count of clinics in a radius of 2400. 

I compare the log and original form of a selected list of variables:  
![](/home/nyartsgnaw/pyproject/pingan-takehome/visualize/density_compare_log.png)

Now we have recovered/transformed enough features, now we need to reduce redundancies/noises. To achieve these, I will train a model and use its validation performance as criteria to identify key features. 

Code of preparing data can be found in ```./lib/prepare_data.py```.

#### 3. Model Setup
When a human customer goes to buy a house it can be thought as a combination of decisions over different perspectives, a decision tree is born to mimic human decision-making process, therefore, I choose it as the root of my modeling. Since the perspectives to be considered are many, I use a bootstrap-upgraded decision called Random Forest to ensure robustness.

The best part for using Random Forest is that it doesn't have any assumptions like those of Linear Regressions: Multivariate normality, Linear relationship, No multicollinearity, No auto-correlation etc. Although this should be checked to help approach better performance or robustness, but these are not required.  

A bonus for Random Forest is that it still have the node importance feature, which helps us to find the most significant features.


#### 4. Loss function and metrics
I choose Mean Squared Error as loss function for machine learning since it is faster than Mean Absolute Error. Doing so causes a potential problem of bias over the bigger value outlier, two solutions for this:
1) Delete outliers with a threshold.   
2) Transformed the skewed data with logarithm.  

On evaluating results, I didn't use MSE directly but use proportional MAE, which is Mean Absolute Error divided by mean of true values. 

#### 5. Experiments
#### 5.1. Do added features make sense?
I show the effects of added features in the following plot:    
![](./experiments/comparison_added_features.png)

Comparing to the baseline (NON_KEPT), all features improve the predictions. Among them, COUNT_KEPT and LOCALITY_KEPT significantly improve the prediction. 

Though not shown in the plot, testing Proportional MAE indicates uses of locality and neighborhood features lead to overfitting. I merged small categories under locality/neighborhood to a category called "Others" (which is also the category for unobserved locality/neighborhood in training dataset). The following plot indicates that the quantile of 30% is a good threshold for deciding "small categories".  
![](./experiments/scatter_threshould_categorical.png)

Another way to deal with this Dimension Reduction by LASSO/AIC, however, given limited time I only tried AIC, but the first try wasn't good enough for selecting locality.  

#### 5.2. Does logarithm transformation help to rebalance skewness?
As stated in section 4, we use logarithm transformation to deal with skewed data issue. We first try to implement it on price:   
![](./experiments/log_price_vs_price.png)

Results show that it harms the validating Proportional MAE, though a further check shows that it helps increase robustness (differences between validating Proportional MAE and testing Proportional MAE). I haven't got enough time to analyze the incorrect predictions are for the smaller or bigger values, but Box-Cox transformation may help improve this.

#### 5.3. Does Threshold help to cutoff outliers?
To cutoff outliers bigger than a threshold of 90% quantile help to decrease validating MSE:  
![](./experiments/scatter_threshould_numeric.png)

Given limited time, I only show the points, a better way of doing this is clustering/Gaussian-based cutoff.

#### 5.4. Current Results
Based on insights from 5.1~5.3, I zoom in on 127 different experiment factors. With the criteria of validating Proportional MAE and testing Proportional MAE, I found the best models and list them here:  
![](./experiments/best_models.png)

The best models have an average validating Proportional MAE of 0.1426 and a testing Proportional MAE of 0.1553. 
The baseline models have a validating Proportional MAE of 0.2310 and an average testing Proportional MAE of 0.1995. 
An average of 62% improvement has been seen on the dataset!

The grid search experiments can be found in ```./experiments/exp_logs.xlsx```.
The code of feature engineering can be found in ```./lib/feature_engineering.py```.

#### 6. Error Analysis 
Plot the error and check its distribution, better be normal distributed.


#### 7. Node Importance Analysis
Identify the most important nodes for the optimal model.  

### Conclusions
1. Log prices help increase robustness but hurt model fitting, it can be improved by implementing Box-Cox.  Logged count, logged age, and logged sqft didn't contribute significantly to the final improvement comparing to their original forms, but they should be kept since they help to moderate robustness issue caused by taking MSE loss. In other words, the log-transformed features may keep the model consistently good even when encountering outlier values in the future.

2. Count and locality provide the biggest improvement for prediction (minimizing validating Proportional MAE). But a change of testing dataset may show overfitting issue since locality and neighborhood cover only information for Washington. In the future, wider tests across states should be directed. 

3. Given more data and longer time, the market-level analysis is expected to improve the models.


## House Prices Analysis Report

### Strayn Wang

This report has 3 sections: Understanding the Problem, Data Exploration, Machine Learning.


### Understanding the Problem
Essentially, the price changes can be divided into changes from the vehicle of housing market and the quality of house products.
Since I only get 2 years of House Pricing Data, it is impossible to do market-level analysis, which could be 
1) Substitutes Analysis, trends of House Renting can influence the price of house prices. 
2) Rivalry Analysis, where competitors' new products can change the rivalry dynamic therefore influence the prices.
3) New Entrants Analysis, where new commers with better sense of big data or simply takes up the production materials that changes costs and influence house prices. 
4) Bargaining Power of Buyers Analysis, which could be the demographical changes over the area (growth of population, students number etc.), customer credit histories, etc. 

The only thing left to be consider is the product-level analysis, which influence the Bargaining Power of Suppliers. The major need of house buyer can be found in this [quora answer](https://www.quora.com/What-factors-affect-real-estate-prices).  
To summarize, the following 4 factors are the biggest one:
1. Property: Amenity, Age, structure, area size, decoration, luxry.
2. Policy: tax rate, interest rate, economic conditions.
3. Location: Neighborhood Safety, Transportation, Commercials (groceries, markets, restaurants), Schools (benefit for new parents), Hospitals (benefit for the old).

Our data include rich information for the 1st factor "property":
    overall: 'grade'
    structures: 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view'
    area size: 'sqft_living','sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 
    age: 'condition','date', 'yr_built', 'yr_renovated'

Since the 2nd factor policies are the same for places in one state, state name can carry information of it. Our data include sales in only Washington area, hence it could be ignored.
it is rewarding to recover information

Location carries lots of deal-breaking information, and since we get the latitude and longitude, it is rewarding to recover it.   
I use Google Reverse Geocoding API to recover the following information based on latitude and longitude: **Street, Neighborhood, County, State, Country**.  
I then use Google Place API to search & count **Transportation, Supermarkets, Restaurants** in 10 minutes walking distance; and I search & count **Schools and Hospitals** in 20 minutes driving distance.  

The imputed data can be found in ```./tmp/data_imputed.csv```.   
The API query code can be found in ```./lib/geo_recovering.py```.

### Data Exploration


#### 1. How prices flow with Time (Time Series Analysis)
To decide whether to train machine learning model to predict future, first thing to do is to check the whether price pattern is stable through time. I plot the daily mean price for each city (locality from google results) in the county (in which I merge group with small sizes to "Others" category):   
![](./visualize/time_series_price_by_locality.png) 

As you can see, significants peaks can be found in 8/32 (Seattle, Carnation, Lake Forest Park, Burien, Issaquah, Normandy Park, Believue, Mercer Island) of our cities, but even in these cities the peaks are not too many.
Therefore we can conclude the price can be considered stable during the 2 year window, based on this observation, we can delete/transform columns that state absolute dates.

#### 2. How prices stretch by values (Distribution Analysis)
I print the mean price for differnt cities in Washington. 
![](./visualize/reginoal_price.png)
Seattle, the only city I ever heard of in Washington, whose house average prices rank 16. And [Medina](https://en.wikipedia.org/wiki/Medina,_Washington), the most expensive city in Washington has itself surrounding by the pretty Lake Washington.   

More details can be found in the following density plot:  
![](./visualize/density_price_by_locality.png)

The number on the plot indicate the count of data in that region, this time, most of the sales go to Seattle. Besides, a majority of the regional prices have long tails, normally, if the long tails take up big proportion in a variable, we may benefit from logarithm transformation of the variable since it rebalance the data by reducing the skewness.

#### 3. What are the correlations between price and its indicators (Correlation Analysis)  
To predict price given all other columns, we need to know how their relationship looks like. Here is the scatter plots of each column to the logorithm of price:    
![](./visualize/scatter_all_by_log_price.png)

Out of all columns, the overall grade, size of the living room (sqft_living)


### Machine Learning

#### 1. Data prepartion
I take [Issaquah](https://en.wikipedia.org/wiki/Issaquah,_Washington), a city with 761 records whose average price is 644439.05 to be the test set.

I split the rest data into 8/2, where training set has 16663 records and validation set has 4166 records.

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

    log of a slected list of variables, which I plots it below, they are fed together with their original form for dimension reduction.    
![](/home/nyartsgnaw/pyproject/pingan-takehome/visualize/density_compare_log.png)

#### 3. Model Setup
When a human customer go to buy a house it can be thought as a combination of decisions over different perspectives, a decision tree is born to mimic human decision making process therefore I choose it as model base. Since there are quite many different perspectives to be considered, to improve robustness, I use Random Forest, a upgrade of decision tree with advatange of boostrap, as my model.  

#### 4. Loss function
I choose Mean Squared Error as loss function since it is significantly faster than Mean Absolute Error. Doing so causes a potential problem of bias over the bigger value outlier, to avoid I do two things:
1) Delete outliers with a threshould.  
2) Balance the skewed data with logarithm transformation.

#### 5. Experiments

#### 5.1. Do added features make sense?
I show effects of added features in the following plot:  
![](./experiments/comparison_added_features.png)

Comparing to the baseline (NON_KEPT), all features improve the predictions.
Among them, COUNT_KEPT and LOCALITY_KEPT significantly improve the prediction. 

However, performances over test dataset indicate overfitting is caused by using locality and neighborhood information. I merged small categories under locality/neighborhood to a category called "Others", which is also the category for unobserved locality/neigborhood in training dataset. The following plot indicate that the quantile of 30% is good threshould for defining "small categories".  
![](./experiments/scatter_threshould_categorical.png)

#### 5.2. Does logarithm transformation help to reblance skewness?
As stated in section 4, we use logarithm transformation to deal with skewed data issue. We first try implement it on price:  
![](./experiments/log_price_vs_price.png)

Results show that it will decrease the validation loss significantly. A further analysis has shown it help to increase robustness (differences between validating MSE and testing MSE), I haven't got enough time to analyze whether the smaller or bigger values were predicted incorrectly, but Box-Cox transformation may help improve this.

#### 5.3. Does Threshould help to cutoff outliers?
To cutoff outliers bigger than a threshould of 90 percentage help to decrease validating MSE:  
![](./experiments/scatter_threshould_numeric.png)

A better way of doing this may be clustering but given limited time that shows the point.  


### Conclusions
1. Log prices helps on robustness, but hurt more on model fitting, it should be improved by implementing Box-Cox, otherwise it should not be used.  
Logged count, logged age and logged sqft don't change much, but they should be kept, since they help to moderate robustness issue caused by using MSE loss, their contributions may be more significant for extended dataset.

2. Count and locality provide biggest improvement for prediction (minimizing validation MSE), on the other hand, looking at testing MSE, any use of locality and neighborhood will cause different level of overfitting.


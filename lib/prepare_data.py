#package loading
import pandas as pd
import numpy as np

import datetime
import re
import os 
pd.options.mode.chained_assignment = None  # default='warn'



def add_price_range(df):
    l_80 = np.percentile(df['price'],80)
    l_50 = np.percentile(df['price'],50)
    l_30 = np.percentile(df['price'],30)
    i = 0
    ranges = []
    while i < df.shape[0]:
        row = df.iloc[i]
        price = row['price']
        if  price > l_80:
            price_range = 'high'
        elif  (price <= l_80) & (price > l_50):
            price_range = 'avg_high'
        elif  (price <= l_50) & (price > l_30):
            price_range = 'avg'
        elif  price <= l_30:
            price_range = 'low'
        ranges.append(price_range)
        i+=1

    df['price_range'] = ranges
    return df


def add_yr_renovated_transformed(df):
    xs = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row['yr_renovated'] == 0:
            x = row['yr_built'].year
        else:
            x = row['yr_renovated']
        xs .append(x)
    df['yr_renovated_transformed'] = pd.to_datetime(xs,format='%Y')
    return df


if __name__ == '__main__':
    path_imputed = './../tmp/data_imputed.csv'
    path_data = './../data/data.csv'
    df_raw = pd.read_csv(path_data)
    df = df_raw.copy()
    if os.path.isfile(path_imputed):
        df_imputed = pd.read_csv(path_imputed)
        df = df_imputed.copy()        
    df['date'] = pd.to_datetime(df['date'])

    names_sqft = [x for x in df_raw.columns if re.search('^sqft_[a-z]+$',x)!=None]
    df['sqft'] = df[names_sqft].sum(1)
#    for k in names_sqft+['sqft_lot15','sqft_living15']:
#        del df[k]
    
    df['yr_built'] = pd.to_datetime(df['yr_built'],format='%Y')
    df = add_yr_renovated_transformed(df)
    df['yr_renovated'] = pd.to_datetime(df['yr_renovated_transformed'],format='%Y')

    df['age'] = [int((x.days)/365) for x in df['date'] - df['yr_built']]
    df['age_renovated'] = [int(x.days/365) for x in df['date'] - df['yr_renovated']]
    names_yr = [x for x in df.columns if re.search('yr_',x)!= None]
    for k in names_yr:
        del df[k]

    transform = ['price','sqft_living','sqft_lot','sqft','sqft_above','sqft_basement',
            'count_markets','count_restaurants','count_stations','count_schools','count_clinics',
            'age','age_renovated']

    for k in transform:
        tmp = df[k]
        tmp[tmp==0] = 0.5
        tmp = np.log10(tmp)
        df['log_'+k] = tmp

    df = add_price_range(df)    


    names_types = ['bedrooms','floors']
    count = df[names_types + ['price']].groupby(names_types).count()
    count.columns=['count']
    mean = df[names_types+['price']].groupby(names_types).mean()
    mean.columns=['mean_price']
    mean.sort_values('mean_price')
    df_types = count.join(mean).sort_values('count',ascending=False)

    #whether to get rid of waterfront
    df['bathrooms']*df['bedrooms']
    df[['waterfront','price']].groupby('waterfront').mean()

    df_clean = df.copy()
    print(df_clean.isnull().sum())
    df_clean = df_clean.dropna()
    path_clean_data = './../tmp/data_clean.csv'        
    df_clean.to_csv(path_clean_data,index=False)
    print('Output to {}'.format(path_clean_data))






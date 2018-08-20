from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd 
import os
import sklearn
import numpy as np 
import re
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	
import datetime
pd.options.mode.chained_assignment = None  # default='warn'

def get_XY(df):
	price = [x for x in df.columns if re.search('price',x )!= None][0]
	Y = df[price].values
	for k in [price,'id','split']:
		del df[k]
	X = df.values
	return X, Y, list(df.columns)
def count_type(ls):
	count = dict()
	for k in ls:
		v = count.get(k,0)
		count[k] = v+1
	return count
def fold_col_by_percentile(df,key='locality',percentile=30):
	d_loc = sorted(count_type(df[key]).items(), key=lambda x:x[1],reverse=True)
	l = np.percentile(list(map(lambda x: x[1],d_loc)),percentile)
	keep = [x for x,y in list(filter(lambda x:x[1]>l,d_loc))]
	dump = [x for x,y in list(filter(lambda x:x[1]<=l,d_loc))]

	for name in dump:
		df.loc[df[key]==name,key] = 'Others'
	output = df
	return output

def test_model(X_val,Y_val,model):
	Yhat = model.predict(X_val)
	normalized_mae = sklearn.metrics.mean_absolute_error(Y_val,Yhat)/Y_val.mean()
	print('Normalized MAE:',normalized_mae)
	return normalized_mae

def test_model_log(X_val,Y_val,model):
	Yhat = model.predict(X_val)
	normalized_mae = sklearn.metrics.mean_absolute_error(10**Y_val,10**Yhat)/10**Y_val.mean()
	print('Normalized MAE LOG:',normalized_mae)
	return normalized_mae


def train_RFE(X_train,Y_train,n_features=30):
	from sklearn.linear_model import LinearRegression 
	from sklearn.feature_selection import RFE
	lm = LinearRegression()
	model = RFE(lm, n_features)             # running RFE
	model = model.fit(X_train, Y_train)
	return model



def train_randomforest(X_train,Y_train):

	model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
			max_features='auto', max_leaf_nodes=None,
			min_impurity_decrease=0.0, min_impurity_split=None,
			min_samples_leaf=1, min_samples_split=10,
			min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=3,
			oob_score=False, random_state=0, verbose=0, warm_start=False)

	model.fit(X_train, Y_train)
	return model 


def update_RF_imp_logs(d_imp):
	"""
	features =list(set( list(df.columns)+transform))
	df_features = pd.DataFrame(columns=['EXP_ID']+features)
	df_features.to_csv(path_features,index=False)
	"""
	path_features = './../logs/RF_importance_logs.csv'
	df_features = pd.read_csv(path_features)
	row_features = pd.DataFrame([d_imp])
	if df_features.loc[df_features['EXP_ID']==d_imp['EXP_ID'],].shape[0] == 0:
		df_features = df_features.append(row_features)
	else:
		idx = df_features.loc[df_features['EXP_ID']==d_imp['EXP_ID'],].index[0]
		df_features.iloc[idx] = row_features.iloc[0]
	df_features = df_features.sort_values('EXP_ID',ascending=True)
	df_features.to_csv(path_features,index=False)

	return df_features


def start_exp(exp):
	#setup
	"""
	OUTLIER_FLOOR = 99
	FOLD_CEIL = 30
	IS_LOG_PRICE = False
	IS_SELECT_RFE = False
	IS_LOG_SQFT = True
	IS_LOG_COUNT = True
	IS_LOG_AGE = True
	IS_KEEP_LOCALITY = True
	IS_KEEP_NEIGHBORHOOD = True
	"""
	EXP_ID = exp['EXP_ID']
	OUTLIER_FLOOR = exp['OUTLIER_FLOOR']
	FOLD_CEIL = exp['FOLD_CEIL']
	IS_LOG_PRICE = exp['IS_LOG_PRICE']
	IS_SELECT_RFE = exp['IS_SELECT_RFE']
	IS_LOG_SQFT = exp['IS_LOG_SQFT']
	IS_LOG_COUNT = exp['IS_LOG_COUNT']
	IS_LOG_AGE = exp['IS_LOG_AGE']
	IS_KEEP_LOCALITY = exp['IS_KEEP_LOCALITY']
	IS_KEEP_NEIGHBORHOOD = exp['IS_KEEP_NEIGHBORHOOD']
	IS_KEEP_COUNT = exp['IS_KEEP_COUNT']
	IS_KEEP_AGE = exp['IS_KEEP_AGE']

	start_time = datetime.datetime.now()

	#read the data
	path_data_ml = os.path.join(CWDIR,'./../data/data_ml.csv')
	df_ml = pd.read_csv(path_data_ml).dropna()
	df = df_ml.copy()


	names_sqft = ['sqft_living','sqft_lot','sqft','sqft_above','sqft_basement',]
	names_count = ['count_markets','count_restaurants','count_stations','count_schools','count_clinics']
	names_age = ['age','age_renovated']
	names_price = ['price']


	transform = ['price','sqft_living','sqft_lot','sqft','sqft_above','sqft_basement']
	if IS_KEEP_COUNT !=True:
		for k in names_count+['log_'+k for k in names_count]:
			del df[k]
	else:
		transform += names_count

	if IS_KEEP_AGE !=True:
		for k in names_age+['log_'+k for k in names_age]:
			del df[k]
	else:
		transform += names_age


	#delete outlier for numeric data

	try:
		limits =np.percentile(df[['bedrooms','bathrooms','floors']+transform],OUTLIER_FLOOR,axis=0)
	except:
		limits =np.percentile(df[['bedrooms','bathrooms','floors']+['log_'+k for k in transform]],OUTLIER_FLOOR,axis=0)
	judge = (df['bedrooms']<=limits[0]) &(df['bathrooms']<=limits[1]) &(df['floors']<=limits[2])
	df = df.loc[judge,]

	#fold scarce categories for categorical data
	keep_loc = []
	if IS_KEEP_LOCALITY == True:
		keep_loc += ['locality']
	if IS_KEEP_NEIGHBORHOOD == True:
		keep_loc += ['neighborhood']    

	for k in keep_loc:
		df = fold_col_by_percentile(df,k,percentile=FOLD_CEIL)
		dummies = pd.get_dummies(df[k])
		df = pd.concat([df,dummies],axis=1)
		del df[k]

	#log transform data, should be after outlier detection, otherwise will report bug
	keep_not_log = []
	if IS_LOG_SQFT != True:
		keep_not_log += names_sqft
	if IS_LOG_COUNT != True:
		keep_not_log += names_count
	if IS_LOG_AGE != True:
		keep_not_log += names_age
	if IS_LOG_PRICE != True:
		keep_not_log += names_price

	for k in transform:
		if k not in keep_not_log:
			del df[k]
		else:
			del df['log_'+k]



	#get rid of redundant locality categories
	names_loc = ['locality','neighborhood','country','administrative_area_level_1','administrative_area_level_2','route']
	for k in names_loc:
		if k not in keep_loc:
			del df[k]

	#prepare training/testing dataset
	df_train = df.loc[df['split']=='train',]
	X_train, Y_train, names = get_XY(df_train)

	df_val = df.loc[df['split']=='val',]
	df_outlier = df.loc[~judge,]
	if df_outlier.shape[0]>0:
		df_val = pd.concat([df_val,df_outlier],axis=0)
	X_val, Y_val, names = get_XY(df_val)

	df_test = df.loc[df['split']=='test',]
	X_test, Y_test, names = get_XY(df_test)

	#feature selection
	if IS_SELECT_RFE == True:
		n_features = min(40,X_train.shape[1])
		
		model_RFE = train_RFE(X_train,Y_train,n_features=n_features)
		print()
		X_train = X_train[:,model_RFE.support_]
		X_val = X_val[:,model_RFE.support_]

	#train the model
	model = train_randomforest(X_train,Y_train)
	
	d_imp = dict(zip(names,model.feature_importances_))
	d_imp['EXP_ID'] = EXP_ID
	print('Randomforest Feature Importance:')
	sorted_dict = sorted(d_imp.items(), key =lambda x: x[1],reverse=True)
	for k,v in sorted_dict:
		print(k,v)
	df_features = update_RF_imp_logs(d_imp)

	#evaluate the model
	if IS_LOG_PRICE == True:
		mse_val = test_model_log(X_val,Y_val,model)
	else:
		mse_val = test_model(X_val,Y_val,model)

	if IS_LOG_PRICE == True:
		mse_test = test_model_log(X_test,Y_test,model)
	else:
		mse_test = test_model(X_test,Y_test,model)

	exp['mse_val'] = mse_val
	exp['mse_test'] = mse_test
	exp['n_features'] = X_train.shape[1]
	exp['start_time'] = str(start_time.replace(microsecond=0))
	exp['end_time'] = str(datetime.datetime.now().replace(microsecond=0))

	exp['n_tested']+=1

	return exp



if __name__ == '__main__':
	os.system('mkdir -p ./../logs')
	exp_addr = os.path.join(CWDIR,'./../experiments/exp_logs.xlsx')
	df_exp = pd.read_excel(exp_addr)
	for i in range(df_exp.shape[0]):
		if df_exp.iloc[i]['n_tested'] == 0:
			exp = df_exp.iloc[i]
			print(exp)
			try:
				exp = start_exp(exp)
			except Exception as e:
				print(e) 
			df_exp.loc[i,:] = exp
#			df_exp = df_exp.sortlevel(axis=1)
			df_exp.to_excel(exp_addr,index=False)	

	




	





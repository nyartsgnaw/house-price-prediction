import pandas as pd 
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
	path_data_clean = os.path.join(CWDIR,'./../tmp/data_clean.csv')
	path_data_ml = os.path.join(CWDIR,'./../data/data_ml.csv')
	df_clean = pd.read_csv(path_data_clean).dropna()


	df = df_clean.copy()
	df_test = df.loc[df['locality']=='Issaquah',]
	df_test['split'] = 'test'
	df_test['locality'] = 'Others'
	df_test['neighborhood'] = 'Others'
	df_test['administrative_area_level_1'] = 'Others'
	df_test['administrative_area_level_2'] = 'Others'
	df_test['country'] = 'Others'
	df_test['route'] = 'Others'

	from sklearn.model_selection import train_test_split
	df_train, df_val = train_test_split(df.loc[df['locality']!='Issaquah',],train_size=0.8,random_state=100)
	df_train['split'] = 'train'
	df_val['split'] = 'val'

	df_ml = pd.concat([df_train,df_val,df_test],axis=0).sort_index()

	# below should not be used for training/prediction
	for k in ['date','price_range','lat','long','zipcode']:
		del df_ml[k]

	df_ml.to_csv(path_data_ml,index=False)
	print('Output to {}'.format(path_data_ml))
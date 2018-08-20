import pandas as pd 
import numpy as np 
import googlemaps
import time
import re 
import os 

api_key = 'AIzaSyC_kXSRnBiQk5ZKSvctKOF7wfDHUJ68P8A'
def reverse_geocode(codes):
	names_info = ['route','neighborhood','locality','administrative_area_level_2',
				'administrative_area_level_1','country']
	gmaps = googlemaps.Client(key=api_key)
	output = []
	i = 0
	while i <len(codes):
		try:
			lat,lon = codes[i]
			result = gmaps.reverse_geocode((lat,lon))
			if len(result)>0:
				components = result[0]['address_components']
				new = {'i':i,
						'lat':lat,
						'long':lon}
				for k in names_info:
					j = 0
					while j < len(components):
						if k in components[j]['types']:
							new[k] = components[j]['long_name']
						j+=1
					new[k] = new.get(k,'')
			else:
				for k in names_info:
					new.update({k:''})

			output.append(new)
		except Exception as e:
			print(e)
			error = e
			if error.status == 'ZERO_RESULTS':
				for k in names_info:
					new.update({k:''})
					output.append(new)
			else:
				time.sleep(10)                            
				i-=1
		i+=1
		print(i,new)
	return output


def impute_missing(df_geo):
	names_info = ['route','neighborhood','locality','administrative_area_level_2',
				'administrative_area_level_1','country']
	length = len(names_info)
	i = 0
	while i < df_geo.shape[0]:
		
		row = df_geo.iloc[i]
		if row.isna().sum()>0:
			j =2
			while j >=0:
				if pd.isna(row[names_info[j]]):
					filled = False
					k = j-1
					while (filled == False)&(k>=0):
						if not pd.isna(row[names_info[k]]):
							row[names_info[j]] = row[names_info[k]]
							filled= True    
							print(i)                        
						k+=1
				j-=1
			df_geo.iloc[i] = row
		i+=1
	return df_geo


def impute_missing(df_geo):
	names_info = ['route','neighborhood','locality','administrative_area_level_2',
				'administrative_area_level_1','country']
	length = len(names_info)
	i = 0
	while i < df_geo.shape[0]:
		
		row = df_geo.iloc[i]
		if row.isna().sum()>0:
			j =2
			while j >=0:
				if pd.isna(row[names_info[j]]):
					filled = False
					k = j-1
					while (filled == False)&(k>=0):
						if not pd.isna(row[names_info[k]]):
							row[names_info[j]] = row[names_info[k]]
							filled= True    
							print(i)                        
						k+=1
				j-=1
			df_geo.iloc[i] = row
		i+=1
	return df_geo



def count_place_nearby(codes,keyword='supermarket',radius=800,types=['grocery_or_supermarket','supermarket','store','liquor_store']):
	gmaps = googlemaps.Client(key=api_key)
	output = []
	i = 0
	while i <len(codes):
		try:
			lat,lon = codes[i]
			result = gmaps.places_nearby(location=(lat,lon),radius=radius,keyword=keyword) #800 meters stands for 10-minutes walking disctance
			if len(result) >0:
				length = len(result['results'])
				count = length
				j = 0
				count = 0
				while j<length:
#					print(result['results'][j]['types'][0])
					if result['results'][j]['types'][0] in types:
						count+=1
					j+=1 
			else:
				count = -1
		except Exception as e:
			print(e)
			error = e
			if error.status == 'ZERO_RESULTS':
					count = 0
			else:
				time.sleep(10)                            
				i-=1
		output.append(count)
		i+=1
		print('{}/{} th {} \'s count: {}'.format(i,len(codes),keyword,count))

	return output


if __name__ == '__main__':
	path_data = './../data/data.csv'
	path_geo = './../tmp/df_geo.csv'
	path_geo_clean = './../tmp/df_geo_clean.csv'
	path_geo_count_clean = './../tmp/df_geo_count_clean.csv'
	path_imputed = './../tmp/data_imputed.csv'

	df_raw = pd.read_csv(path_data)
	if os.path.isfile(path_geo):
		df_geo = pd.read_csv(path_geo)
	else:
		codes = list(zip(df_raw['lat'],df_raw['long']))
		df_geo = reverse_geocode(codes)
		df_geo.to_csv(path_geo,index=False)

	if os.path.isfile(path_geo_clean):
		df_geo_clean = pd.read_csv(path_geo_clean)
	else:
		df_geo_clean = df_geo.copy()
		df_geo_clean.loc[df_geo_clean['locality'].isna(),'locality'] = df_geo_clean.loc[df_geo_clean['locality'].isna(),'administrative_area_level_2']
		df_geo_clean.loc[df_geo_clean['neighborhood'].isna(),'neighborhood'] = df_geo_clean.loc[df_geo_clean['neighborhood'].isna(),'locality']
		#df_geo_clean = impute_missing(df_geo_clean)   #more detailed imputation for data with bad quality
		df_geo_clean.to_csv(path_geo_clean,index=False)


	keys = ['neighborhood','locality','administrative_area_level_2',
				'administrative_area_level_1','country']
	if os.path.isfile(path_geo_count_clean):
		df_geo_count_clean = pd.read_csv(path_geo_count_clean)
	else:
		mean = df_geo_clean[keys+['lat','long']].groupby(keys).mean()
		#get the counts from google api
		market_types = ['grocery_or_supermarket','supermarket','store','liquor_store']
		food_types = ['restaurant','food','bar','cafe','bakery','night_club']
		transit_types = ['bus_station','transit_station','train_station','light_rail_station']
		school_types = ['school','library','university']
		hospital_types = ['health','doctor','hospital','dentist','pharmacy']
		codes_mean = list(zip(mean['lat'],mean['long']))
		count_markets = count_place_nearby(codes_mean,keyword='supermarket',radius= 800,types=market_types) #800 is 10 minutes' walk
		count_restaurants = count_place_nearby(codes_mean,keyword='restaurant',radius= 800,types=food_types)
		count_stations = count_place_nearby(codes_mean,keyword='transit station',radius= 800,types=transit_types)
		count_clinics = count_place_nearby(codes_mean,keyword='clinic',radius= 2400,types=hospital_types) #2400 is 3 minutes' drive
		count_schools = count_place_nearby(codes_mean,keyword='school',radius= 2400, types=school_types)
		#create new table
		df_geo_count_clean = mean.copy()
		df_geo_count_clean['count_markets'] = count_markets
		df_geo_count_clean['count_restaurants'] = count_restaurants
		df_geo_count_clean['count_stations'] = count_stations
		df_geo_count_clean['count_schools'] = count_schools
		df_geo_count_clean['count_clinics'] = count_clinics
		df_geo_count_clean = df_geo_count_clean.reset_index(keys)
		df_geo_count_clean.to_csv(path_geo_count_clean,index=False)
	tmp = pd.merge(df_geo_clean, df_geo_count_clean, on =keys,how='left')
	for k in [x for x in tmp.columns if re.search('lat|long|^i$',x ) !=None]:
		del tmp[k]
	df_imputed = pd.concat([df_raw,tmp],axis=1)
	df_imputed.to_csv(path_imputed,index=False)
	print('Output to {}'.format(path_imputed))
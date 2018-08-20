import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import pandas as pd 
import numpy as np 
import seaborn as sns
from pylab import rcParams
from pandas.plotting import scatter_matrix
import re
def sub_plots1(loop,nrow,ncol,addr='foo.png'):
    FIGRE_SIZE = (20,int(20*nrow/ncol))
    fig,axes = plt.subplots(nrow,ncol,sharey=True,figsize=FIGRE_SIZE)
    fig.suptitle(addr[:addr.find('.')], fontsize=20)
    i=0
    while i < nrow:
        j=0
        while j < ncol:
            ax = axes[i,j]
            ax.grid(linestyle='-',linewidth=.25,color='gray')
            try:
                _t, _d = next(loop)
            except:
                print(_t)
                break                
            sns.distplot(_d, hist = False, kde = True,
                        kde_kws = {'linewidth': 3},
                        label = _t, ax = ax)
            ax.axvline(x=_d.mean(),color='r')
            ax.label_outer()
            ax.set(xlabel='values', ylabel='density')

            ax.text(0.05, .95, _d.count(), transform = axes[i, j].transAxes,
                        verticalalignment = 'top',fontsize=15)
                    
            j+=1
        i+=1

    fig.subplots_adjust(left=0.05, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
    fig.savefig(addr)
    if DISPLAY == True:
        plt.show()

    plt.close()
    return 

def sub_plots2(loop,nrow,ncol,addr='foo.png'):
    FIGRE_SIZE = (20,int(20*nrow/ncol))
    fig,axes = plt.subplots(nrow,ncol,sharey=True,figsize=FIGRE_SIZE)
    fig.suptitle(addr[:addr.find('.')], fontsize=20)
    i=0
    while i < nrow:
        j=0
        while j < ncol:
            ax = axes[i,j]
            ax.grid(linestyle='-',linewidth=.25,color='gray')
            try:
                _t, _d = next(loop)                
            except:
                print(_t)
                break
            
            try:
                dtime = pd.to_datetime(list(_d.index),format='%Y-%m-%d')
                """
                low_time = min(_d.index)
                up_time = max(_d.index)
                up_time = up_time.replace(day=up_time.day+1)
                dtime = mdates.drange(low_time,up_time,delta=datetime.timedelta(days=1))
                """                
            except:
                dtime = _d.values
                
            ax.plot(dtime,_d.values, alpha=0.5)

            ax.label_outer()

            ax.set(xlabel='year', ylabel='values')
            ax.text(0.05, .95, _t, transform = axes[i, j].transAxes,
                        verticalalignment = 'top')


            j+=1
        i+=1
    fig.subplots_adjust(left=0.05, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
    fig.savefig(addr)
    if DISPLAY == True:
        plt.show()
    plt.close()
    return 

def sub_plots3(loop,nrow,ncol,addr='foo.png'):
    FIGRE_SIZE = (20,int(20*nrow/ncol))
    fig,axes = plt.subplots(nrow,ncol,sharey=True,figsize=FIGRE_SIZE)
    fig.suptitle(addr[:addr.find('.')], fontsize=20)
    i=0
    while i < nrow:
        j=0
        while j < ncol:
            ax = axes[i,j]
            ax.grid(linestyle='-',linewidth=.25,color='gray')
            try:
                _t, _d = next(loop)
            except:
                print(_t)
                break    
            x,y = _d            
            ax.scatter(x,y)
            ax.label_outer()
            ax.set(xlabel='values', ylabel='density')

            ax.text(0.05, .95, _t, transform = axes[i, j].transAxes,
                        verticalalignment = 'top',fontsize=15)
            j+=1
        i+=1

    fig.subplots_adjust(left=0.05, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
    fig.savefig(addr)
    if DISPLAY == True:
        plt.show()
    plt.close()
    return 

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

def get_group_avg(df,group_by='locality',value='price'):
    mean = df[[group_by,value]].groupby(group_by).mean()
    mean.columns = [value+'_avg_'+group_by]
    count = df[[group_by,value]].groupby(group_by).count()
    count.columns = ['count_'+group_by]
    std = df[[group_by,value]].groupby(group_by).std()
    std.columns = ['std_'+group_by]
    comb = mean.join(count).join(std).sort_values(value+'_avg_'+group_by).reset_index(group_by)

    return comb

def draw_density(df,group_by='locality',value='sqft',nrow=None,ncol=None):
    comb = get_group_avg(df,group_by=group_by,value=value)
    loop_ls = []
    for x in comb[group_by]:
            _d = df.loc[df[group_by]==x,value]
            _t = '\"{}\"'.format(x)
            loop_ls.append((_t,_d))
    
    sqrt = np.sqrt(len(loop_ls))
    if nrow == None:
        nrow = int(np.floor(sqrt))
    if ncol == None:
        ncol = int(np.ceil(sqrt))
    sub_plots1(iter(loop_ls),nrow,ncol,addr='density_{}_by_{}.png'.format(value,group_by))
    return comb
def draw_time_series(df,group_by='locality',value='sqft',nrow=None,ncol=None):
    comb = get_group_avg(df,group_by=group_by,value=value)
    loop_ls = []
    for x in comb[group_by]:
        _d = df.loc[df[group_by]==x,['date',value]]
        v = _d.groupby('date')[value].mean()
        _t = '\"{}\"'.format(x)
        loop_ls.append([_t,v])

    sqrt = np.sqrt(len(loop_ls))
    if nrow == None:
        nrow = int(np.floor(sqrt))
    if ncol == None:
        ncol = int(np.ceil(sqrt))
    sub_plots2(iter(loop_ls),nrow,ncol,addr='time_series_{}_by_{}.png'.format(value,group_by))
    return comb

def draw_scatter(df,nrow=None,ncol=None):
    price = [x for x in df.columns if re.search('price',x)!=None][0]
    loop_ls = []
    for k in df.columns:
        _d = df[k], df[price]
        _t = '{} vs {}'.format(k,price)
        loop_ls.append([_t,_d])
    sqrt = np.sqrt(len(loop_ls))
    if nrow == None:
        nrow = int(np.floor(sqrt))
    if ncol == None:
        ncol = int(np.ceil(sqrt))
    sub_plots3(iter(loop_ls),nrow,ncol,addr='scatter_{}_by_{}.png'.format('all',price))
    return 



if __name__ == '__main__':

    #steup
    DISPLAY = False  
    GROUP_DENSITY = True
    DENSITY_BY_LOCALITY = True
    SCATTER = True
    DENSITY_ALL = True
    DENSITY_COMPARE_LOG = True
    
    #read the data
    path_clean_data = './../tmp/data_clean.csv'  
    df_clean = pd.read_csv(path_clean_data)
    df = df_clean.copy()

    #fold the categorical columns
    df = fold_col_by_percentile(df,key='locality',percentile=30)
    df = fold_col_by_percentile(df,key='neighborhood',percentile=70)
    df = fold_col_by_percentile(df,key='country',percentile=30)
    df = fold_col_by_percentile(df,key='administrative_area_level_2',percentile=30)
    df = fold_col_by_percentile(df,key='administrative_area_level_1',percentile=30)



    #print the categories distribution 
    stats = dict()
    for k in ['bedrooms','bathrooms','floors','waterfront',
                'condition','grade','price_range',
                'administrative_area_level_2','locality','neighborhood',
                'administrative_area_level_1','country']:
        stats[k] = count_type(df[k])
        print('{}: {}'.format(k,stats[k]))


    """
    df.groupby(['locality']).price.plot.kde()
    plt.legend()
    plt.show()
    """

    if DENSITY_BY_LOCALITY == True:
        print('Drawing density_by_locality...')
        comb = draw_density(df,group_by='locality',value='price',nrow=4,ncol=8)
        comb = draw_density(df,group_by='locality',value='sqft',nrow=4,ncol=8)
        comb = draw_time_series(df,group_by='locality',value='price',nrow=4,ncol=8)
        comb = draw_time_series(df,group_by='locality',value='sqft',nrow=4,ncol=8)
    
    transform = ['price','sqft_living','sqft_lot','sqft','sqft_above','sqft_basement',
            'count_markets','count_restaurants','count_stations', 'count_schools','count_clinics',
            'age','age_renovated']
    interested = ['bedrooms', 'bathrooms','floors', 'waterfront', 'view', 'condition', 'grade']
    if SCATTER == True:
        print('Drawing scatter...')
        draw_scatter(df[interested+['log_'+k for k in transform]],nrow=4,ncol=5)
        if DISPLAY == True:
            plt.show()
        plt.close()

        draw_scatter(df[interested+[k for k in transform]],nrow=4,ncol=5)
        if DISPLAY == True:
            plt.show()
        plt.close()


    if DENSITY_ALL == True:
        addr = 'density_all.png'
        print('Drawing density_all...')
        (df[interested+['log_'+k for k in transform]]).plot(kind='density', subplots=True, layout=(3,7), sharex=False,figsize=(25,10))
        plt.suptitle(addr[:addr.find('.')], fontsize=20)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=1, top=0.9, wspace=0, hspace=0.3)
        plt.savefig(addr)
        if DISPLAY == True:
            plt.show()
        plt.close()

    if DENSITY_COMPARE_LOG == True:
        addr = 'density_compare_log.png'
        print('Drawing density_compare_log...')
        df[transform+['log_'+k for k in transform]].plot(kind='density', subplots=True, layout=(2,13), sharex=False,figsize=(25,10))
        plt.suptitle(addr[:addr.find('.')], fontsize=20)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=1, top=0.9, wspace=0, hspace=0.3)
        plt.savefig(addr)
        if DISPLAY == True:
            plt.show()
        plt.close()

    mean= df_clean[['locality','price']].groupby('locality').mean()
    mean.columns = ['avg_price']
    mean = mean.sort_values('avg_price',ascending=False).reset_index('locality')
    mean.to_csv('avg_price_by_locality.csv',index=False)

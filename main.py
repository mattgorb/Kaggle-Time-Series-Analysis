

 #import matplotlib as plt
import pandas as pd
import numpy as np
from fbprophet import Prophet
import sys
#import matplotlib.pyplot as plt
import multiprocessing
import time
import math
from threading import Thread
import threading
import os
import errno

def list_fds():
    """List process currently open FDs and their target """
    if sys.platform != 'linux2':
        raise NotImplementedError('Unsupported platform: %s' % sys.platform)

    ret = {}
    base = '/proc/self/fd'
    for num in os.listdir(base):
        path = None
        try:
            path = os.readlink(os.path.join(base, num))
        except OSError as err:
            # Last FD is always the "listdir" one (which may be closed)
            if err.errno != errno.ENOENT:
                raise
        ret[int(num)] = path

    return ret


class suppress_stdout_stderr(object):
    def __init__(self):
	null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
	save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
	os.dup2(null_fds[0], 1)
	os.dup2(null_fds[1], 2)

    def __exit__(self, *_):
	os.dup2(save_fds[0], 1)
	os.dup2(save_fds[1], 2)
	os.close(null_fds[0])
	os.close(null_fds[1])

N=60


print('Reading data...')
train_1 = pd.read_csv('input/train_1.csv')
train, test = train_1.T.iloc[0:-N,:], train_1.T.iloc[-N:,:]
	
ss_1 = pd.read_csv('input/sample_submission_1.csv')
key = pd.read_csv("input/key_1.csv", index_col='Page')

if(os.path.isfile("submission.csv")==False):
	print "Creating submission.csv"
	print('Processing...')
	ids = key.Id.values
	pages = key.index.values

	print('key_1...')
	d_pages = {}
	for id, page in zip(ids, pages):
    	    d_pages[id] = page[:-11]

	print('train_1...')
	pages = train_1.Page.values

	visits = np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -56:], axis=1)))

	d_visits = {}
	for page, visits_number in zip(pages, visits):
		d_visits[page] = visits_number

	print('Modifying sample submission...')
	ss_ids = ss_1.Id.values
	ss_visits = ss_1.Visits.values

	for i, ss_id in enumerate(ss_ids):
    	    ss_visits[i] = d_visits[d_pages[ss_id]]

	print('Saving submission...')
	submission = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})
	submission.to_csv('submission.csv', index=False)


print('Reading submission.csv...')
submission=pd.read_csv('submission.csv', index_col='Id')
	
all_data = pd.read_csv("input/train_1.csv").T
names=all_data.T.iloc[:,0]

#all_data_minus_name=all_data_cleaned.iloc[:,1:]
test_cleaned = test.T.fillna(method='ffill').T
train_cleaned = train.T.iloc[:,1:].fillna(method='ffill').T


all_data_cleaned=all_data.T.iloc[:,1:].fillna(method='ffill').T





		

def smape(self,y_true, y_pred,page_name):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    #print ("SMAPE score for "+str(np.asarray(page_name))+": "+str(200 * np.median(diff)))
    return 200 * np.median(diff)


def KeyLookup_Add(self,y_forecasted, page_name):
    try:
	#global submission
	for i in y_forecasted:
	    lookupstr=page_name+'_'+str(i[0]).split(' ')[0]
	    ID=key.loc[str(lookupstr)].Id
	    if(int(i[1])<0):
	        i[1]==0
	    i[1]=int(i[1])

	    if(ID in submission.index):
		submission.loc[ID, 'Visits']=i[1]
	    else:
		print "Not in index"    

    except Exception as e: 
	print "Exception1:"
	print(e)





def RunForecast(self,i, std,win):
    try:
	data=all_data_cleaned.iloc[:,i].to_frame()
    	data.columns = ['visits']
    	data['median'] = pd.Series().rolling(min_periods=1, window=win, center=False).median()
    	data.ix[np.abs(data.visits-data.visits.median())>=(std*data.visits.std()),'visits'] \
		= data.ix[np.abs(data.visits-data.visits.median())>=(std*data.visits.std()),'median']   	
	data.index = pd.to_datetime(data.index)
	X = pd.DataFrame(index=range(0,len(data)))
	X['ds'] = data.index
	X['y'] = data['visits'].values
    	X.tail()
#CAUSED EXCEPTION with suppressing_stdout	    	with suppress_stdout_stderr():
	proph = Prophet(yearly_seasonality=True, uncertainty_samples=0)
	proph.fit(X)
	future =proph.make_future_dataframe(periods=60)
	future.tail()
	forecast = proph.predict(future)
	forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

	y_forecasted=forecast.iloc[-N:,[0,16]].values
    	y_dates=future.iloc[-N:].values
	page_name = names[i]
	KeyLookup_Add(y_forecasted,page_name)
	del data,X,y_forecasted, y_dates,page_name	
	with open("submission.csv", "w") as f:
		f.truncate()
		submission.to_csv(f, index=True)
#		print "Run"
	openfiles=list_fds()
	print openfiles
	print str(i)+" completed"
    except Exception as e:
	print "Exception2:"
	print (e)


def TuneValidationParams(self,i):
    std_mult_size=[1.3,1.4,1.5,1.6]
    window_size=[40,50,60]
    min_std=0
    min_win=0
    min_score=200
    continueTuning=True
    for std_mult in std_mult_size:
	for window_ in window_size:
	    data=train_cleaned.iloc[:,i].to_frame()
    	    data.columns = ['visits']
	    data['median'] = pd.Series().rolling(min_periods=1, window=window_, center=False).median()
	    data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'visits'] \
		= data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'median']
	    data.index = pd.to_datetime(data.index)
	    X = pd.DataFrame(index=range(0,len(data)))
	    X['ds'] = data.index
	    X['y'] = data['visits'].values
	    X.tail()
	    #with suppress_stdout_stderr():
	    m = Prophet(yearly_seasonality=True,uncertainty_samples=0)
	    m.fit(X)
	    future = m.make_future_dataframe(periods=N)
	    future.tail()
	
	    forecast = m.predict(future)
	    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
	    y_truth = test_cleaned.iloc[:,i].values
	    y_forecasted = forecast.iloc[-N:,2].values
	    print names[i].split("_")[0]
	    page_name=names[i][0].split("_")[0]
	        
	    print page_name
	    

	    score=smape(y_truth,y_forecasted,page_name)
	    if(score<min_score):
		min_score=score
		min_std=std_mult
		min_win=window_
	    
	    del m,future, forecast,  data,X, y_truth, y_forecasted
	    if(min_score>70 or min_score<30):
		continueTuning=False
		break
	if(continueTuning==False):
		break

    #if(min_score<50):
    RunForecast(i,min_std,min_win)
    return submission
    #else:
	#print "Skipping FbProphet prediction for "+str(i)	    

train=Train( submission, all_data_cleaned,test_cleaned,train_cleaned,names, N)

print "Running..."
pool = multiprocessing.Pool(24)
result_list = pool.map(train.TuneValidationParams, (i for i in range(148000)))





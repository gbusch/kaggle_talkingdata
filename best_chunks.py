import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt
import gc

def procdata(dat, ip_count):
    print("Processing data...")
    label = dat['is_attributed']
    X = dat.drop(['is_attributed', 'click_time'], axis=1)
    X['day'] = dat['click_time'].dt.day
    X['hour'] = dat['click_time'].dt.hour
    del dat
    gc.collect()
    
    #clicks per IP
    X = pd.merge(X, ip_count, on='ip', how='left')
    X.drop(['ip'], axis=1, inplace=True)
    gc.collect()
    
    X = np.array(X)
    label = np.array(label)
    
    return X, label


def prediction(model, ip_count):    
	print("started reading and processing submission test data...")
	dat2 = pd.read_csv("./input/test.csv", dtype=dtypes, usecols=['ip','app','device','os','channel','click_time'], parse_dates=['click_time'])
	test = dat2.drop(['click_time'], axis=1)
	test['day'] = dat2['click_time'].dt.day
	test['hour'] = dat2['click_time'].dt.hour
	del dat2
	gc.collect()

	#clicks per IP
	test = pd.merge(test, ip_count, on='ip', how='left')
	test.drop(['ip'], axis=1, inplace=True)
	gc.collect()

	test = np.array(test)

	y_pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
	del test
	gc.collect()
	
	return y_pred


	

params = {'eta': 0.1, 
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.7,           
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':99.7,
          'eval_metric': 'auc', 
          'nthread':4,
          'random_state': 123, 
          'silent': True}



dtypes={'ip': 'uint32','app': 'uint16','device': 'uint16','os': 'uint16','channel': 'uint16','is_attributed': 'uint8', 'click_id': 'uint32'}

pred = []


nchunks = 10#0#5
lchunk = 12257838#24515677
lval = 5000000

d6 = np.arange(1, 9308569)
d7a8 = np.arange(9308569, 131886954)
d9 = np.arange(131886954, 184903891)

start = time.time()
random.shuffle(d7a8)
random.shuffle(d9)
end = time.time()
delta = end - start
print("took {} seconds to process".format(delta))

print("read IP column for counting")
train = pd.read_csv('./input/train.csv', dtype=dtypes, usecols=['ip'])
suppl = pd.read_csv('./input/test_supplement.csv', dtype=dtypes, usecols=['ip'])
merge = pd.concat([train, suppl])
del train, suppl
gc.collect()

print("calculate clicks per IP")
merge = merge.reset_index()
ip_count = merge.groupby(['ip'])['index'].count().reset_index()
ip_count.columns = ['ip', 'clicks_per_ip']
del merge
gc.collect()


for chunk_no in range(nchunks):
	start = time.time()
	skchunk = np.concatenate((d6, d7a8[:chunk_no*lchunk], d7a8[(chunk_no+1)*lchunk:], d9))
	skchunk_v = np.concatenate((d6, d7a8, d9[lval:]))
	gc.collect()

	print('load chunk...')
	chunk_train = pd.read_csv('./input/train.csv', skiprows=skchunk, dtype=dtypes, \
	usecols=['ip','app','device','os','channel','click_time','is_attributed'], parse_dates=['click_time'])
	del skchunk
	gc.collect

	X_train, y_train = procdata(chunk_train, ip_count)
	del chunk_train
	gc.collect()

	chunk_valid = pd.read_csv('./input/train.csv', skiprows=skchunk_v, dtype=dtypes, \
	usecols=['ip','app','device','os','channel','click_time','is_attributed'], parse_dates=['click_time'])
	del skchunk_v
	gc.collect()

	X_valid, y_valid = procdata(chunk_valid, ip_count)
	del chunk_valid
	gc.collect()
    
	watchlist = [(xgb.DMatrix(X_train, y_train), 'train'),(xgb.DMatrix(X_valid, y_valid), 'valid')]
	start = time.time()
	if chunk_no < 1:
		model = xgb.train(params, xgb.DMatrix(X_train, y_train), 250, watchlist, maximize=True, early_stopping_rounds = 10, verbose_eval=5)
	else:
		model = xgb.train(params, xgb.DMatrix(X_train, y_train), 250, watchlist, maximize=True, early_stopping_rounds = 10, verbose_eval=5, xgb_model=model)
	end = time.time()
	delta = end - start
	print("chunk no {}, took {} seconds to fit".format(chunk_no, delta))

	del X_train
	del y_train
	del X_valid
	del y_valid
	gc.collect()
	
	y_pred = prediction(model, ip_count)
	pred.append(y_pred)
	
	del y_pred
	gc.collect()


pred = np.array(pred)

dat2 = pd.read_csv("./input/test.csv", dtype=dtypes, usecols=['click_id'])
subm_label = dat2['click_id']

subm = pd.DataFrame()
subm['click_id'] = subm_label
subm['is_attributed'] = np.mean(pred, axis=0)
subm.to_csv("subm_prob_allch_ipcount.csv", float_format='%.8f', index=False)

import pandas as pd
import numpy as np
import nltk
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import math
from sklearn.decomposition import PCA



#import pandas.io.data
data = pd.DataFrame()

def dataRead(filename):
    df = pd.read_csv(filename,skipinitialspace=True)
    
    return df
    
def dataAnalysis(filename):
    df = dataRead(filename)
    df['VEHICLE_TYPE'] = map(lambda x: x.upper(), df['VEHICLE_TYPE'])     
    
    #Now we remove the wait-time and total-luggage-weight and check for percentage of missing data points
    wf2 = df.drop(['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT'],axis=1)
    print 'New workframe after deleting waait-time and total-luggage-weight'
    print wf2.shape
    
    
    return wf2.dropna()

    
def dataProcess(filename):
    tf = dataAnalysis(filename) #temporary frame
    
    
    tf['TIMESTAMP'] = map(lambda x: pd.to_datetime(x), tf['TIMESTAMP'])
    
    
    tf['YEAR'] = map(lambda dt: dt.date().year, tf['TIMESTAMP'])
    tf['MONTH'] = map(lambda dt: dt.date().month, tf['TIMESTAMP'])
    tf['DAY'] = map(lambda dt: dt.date().day, tf['TIMESTAMP'])
    tf['HOUR'] = map(lambda dt: dt.time().hour, tf['TIMESTAMP'])
    tf['MINUTE'] = map(lambda dt: dt.time().minute, tf['TIMESTAMP'])
    tf['SECOND'] = map(lambda dt: dt.time().second, tf['TIMESTAMP'])
    
    del tf['TIMESTAMP']
    
    
    mapping = {'BUS':0, 'TAXI AC':1, 'TAXI NON AC':2, 'METRO':3, 'AC BUS':4, 'MINI BUS':5, 'AUTO RICKSHAW':6}
    tf.replace({'VEHICLE_TYPE':mapping}, inplace = True)
    
    tf2 = np.array(tf)
    
    
    return tf2


#PCA applied
def main_pred():
    
    Train_file = 'intracity_fare_train.csv'
    Test_file = 'intracity_fare_test.csv'
    
    train_data = dataProcess(Train_file)
    test_data = dataProcess(Test_file)
    
    train_features = train_data[:,0:8]
    train_features = np.append( train_features, train_data[:,9:15], axis=1 )
    train_labels=train_data[:,8:9]
    
    test_features = test_data
    
    from sklearn.svm import SVR
    from sklearn.metrics import f1_score
    clf=DecisionTreeRegressor()#(C=10.0,class_weight='balanced')
    
    train_labels=(train_labels.ravel())
    
    '''
    pca = PCA(n_components=4)# adjust yourself
    pca.fit(train_features)
    
    train_red_features = pca.transform(train_features)
    test_red_features = pca.transform(test_features)
'''
    clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    
    out_arr = np.vstack((test_data[:,0],pred))
    
    print out_arr.shape
    
    out_df = DataFrame(out_arr.T)
    
    out_df.to_csv('Output-2.csv')

main_pred()    

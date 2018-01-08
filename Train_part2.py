from __future__ import print_function

import pandas as pd
import numpy as np
import nltk
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import math
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer


def dataRead():
    df = pd.read_csv('intracity_fare_train.csv',skipinitialspace=True)

    return df
    
def dataAnalysis():
    df = dataRead()
    df['VEHICLE_TYPE'] = map(lambda x: x.upper(), df['VEHICLE_TYPE'])
    
    '''we set a working frame'''
    wf = df
    
    
    '''Next we need to calculate the perccentage of dataset that has missing data points'''
    
    total_cells = df.shape[0]*df.shape[1]
    n_empty_cells = df.isnull().values.ravel().sum()   #total number of empty cells
    
    n_rows = df.shape[0]
    n_empty_rows = df.shape[0] - wf.dropna().shape[0]
    
    percentage_null_cells = float(n_empty_cells)/total_cells
    percentage_null_rows = float(n_empty_rows)/n_rows
    
        
    
    #Now we remove the wait-time and total-luggage-weight and check for percentage of missing data points
    wf2 = df.drop(['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT'],axis=1)
    
    
    
    wf2_n_rows = wf2.shape[0]
    wf2_n_empty_rows = wf2.shape[0] - wf2.dropna().shape[0]
    wf2_missing_row_ratio = float(wf2_n_empty_rows)/wf2_n_rows
    
    print ('New percentage of empty rows : ', wf2_missing_row_ratio)
    
    return wf2.dropna()

    
def dataProcess():
    tf = dataAnalysis() #temporary frame
    
    
    tf['TIMESTAMP'] = map(lambda x: pd.to_datetime(x), tf['TIMESTAMP'])
    
    
    tf['YEAR'] = map(lambda dt: dt.date().year, tf['TIMESTAMP'])
    tf['MONTH'] = map(lambda dt: dt.date().month, tf['TIMESTAMP'])
    tf['DAY'] = map(lambda dt: dt.date().day, tf['TIMESTAMP'])
    tf['HOUR'] = map(lambda dt: dt.time().hour, tf['TIMESTAMP'])
    tf['MINUTE'] = map(lambda dt: dt.time().minute, tf['TIMESTAMP'])
    tf['SECOND'] = map(lambda dt: dt.time().second, tf['TIMESTAMP'])
    
    del tf['TIMESTAMP']
    
    tf = tf.ix[:,:-1].values
    standard_scaler = StandardScaler()
    tf = standard_scaler.fit_transform(tf)
    
    mapping = {'BUS':0, 'TAXI AC':1, 'TAXI NON AC':2, 'METRO':3, 'AC BUS':4, 'MINI BUS':5, 'AUTO RICKSHAW':6}
    tf.replace({'VEHICLE_TYPE':mapping}, inplace = True)
    
    '''FDA_plot3 = sns.pairplot(tf,kind='reg')
    FDA_plot3.savefig('FDA_plot3.png')
    plt.show()'''
    
    tf2 = np.array(tf)
    
    
    return tf2

def split(features,labels, percentage):
    total_samples = features.shape[0]
    train_samples = int(round(total_samples*(1-percentage)))
    #print train_samples
    train_features = features[0:train_samples,:]
    train_labels =     labels[0:train_samples,:]
    test_features  = features[train_samples:total_samples,:]
    test_labels   =    labels[train_samples:total_samples,:]
    #print test_labels.shape

    return train_features,test_features,train_labels,test_labels

def TrainTest(ratio):
    
    from sklearn.model_selection import train_test_split
    data = dataProcess()
    #train_data = data.as_matrix()
    train_data = data
    #print train_data

    #print train_data[0]

    #print np.transpose(train_data[:,10])

    features = train_data[:,0:8]
    features = np.append( features, train_data[:,9:12], axis=1 )
    #print features

    labels=train_data[:,8:9]
    print (labels)

    ratio=ratio/10.0
    
    #return split(features, labels, ratio)
    return train_test_split(features, labels, test_size = ratio)

def SVR_train(ratio):
    
    train_features,test_features,train_labels,test_labels = TrainTest(ratio)
    
    X = np.vstack((train_features,test_features))
    y = np.vstack((train_labels,test_labels))
    y = (y.ravel())
    
    train_labels=(train_labels.ravel())
    test_labels=(test_labels.ravel())
    
    from sklearn.svm import SVR
    from sklearn.metrics import f1_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import RANSACRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import linear_model
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel':['poly'],'degree':[2,3,4],'C':[1,3,10,30]}]
                    
    scores = {'MAE':make_scorer('neg_mean_absolute_error'),'MSE':make_scorer('neg_mean_squared_error'), 'R2':make_scorer('r2')}
    
    

    '''print('=====================MODEL SELECTION==============================')
    
    gs = GridSearchCV(SVR(),
                  param_grid=tuned_parameters,
                  scoring=scores, cv=5, refit='R2')
    
    
    
    print(X.shape,' ',y.shape)
    
    gs.fit(X,y)
    results = gs.cv_results_
    
    print('results = ', results)'''
    #========================================================================================
    
    '''for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVR(), tuned_parameters, cv=20,
                       scoring=scores)
        clf.fit(train_features, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed Task report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_labels, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        #print()'''
    
    #=========================================================================================

    
    #clf=SVR(kernel = 'linear', C=10.0)#,class_weight='balanced')
    #clf = RandomForestRegressor(n_estimators = 100)
    #clf = ElasticNetCV(cv=10)
    #clf = RANSACRegressor()
    clf = MLPRegressor(max_iter=1000000000)
    #clf = AdaBoostRegressor(base_estimator='RandomForestRaRegressor', n_estimators=1000000,learning_rate = 0.1)
    #clf = GradientBoostingRegressor()
    #clf = linear_model.SGDRegressor()
    #clf = linear_model.LinearRegression()
    #clf = linear_model.ElasticNet()
    
    '''pca = PCA(n_components=4)# adjust yourself
    pca.fit(train_features)
    
    train_red_features = pca.transform(train_features)
    test_red_features = pca.transform(test_features)'''
    
    #clf.fit(X,y)
    clf.fit(train_features,train_labels)
    pred=clf.predict(test_features)

    #score=f1_score(test_labels,pred,average=None)
    acc_score = clf.score(test_features, pred)
    r2_score = metrics.r2_score(test_labels,pred)
    MAE_score = metrics.mean_absolute_error(test_labels,pred)
    MSE = metrics.mean_squared_error(test_labels, pred)

    #print 'f1-score: ',score
    print ('model-accuracy-score', acc_score)
    print ('r2-score: ',r2_score)
    print ('Mean Absolute Error score: ',MAE_score)
    print ('Mean Squared Error score: ', MSE)

    for i in range(0, pred.shape[0]):
        print (test_labels[i],pred[i])
    return acc_score

SVR_train(5)

'''
avg_acc = 0.0
for i in range(100):
    curr_acc = SVR_train(3.0+(i/100.0))
    avg_acc = avg_acc + curr_acc
    
avg_acc = avg_acc/100.0

print avg_acc
'''    

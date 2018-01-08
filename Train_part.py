import pandas as pd
import numpy as np
import nltk
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import math



#import pandas.io.data
data = pd.DataFrame()

def dataRead():
    df = pd.read_csv('intracity_fare_train.csv',skipinitialspace=True)
    #print df.head()

    null_rows = df[df.isnull().any(axis=1)]# All the rows that have a null value

    null_cols = df.columns[df.isnull().any()].tolist()# Alll the columns that have a null value
    
    

##    print 'Null_rows'
##    print null_rows

    #print 'Null_columns'
    #print null_cols
    #print type(df['site'][3])

    return df
    
def dataAnalysis():
    df = dataRead()
    df['VEHICLE_TYPE'] = map(lambda x: x.upper(), df['VEHICLE_TYPE'])
    
    '''we set a working frame'''
    wf = df
    print wf.shape
    
    '''we start with identifying the columns in which data is missing'''
    null_cols = df.columns[df.isnull().any()].tolist()# Alll the columns that have a null value
    all_cols = df.columns
    
    print "Null_Columns:"
    print null_cols
    
    print "All Columns:"
    print all_cols
    
    '''Next we need to calculate the perccentage of dataset that has missing data points'''
    
    total_cells = df.shape[0]*df.shape[1]
    n_empty_cells = df.isnull().values.ravel().sum()   #total number of empty cells
    
    n_rows = df.shape[0]
    n_empty_rows = df.shape[0] - wf.dropna().shape[0]
    
    percentage_null_cells = float(n_empty_cells)/total_cells
    percentage_null_rows = float(n_empty_rows)/n_rows
    
    #print n_empty_cells
    #print n_empty_rows
    print '% of empty rows: ',percentage_null_rows
    print '% of empty cells: ',percentage_null_cells
    
    
    
    '''#Next we try to identify if there is any relationship between missing data values and any other feature
    #We do this by investigating the values of non-missing features and features with missing values'''
    
    tot_vehicle_types = df.VEHICLE_TYPE.unique().tolist()
    
    #Vehicle-Type and Wait-Time
    index = df['WAIT_TIME'].index[df['WAIT_TIME'].apply(np.isnan)]
    df_index = df.index.values.tolist()
    wait_time_idx = [df_index.index(i) for i in index]
    print set([df['VEHICLE_TYPE'][i] for i in wait_time_idx])
    print tot_vehicle_types
    
    #vehicle-type and total-luggage weight
    #---COPY CODE FROM ABOVE--
    
    #Find correlation b/w wait-time and fare and total-luggage-weight and fare
    #---WRITE CODE--
    
    
    #This code is used to see correlations b/w the columns in which values were missing and fare, at the points where the values were present
    '''temp_f = wf.dropna()
    sns.pairplot(temp_f, hue='VEHICLE_TYPE', x_vars = 'FARE',y_vars = ['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT'], kind = 'reg')
    plt.show()
    #sns_plot = sns.pairplot(temp_f, hue='VEHICLE_TYPE', x_vars = 'FARE',y_vars = ['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT'], kind = 'reg')
    #sns_plot.savefig('output.png')
    sns.pairplot(temp_f,vars = ['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT','FARE'], kind = 'reg')
    plt.show()
    sns.pairplot(temp_f,hue = 'VEHICLE_TYPE',vars = ['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT','FARE'], kind = 'reg')
    plt.show()'''
    
    #This was used only for checking. Hence, not needed anymore
    '''
    print 'Now We check if wait-time and total-luggage-weight go null simultaneously'
    for index, row in df.iterrows():
        if math.isnan(row['WAIT_TIME']) and not math.isnan(row['TOTAL_LUGGAGE_WEIGHT']):
            print row["WAIT_TIME"], row["TOTAL_LUGGAGE_WEIGHT"]
            '''
        
    
    #Now we remove the wait-time and total-luggage-weight and check for percentage of missing data points
    wf2 = df.drop(['WAIT_TIME','TOTAL_LUGGAGE_WEIGHT'],axis=1)
    print 'New workframe after deleting waait-time and total-luggage-weight'
    print wf2.shape
    
    wf2_n_rows = wf2.shape[0]
    wf2_n_empty_rows = wf2.shape[0] - wf2.dropna().shape[0]
    wf2_missing_row_ratio = float(wf2_n_empty_rows)/wf2_n_rows
    
    print 'New percentage of empty rows : ', wf2_missing_row_ratio
    
     
    #Now we pairplot the remaining data and visualize the relations
    
    '''FDA_plot1 = sns.pairplot(wf2.dropna(),hue='VEHICLE_TYPE',vars=['ID','STARTING_LATITUDE','STARTING_LONGITUDE','DESTINATION_LATITUDE', 'DESTINATION_LONGITUDE','FARE'])#Do it in 2 groups
    FDA_plot1.savefig('FDAplot1.png')
    plt.show()
    
    
    FDA_plot2 = sns.pairplot(wf2.dropna(),hue='VEHICLE_TYPE',vars=['TRAFFIC_STUCK_TIME','DISTANCE', 'FARE'])
    FDA_plot2.savefig('FDAplot2.png')
    plt.show()'''
    
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
    print labels

    ratio=ratio/10.0
    
    #return split(features, labels, ratio)
    return train_test_split(features, labels, test_size = ratio)

def SVR_train(ratio):
    
    train_features,test_features,train_labels,test_labels = TrainTest(ratio)
     
    
    from sklearn.svm import SVR
    from sklearn.metrics import f1_score
    clf=SVR(kernel = 'poly', degree= 4, C=1.0)#,class_weight='balanced')

    train_labels=(train_labels.ravel())
    test_labels=(test_labels.ravel())

    clf.fit(train_features,train_labels)
    pred=clf.predict(test_features)

    #score=f1_score(test_labels,pred,average=None)
    acc_score = clf.score(test_features, pred)

    #print 'f1-score: ',score
    print 'accuracy-score', acc_score

    for i in range(0, pred.shape[0]):
        print test_labels[i],pred[i]
    return acc_score

SVR_train(3)

'''
avg_acc = 0.0
for i in range(100):
    curr_acc = SVR_train(3.0+(i/100.0))
    avg_acc = avg_acc + curr_acc
    
avg_acc = avg_acc/100.0

print avg_acc
'''    

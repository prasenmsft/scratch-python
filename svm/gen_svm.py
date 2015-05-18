import numpy as np 
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn import svm
import random

def dict_vectorize(features_df):
	#vect = TfidfVectorizer()
	#X = vect.fit_transform(df['tweets'])	
	#y = df['class']
	#cols = df.columns
	#cols_to_retain = list(cols[:-1])# last col is drop/label
	#features_df = df[cols_to_retain]
	features_dict = features_df.T.to_dict().values()
	
	vect = DV( sparse = False )	
	X = vect.fit_transform(features_dict)
	return X

def labelencoder_vectorize(features_df, encoders = None):
    
    encoders_is_passed = True
    if encoders is None: 
       encoders = {}
       encoders_is_passed = False 
    for column in features_df.columns:
        #_ = encoder.fit(features_df[column])
        if encoders_is_passed and column not in encoders: 
          print(column+" not in encoder")
          exit(0)  
        if column not in encoders: encoders[column]= LabelEncoder()
		
        features_df[column] = encoders[column].fit_transform(features_df[column])
        
    return (features_df, encoders)


def main():
	#dtype_={'names': names, 'formats': formats}
	#f = open('StudentData.csv')
	#np.loadtxt(f, dtype=dtype_, delimiter='\t')
	
	df = pd.read_csv('StudentData.csv', delimiter='\t', dtype=object)
	df = df.astype(object).fillna('0')
	#df.dtypes
	#df.index
	#df.columns
	
	cols_to_drop = ['NAME', 'ROLLNO2', 'REC2', 'FNAME', 'MNAME', 'DROP']
	#cols_to_drop = ['NAME', 'REC2', 'FNAME', 'MNAME', 'DROP']
	
	features_df = df.drop(cols_to_drop, axis =1 )
	#X = dict_vectorize(features_df)
	(X_all, encoders) = labelencoder_vectorize(features_df)
	training_rows = random.sample(list(df.index), 10000)
	X = X_all.ix[training_rows]
	
	print("SEX: "+str(encoders['SEX'].classes_)+" "+str(encoders['SEX'].transform(["B"])))
	print("RELIGION: "+str(encoders['RELIGION'].classes_)+" "+str(encoders['RELIGION'].transform(["H"])))
	print("URB_RUR: "+str(encoders['URB_RUR'].classes_)+" "+str(encoders['URB_RUR'].transform(["U"])))
	print("MED_DESC: "+str(encoders['MED_DESC'].classes_)+" "+str(encoders['MED_DESC'].transform(["TELUGU"])))
	print("COMM_DESC: "+str(encoders['COMM_DESC'].classes_))
	
	y = df.ix[training_rows]['DROP']
	print('Training Set')
	#df.iloc[0] # integer based, gives the first row
	#df.loc[10] # label based, gives the row with label 10
	for i in training_rows:
		#print(str(i)+" "+str(y[i]+" <><> "+str(X.loc[i])))
		#print(str(i)+" "+str(y[i]))
		pass
	
	clf = svm.SVC(kernel='rbf', probability=True)
	#features = [[0, 0], [1, 1]] # [n_samples, n_features]
	#labels = [0, 1] # [n_samples]
	clf = clf.fit(X, y)
	print('Training Done')
	
	testing_rows = random.sample(list(df.index), 5)
	test_df = df.ix[testing_rows]
	test_drop_df = test_df[['DROP', 'ROLLNO2']]
	test_df = test_df.drop(cols_to_drop, axis =1)
	test_df_orig = test_df.copy(deep=True)
	#(test_X, _) = labelencoder_vectorize(X.ix[testing_rows],encoders)
	print("Now priniting testing samples")
	print(test_df_orig)
	print("Now priniting testing samples ( with transformation)")
	test_X = X_all.ix[testing_rows]
	print(test_X)
	#print("Now priniting few traning samples")
	#print(X.iloc[:3])
	res = clf.predict_proba(test_X)
	print(res)
	res = clf.predict(test_X)
	print(res)
	print(test_drop_df)
	
	#clf.predict([[2., 2.]])
	

main()
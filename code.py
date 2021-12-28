# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import json
with open('train-1.json') as json_file:
    traindf = pd.DataFrame(json.load(json_file))

#SPLITTING THE VENUE AND KEEPING THE FIRST ONE IN A NEW COLUMN
full = []
cleanven = []
for i in traindf['venue']: #saving only the first string before the separators to a list, assuming that the first venue is the most important/relevant
    if '*' in i: #removing the star
        i = i.replace('*', '')
    if '@' in i:
        i = i.split('@')
        full.append(i[0])
    elif '/' in i:
        i = i.split('/')
        full.append(i[0])
    elif ' ' in i:
        i = i.split(' ')
        full.append(i[0])
    elif '-' in i:
        i = i.split('-')
        full.append(i[0])
    else:
        full.append(i)
        
for i in full: #some can have more separators, so re-running it on the list again to be sure
    if '@' in i:
        i = i.split('@')
        cleanven.append(i[0])
    elif '/' in i:
        i = i.split('/')
        cleanven.append(i[0])
    elif ' ' in i:
        i = i.split(' ')
        cleanven.append(i[0])
    elif '-' in i:
        i = i.split('-')
        cleanven.append(i[0])
    else:
        cleanven.append(i)
traindf['cleanven'] = cleanven #adding it as a new col to the df

#CHANGING OPEN ACCESS TO DUMMY
open_access_dummy = []
for i in traindf['is_open_access']: #changing bool to int (0/1)
  if i == True:
    open_access_dummy.append(1)
  else:
    open_access_dummy.append(0)
traindf['open_access_dummy'] = open_access_dummy #adding it as a new col as well

#ADDING NUMBER OF AUTHORS
authornum = []
for i in traindf['authors']:
  authornum.append(len(i))
traindf['author_num'] = authornum

#ADDING WORDNUM FOR THE TITLE
title_wordnum = []
for i in traindf['title']:
  if i is None:
    title_wordnum.append('')
  else:
    splitlist = i.split(' ')
    title_wordnum.append(len(splitlist))    
traindf['title_wordnum'] = title_wordnum

#ADDING WORDNUM FOR THE ABSTRACT
abstract_wordnum = []
for i in traindf['abstract']:
  if i is None:
    abstract_wordnum.append(0) #first append 0
  else:
    splitlist = i.split(' ')
    abstract_wordnum.append(len(splitlist))
#CHANGING 0 TO MEAN
abstract_num = []
abstr_mean = int(sum(abstract_wordnum) / len(abstract_wordnum))
for num in abstract_wordnum:
  if num > 0:
    abstract_num.append(num)
  else:
    abstract_num.append(abstr_mean)
traindf['abstract_wordnum'] = abstract_num

#ADDING COLUMN WITH YES 1 AND NO 0 IF THE TITLE HAS PUNCTUATION
title_haspunc = []
punctuation = ['.', '!', ':', ',', ';', '?']
for i in traindf['title']:
  if i is None:
    title_haspunc.append(0)
  else:
    if any(punc in i for punc in punctuation):
      title_haspunc.append(1)
    else:
      title_haspunc.append(0)
traindf['title_haspunc'] = title_haspunc

#ADDING FIELD OF STUDY BASED ON ABS
count = 0
for i in zip(traindf["abstract"], traindf["fields_of_study"]): #check if and how many papers have missing values for both the abstract and field of study
    if i[0] == None and i[1] == None:
        count += 1

fields = [] #create list of all possible fields of study
for i in traindf["fields_of_study"]:
    if i == None:
      continue
    if len(i) > 1:
      j = 0
      while j < len(i):
          fields.append([i[j]])
          j += 1
    else:
        fields.append(i)

list_fields = []       
for x in fields:
    for item in x: 
        list_fields.append(item)
flatlist_fields = set(list_fields)

for idx,val in enumerate(zip(traindf["fields_of_study"], traindf["abstract"])): 
    if val[0] == None: #add missing fields of study of papers for which the field of study is mentioned in the abstract
        for h in flatlist_fields:
            if val[1] == None:
                continue
            elif h in val[1]:
                traindf.at[idx, 'fields_of_study'] = [h]

#filling in missing field of study with mode
for idx,val in enumerate(traindf["fields_of_study"]):
    if val == None:
      h = traindf['fields_of_study'].mode()[0]
      traindf.at[idx, 'fields_of_study'] = h
    else:
      continue

#Filling in missing values for year with mode
yearmean = traindf['year'].mode()[0]
traindf['year'].fillna(float(yearmean), inplace=True) # Clean NaN in 'year'

#SPLIT FIELDS_OF_STUDY
dffield = pd.DataFrame(traindf.fields_of_study.values.tolist()) #make separate dataset for fields_of_study to split them (https://stackoverflow.com/questions/44663903/pandas-split-column-of-lists-of-unequal-length-into-multiple-columns)
dffield.fillna("",inplace=True) #replace missing values with "" in dffield (https://stackoverflow.com/questions/31295740/how-to-replace-none-only-with-empty-string-using-pandas)
dffield.rename(columns={0: 'field1', 1:'field2'}, inplace=True) #rename columns in dffield
traindf = pd.concat([traindf, dffield], axis=1) #join dffield and traindf (https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)

#SPLIT TOPICS AND ADD FIRST 3 TO TRAINDF
dftopics = pd.DataFrame(traindf.topics.values.tolist()) #make separate dataset for topics to split them (https://stackoverflow.com/questions/44663903/pandas-split-column-of-lists-of-unequal-length-into-multiple-columns
traindf = pd.concat([traindf, dftopics[[0, 1]]], axis=1) #join dftopics and traindf (https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
traindf.rename(columns={0: 'topic1', 1: 'topic2'}, inplace=True) #rename new columns

# ENCODING SOME STRING VALUES
codes, uniques = pd.factorize(traindf['cleanven'])
traindf['cleanven'] = codes

#ENCODING THE FIELDS
stacked = traindf.loc[:, 'field1':'field2'].stack() # https://stackoverflow.com/questions/39390160/pandas-factorize-on-an-entire-data-frame
traindf.loc[:, 'field1':'field2'] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()

# DROPPING SOME FEATURES FOR NOW
traindf.drop(['title', 'abstract', 'authors', 'venue', 'topics', 'fields_of_study','topic1', 'topic2'], axis=1, inplace = True)

#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(traindf[['year', 'references', 'author_num', 'title_wordnum', 'is_open_access']], 
                                                   traindf['citations'],
                                                  test_size=0.2, 
                                                  random_state=75)

from sklearn.ensemble import GradientBoostingRegressor 
y_train = [np.log1p(i) for i in y_train] #Logtransform

model = GradientBoostingRegressor(n_estimators = 200,
                                  learning_rate = 0.2, 
                                  random_state = 999,
                                  loss = 'absolute_error',
                                  min_samples_split = 6, 
                                  )  
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = [np.expm1(i) for i in y_pred] #Logtrans back

#from sklearn.model_selection import RandomizedSearchCV #https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models

#X=traindf[['year', 'references', 'open_access_dummy', 'author_num', 'title_wordnum']]                                                  
#y= traindf['citations']
#n_estimators = [600, 400, 200]
#max_depth = [3, 5, 10]
#loss = ['absolute_error', 'squared_error', 'huber']
#min_samples_split = [2, 4, 6]
#param_grid = dict(n_estimators = n_estimators, max_depth=max_depth, loss = loss, min_samples_split = min_samples_split)

#modi = GradientBoostingRegressor()
#grid = RandomizedSearchCV(estimator=modi, param_distributions=param_grid, cv = 3, n_jobs=-1)

#grid_result = grid.fit(X, y)

# Summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def score(Y_true, Y_pred):
  y_true = np.log1p(np.maximum(0, Y_true))
  y_pred = np.log1p(np.maximum(0, Y_pred))
  return 1 - np.mean((y_true-y_pred)**2) / np.mean((y_true-np.mean(y_true))**2)

#EVALUATION ON THE TRAINSET
x_testi = model.predict(X_train)
x_testi = [np.expm1(i) for i in x_testi]
r2 = r2_score( x_testi, y_train )      
MAE = mean_absolute_error( x_testi, y_train )

print( 'r2 = {}'.format( r2 ))
print( 'MAE = {}'.format( MAE ))

print(score(x_testi, y_train))

#EVALUATION
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score( y_test, y_pred )      # Evaluation metrics
MAE = mean_absolute_error( y_test, y_pred )

print( 'r2 = {}'.format( r2 ))
print( 'MAE = {}'.format( MAE ))

print(score(y_test, y_pred))

#Predicting on the test set
with open('test.json') as json_file:
    testdf = pd.DataFrame(json.load(json_file))

#SPLITTING THE VENUE AND KEEPING THE FIRST ONE IN A NEW COLUMN
full = []
cleanven = []
for i in testdf['venue']: #saving only the first string before the separators to a list, assuming that the first venue is the most important/relevant
    if '*' in i: #removing the star
        i = i.replace('*', '')
    if '@' in i:
        i = i.split('@')
        full.append(i[0])
    elif '/' in i:
        i = i.split('/')
        full.append(i[0])
    elif ' ' in i:
        i = i.split(' ')
        full.append(i[0])
    elif '-' in i:
        i = i.split('-')
        full.append(i[0])
    else:
        full.append(i)

for i in full: #some can have more separators, so re-running it on the list again to be sure
    if '@' in i:
        i = i.split('@')
        cleanven.append(i[0])
    elif '/' in i:
        i = i.split('/')
        cleanven.append(i[0])
    elif ' ' in i:
        i = i.split(' ')
        cleanven.append(i[0])
    elif '-' in i:
        i = i.split('-')
        cleanven.append(i[0])
    else:
        cleanven.append(i)
testdf['cleanven'] = cleanven #adding it as a new col to the df

#CHANGING OPEN ACCESS TO DUMMY
open_access_dummy = []
for i in testdf['is_open_access']: #changing bool to int (0/1)
  if i == True:
    open_access_dummy.append(1)
  else:
    open_access_dummy.append(0)
testdf['open_access_dummy'] = open_access_dummy #adding it as a new col as well

#ADDING NUMBER OF AUTHORS
authornum = []
for i in testdf['authors']:
  authornum.append(len(i))
testdf['author_num'] = authornum

#ADDING WORDNUM FOR THE TITLE
title_wordnum = []
for i in testdf['title']:
  if i is None:
    title_wordnum.append('')
  else:
    splitlist = i.split(' ')
    title_wordnum.append(len(splitlist))    
testdf['title_wordnum'] = title_wordnum

#ADDING WORDNUM FOR THE ABSTRACT
abstract_wordnum = []
for i in testdf['abstract']:
  if i is None:
    abstract_wordnum.append(0) #first append 0
  else:
    splitlist = i.split(' ')
    abstract_wordnum.append(len(splitlist))
#CHANGING 0 TO MEAN
abstract_num = []
abstr_mean = int(sum(abstract_wordnum) / len(abstract_wordnum))
for num in abstract_wordnum:
  if num > 0:
    abstract_num.append(num)
  else:
    abstract_num.append(abstr_mean)
testdf['abstract_wordnum'] = abstract_num

#ADDING COLUMN WITH YES 1 AND NO 0 IF THE TITLE HAS PUNCTUATION
title_haspunc = []
punctuation = ['.', '!', ':', ',', ';', '?']
for i in testdf['title']:
  if i is None:
    title_haspunc.append(0)
  else:
    if any(punc in i for punc in punctuation):
      title_haspunc.append(1)
    else:
      title_haspunc.append(0)
testdf['title_haspunc'] = title_haspunc

#ADDING FIELD OF STUDY BASED ON ABS
count = 0
for i in zip(testdf["abstract"], testdf["fields_of_study"]): #check if and how many papers have missing values for both the abstract and field of study
    if i[0] == None and i[1] == None:
        count += 1
fields = [] #create list of all possible fields of study
for i in testdf["fields_of_study"]:
    if i == None:
      continue
    if len(i) > 1:
      j = 0
      while j < len(i):
          fields.append([i[j]])
          j += 1
    else:
        fields.append(i)
list_fields = []       
for x in fields:
    for item in x: 
        list_fields.append(item)
flatlist_fields = set(list_fields)

for idx,val in enumerate(zip(testdf["fields_of_study"], testdf["abstract"])): 
    if val[0] == None: #add missing fields of study of papers for which the field of study is mentioned in the abstract
        for h in flatlist_fields:
            if val[1] == None:
                continue
            elif h in val[1]:
                testdf.at[idx, 'fields_of_study'] = [h]

#filling in missing field of study with mode
for idx,val in enumerate(testdf["fields_of_study"]):
    if val == None:
      h = testdf['fields_of_study'].mode()[0]
      testdf.at[idx, 'fields_of_study'] = h
    else:
      continue

#Filling in missing values for year with mode
yearmean = testdf['year'].mode()[0]
testdf['year'].fillna(float(yearmean), inplace=True) # Clean NaN in 'year'

#SPLIT FIELDS_OF_STUDY

dffield = pd.DataFrame(testdf.fields_of_study.values.tolist()) #make separate dataset for fields_of_study to split them (https://stackoverflow.com/questions/44663903/pandas-split-column-of-lists-of-unequal-length-into-multiple-columns)
dffield.fillna("",inplace=True) #replace missing values with "" in dffield (https://stackoverflow.com/questions/31295740/how-to-replace-none-only-with-empty-string-using-pandas)
dffield.rename(columns={0: 'field1', 1:'field2'}, inplace=True) #rename columns in dffield
testdf = pd.concat([testdf, dffield], axis=1) #join dffield and traindf (https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)

#SPLIT TOPICS AND ADD FIRST 3 TO TRAINDF
dftopics = pd.DataFrame(testdf.topics.values.tolist()) #make separate dataset for topics to split them (https://stackoverflow.com/questions/44663903/pandas-split-column-of-lists-of-unequal-length-into-multiple-columns
testdf = pd.concat([testdf, dftopics[[0, 1]]], axis=1) ##join dftopics and traindf (https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
testdf.rename(columns={0: 'topic1', 1: 'topic2'}, inplace=True) #rename new columns

# ENCODING SOME STRING VALUES
codes, uniques = pd.factorize(testdf['cleanven'])
testdf['cleanven'] = codes

#ENCODING THE FIELDS
stacked = testdf.loc[:, 'field1':'field2'].stack() # https://stackoverflow.com/questions/39390160/pandas-factorize-on-an-entire-data-frame
testdf.loc[:, 'field1':'field2'] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()

# DROPPING SOME FEATURES FOR NOW
testdf.drop(['title', 'abstract', 'authors', 'venue', 'topics', 'fields_of_study', 'topic1', 'topic2'], axis=1, inplace = True)

#PART FOR ONLY THE TEST SET - PREDICTION
X_test = testdf[['year', 'references', 'author_num', 'title_wordnum', 'is_open_access']]

y_pred = model.predict(X_test) #make prediction easily
y_pred = [np.expm1(i) for i in y_pred] #Logtransform this back as well

#NEW DF WITH DOI AND PRED
preddoi = testdf['doi'] 
preddf = pd.DataFrame()
preddf['doi'] = preddoi
preddf['citations'] = y_pred

#MAKING A DICTIONARY TO EASY EXPORT
preddict = preddf.to_dict('records')

#SAVING THE RESULTS AS JSON
with open("predicted.json", "w") as outfile:
    json.dump(preddict, outfile, sort_keys=False)
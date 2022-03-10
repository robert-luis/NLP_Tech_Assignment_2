#!/usr/bin/env python
# coding: utf-8

# # Argument Classification using SVM
# 
# This script provides a function to execute the argument classification using SVM <br>
# 
# 
# *Input:*  
# - executionMode_dict
# - mode               -> ('production' / 'sample')
# - print_status       -> (True / False)
# 
# *Output:* 
# - executionMode_dict 
# - (end dataframe incl. results for testing)




import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import sys





def classifyArguments(executionMode_dict,
                      mode,                   #('production' / 'sample')
                      print_status   = False):

    
    # assignments
    path_to_input_train = executionMode_dict[mode]['intermediate']['train']['05_identifiedArguments']
    path_to_input_test  = executionMode_dict[mode]['intermediate']['test']['05_identifiedArguments']
    
    path_to_save_df     = '../data/output/' + mode + '_06_predictedArguments.csv'
    executionMode_dict[mode]['output']['06_predictedArguments'] = path_to_save_df

    path_to_save_report = '../data/output/' + mode + '_classifiedArgumentsReport.csv'
    executionMode_dict[mode]['output']['classifiedArgumentsReport'] = path_to_save_report
    
    #path_to_save_model       = '../data/intermediate/' + mode + '_' + model +'_predictionModel'

    # Read tsv file using pandas and turn it into a dataframe
    # Read in dev / test set depeding on argument provided when running the python file

    #train_df 
    train_df = pd.read_csv('../data/intermediate/sample_train_05_identifiedArguments.csv') 
    train_df = train_df.fillna('X')
    train_df = train_df[train_df['label_ident_prediction'] == True]

    #test_df
    test_df = pd.read_csv('../data/intermediate/sample_test_05_identifiedArguments.csv')
    test_df = test_df.fillna('X')
    test_df = test_df[test_df['label_ident_prediction'] == True]

    train_instances = train_df[["sentenceId", "sentenceRepetition", "id", "form", "lemma", "upos", "xpos", "morph", "head", "dep", "head_dep", "space", "predicate_prediction", "label_ident_prediction", "passive", "full_constituent"]].to_dict('records')
    test_instances  = test_df[["sentenceId", "sentenceRepetition", "id", "form", "lemma", "upos", "xpos", "morph", "head", "dep", "head_dep", "space", "predicate_prediction", "label_ident_prediction", "passive", "full_constituent"]].to_dict('records')

    vec = DictVectorizer()
    X_train = vec.fit_transform(train_instances)

    Y_train = train_df.label_gold.tolist()
    Y_test = test_df.label_gold.tolist()

    classifier = LinearSVC(max_iter = 10000)

    parameters = dict(C=(0.01, 0.1, 1.0), loss=('hinge', 'squared_hinge'), tol=(0.0001,0.001,0.01,0.1))

    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5, scoring='f1_macro')
    grid.fit(X_train, Y_train)
    classifier = grid.best_estimator_
    X_test = vec.transform(test_instances)
    predictions = classifier.predict(X_test)

    # test dataframe
    test_df['predictions'] = predictions
    test_df.to_csv(path_to_save_df, index=False)
    
    # report
    report = pd.DataFrame(classification_report(y_true=test_df['label_gold'], y_pred=test_df['predictions'], output_dict=True)).transpose()
    report.to_csv(path_to_save_report, index=True)
    
    if print_status == True:    
        print(' - completed')
    
    
    return executionMode_dict


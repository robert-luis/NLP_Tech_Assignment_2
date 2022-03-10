#!/usr/bin/env python
# coding: utf-8

# # 2 Predicate Prediction
# 
# This script provides a function predict the predicates 
# 
# *Input*:
# 
# executionMode_dict
# mode -> ('production' / 'sample')
# model -> ('train' / 'test')
# print_status -> (True / False)
# sentence_limit = None (limit of sentences to import (default: None)
# 
# 
# *Output*:
# 
# executionMode_dict

# ## Preparation

# In[ ]:


import pandas as pd


# ## Reading data in

# In[ ]:



def predictPredicates(executionMode_dict,
                      mode,                   #('production' / 'sample')
                      model,                  #('train' / 'test')
                      print_status   = False,
                      sentence_limit = None):
    

    path_to_input = executionMode_dict[mode]['intermediate'][model]['1_imported']
    path_to_save = '../data/intermediate/' + mode + '_' + model +'_02_predictedPredicates.csv'
    executionMode_dict[mode]['intermediate'][model]['2_predicatesPredicted'] = path_to_save
    
    if print_status == True:
        print('\n\n#### 2 Predicate Prediction ####')
    
    # read dataframe in
    df = pd.read_csv(path_to_input)
    upos_list = df['upos'].tolist()
    
    # insert predicted predicates here
    predicate_list = []
    for upos in upos_list:
        if upos == 'VERB' or upos == 'AUX':
            predicate_list.append(True)
        else:
            predicate_list.append(False)
    
    df['predicate_prediction'] = predicate_list

    
    #write dataframe out
    df.to_csv(path_to_save, index=False)
    
    
    if print_status == True:
        print(' - completed')
    
    return executionMode_dict


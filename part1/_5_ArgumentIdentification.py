#!/usr/bin/env python
# coding: utf-8

# # 5 Argument Identification
# 
# This script provides a function to identify arguments per predicate <br>
# 
# 
# *Input:*  
# - executionMode_dict
# - mode               -> ('production' / 'sample')
# - model              -> ('train' / 'test')
# - print_status       -> (True / False)
# - sentence_limit = None  (limit of sentences to import (default: None)
# 
# *Output:* 
# - executionMode_dict 
# 

# ## Preparation

# In[ ]:


import pandas as pd
import numpy  as np


# ## Reading data in

# In[ ]:


def identifyArguments(executionMode_dict,
                      mode,                   #('production' / 'sample')
                      model,                  #('train' / 'test')
                      print_status   = False,
                      sentence_limit = None):
    

    path_to_input = executionMode_dict[mode]['intermediate'][model]['04_FeaturesExtracted']
    path_to_save = '../data/intermediate/' + mode + '_' + model +'_05_identifiedArguments.csv'
    executionMode_dict[mode]['intermediate'][model]['05_identifiedArguments'] = path_to_save
    
    
    if print_status == True:
        print('\n\n#### 5 Argument Identification ####')
    
    # read dataframe in
    df = pd.read_csv(path_to_input)
    
    
    ## variable
    #reference column for predicates :  prediction vs truth
    predicate_column = 'predicate_prediction'
    #predicate_column = 'predicate_gold'



    # loop through sentences
    for s_id in df.sentenceId.unique():

        # filter for only this sentence
        df_sentence = df[df.sentenceId == s_id]

        # initiate np array to insert preditioncs
        predicates = np.full(len(df_sentence.id.unique()), False)

        ### find all predicates of that sentence 

        # loop through each repetition
        for s_rep in df_sentence.sentenceRepetition.unique():

            # create new subframe for working within this repetition of sentence
            df_sentence_repetition = df_sentence[df_sentence.sentenceRepetition == s_rep]


            ###  identify arguments (their indices)

            ## 1. identify id of predicate
            predicate_identification = np.where(df_sentence_repetition[predicate_column] == True)[0]
            if len(predicate_identification) == 1:
                index_of_pred = np.where(df_sentence_repetition[predicate_column] == True)[0][0]
                #index_of_pred

                predicates[index_of_pred] = True


        # loop through each repetition
        for s_rep in df_sentence.sentenceRepetition.unique():

            # create new subframe for working within this repetition of sentence
            df_sentence_repetition = df_sentence[df_sentence.sentenceRepetition == s_rep]

            # initiate np array to insert preditioncs
            pred = np.full(len(df_sentence_repetition), False)


            ###  identify arguments (their indices)


            ## 1. identify id of predicate
            temp = df_sentence_repetition[df_sentence_repetition[predicate_column] == True].id

            if len(temp) == 1:
                #print('if true')
                predicate_id = int(temp)
                #predicate_id = int(df_sentence_repetition[df_sentence_repetition[predicate_column] == True].id)

                ## RULE 1
                # ->take subset which have predicate id as head

                pred = np.array(df_sentence_repetition['head'] == predicate_id)

                ## RULE 2
                # -> exclude punctuation
                #punct_index = df_sentence_repetition[df_sentence_repetition.dep == 'punct'].id - 1
                #pred[punct_index] = False
                
                punct_indeces = [df_sentence_repetition[df_sentence_repetition.dep == 'punct'].id]
                for punct_index in np.array(punct_indeces[0]):
                #    #converting id to index vio substracting 1
                    pred[int(punct_index)-1] = False

                ## RULE 3
                # -> exclude all the identified predicates from arguments
                for i in range(len(pred)):
                    # if this token is a predicate
                    if predicates[i] == True:
                        pred[i] = False


            # assign value to all features
            #df_sentence_repetition.label_ident_prediction = pred
            df.iloc[min(df_sentence_repetition.index):max(df_sentence_repetition.index)+1,df.columns.get_loc('label_ident_prediction')] = pred    
    
    
    #write dataframe out
    df.to_csv(path_to_save, index=False)
    
    
    if print_status == True:
        print(' - completed')
    
    return executionMode_dict


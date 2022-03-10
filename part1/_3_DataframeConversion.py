#!/usr/bin/env python
# coding: utf-8

# # 3 Dataframe Conversion
# 
# This script provides a function that converts the dataframe in a sense that each sentence is duplicated as often as the maximum of predicted or actual predicates per sentence <br>
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
import numpy as np


# ## Conversion Function

# In[ ]:


### helper functions for transformation of dataframe

# helper function to combine predicate arrays
# input value of predicate_gold and predicate_predicted. if either one is true, return true
# -> applied via lambda function to each row in respective dataframe containing one sentence
# -> goal is to have a boolean array that dictates the amount of needed repetitions of sentence  
#    and at which index to look for predicate
def findPredicateUnion(predicateGold, predicatePredicted):
    if predicateGold != '_' or predicatePredicted == True:
        return True
    else:
        return False


# In[1]:



### function to retrieve arguments

def convertDataframe(executionMode_dict,
                     mode,                   #('production' / 'sample')
                     model,                  #('train' / 'test')
                     print_status   = False,
                     sentence_limit = None):
    
    


    '''
    dataframe structure information:
    
    input:  a dataframe containing the following columns:
                ['sentenceId', 
                 'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 
                 'predicate', 'label', '_', '_', ... '_', 
                 'predicate_prediction']

                 -> note that 
                    - predicate_prediction has to be created beforehand
                    - a variable amount of '_' columns is possible
                    

    output: the expanded dataframe dataframe containing the following columns
                ['sentenceId', 'sentenceRepetition', 
                 'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space',
                 'predicate_prediction', 'label_ident_prediction', 'label_prediction',
                 'predicate_gold', 'label_ident_gold', 'label_gold']

    '''
    

    path_to_input = executionMode_dict[mode]['intermediate'][model]['2_predicatesPredicted']
    path_to_save = '../data/intermediate/' + mode + '_' + model +'_03_convertedDataframe.csv'
    executionMode_dict[mode]['intermediate'][model]['03_convertedDataframe'] = path_to_save
    
    
    if print_status == True:    
        print('\n\n#### 3 Data Conversion ####')
    
    
    # read dataframe in
    df = pd.read_csv(path_to_input)
    
    

    
    # taken from 01_dataImport
    conll_header_adapted = ['sentenceId', 'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 'predicate', 'label']
    

    ## prepare a dataframe to store all conversions in
    # basic features
    df_expanded = pd.DataFrame(columns=conll_header_adapted[:-2])
    # + these four additional columns, we want to add
    df_expanded['predicate_gold']       = False #np.nan
    df_expanded['label_gold']           = '_'   #np.nan
    df_expanded['predicate_prediction'] = False #np.nan
    df_expanded['sentenceRepetition']   = 0

    df_expanded_columns = df_expanded.columns

    ## do conversion

    # loop through sentences
    for s_id in df.sentenceId.unique():

        # filter for only this sentence
        df_sentence = df[df.sentenceId == s_id].copy()   # remove hardcoing of sentence 2 (equivalent to index 1) as example

        # count rows for which predicate_gold is true (actually != '_') OR predicate_predicted is true
        df_sentence['union_predicates_gold_predicted'] = df.apply(lambda x: findPredicateUnion(x.predicate, x.predicate_prediction), axis=1)


        # return indices of rows with label True of the columns of the predicates
        indices_union     = np.where(np.array(df_sentence.union_predicates_gold_predicted) == True)[0]
        indices_gold      = np.where(np.array(df_sentence.predicate)                       != '_' )[0]
        indices_predicted = np.where(np.array(df_sentence.predicate_prediction)            == True)[0]


        #nr_of_predicates = df_sentence.union_predicates_gold_predicted[df_sentence.union_predicates_gold_predicted == True].count()
        nr_of_predicates = len(indices_union)


        # loop through nr_of_predicates
        for i in range(nr_of_predicates):


            # create new copy for working with within this repetition of sentence
            df_sentence_repetition = df_sentence.copy()


            ### fill values for new important columns

            # id for repition of sentence to be able to loop through afterwards
            df_sentence_repetition['sentenceRepetition']   = i


            ## predicates

            # fill predicate columns with False as default 
            # -> afterwards only replace that one specific row with True, which we look at in this repitition
            predicate_array_gold   = np.full(len(df_sentence_repetition), False)
            predicate_array_pred   = np.full(len(df_sentence_repetition), False)

            # now replace respective index of predicate columns if it is also in the respective column
            if indices_union[i] in indices_gold:
                predicate_array_gold[indices_union[i]] = True
            if indices_union[i] in indices_predicted:
                predicate_array_pred[indices_union[i]] = True

            # assign created arrays to dataframe
            df_sentence_repetition['predicate_gold']       = predicate_array_gold
            df_sentence_repetition['predicate_prediction'] = predicate_array_pred



            ## labels

            # -> transform labels from all label columns to this one column

            # create filler array
            label_array = np.full(len(df_sentence_repetition), '_')

            # slice df_sentence
            row = df_sentence.iloc[indices_union[i], :]
            list_of_column_indices_with_V = np.where(np.array(row) == 'V')[0]

            # sanity check -> columns found with V should be 1
            if len(list_of_column_indices_with_V) == 1:

                # do conversion

                # find respective_label_column
                respective_column_index = list_of_column_indices_with_V[0]

                # retrieve column
                respective_label_column = np.array(df_sentence.iloc[:, respective_column_index])

                # replave 'V' label with '_'
                respective_label_column[respective_label_column == 'V'] = '_'

                # overwrite filler with retrieved labels
                label_array = respective_label_column

            # label_array remains only filled with '_' because no (coherent) labels could be found
            else:
                pass

            # assign retrieved array
            df_sentence_repetition['label_gold']        = label_array



            ### "postprocessing"

            # drop unneccessary columns 
            # search for all headers with 'label' in it
            l = df_sentence_repetition.columns
            columns_to_drop = list(l[[True if 'label' in x else False for x in l]])
            columns_to_drop.remove('label_gold')
            columns_to_drop.append('predicate')
            columns_to_drop.append('union_predicates_gold_predicted')
            #df_sentence_repetition = df_sentence_repetition.drop(labels=['_', 'label', 'predicate', 'union_predicates_gold_predicted'], axis=1)
            df_sentence_repetition = df_sentence_repetition.drop(labels=columns_to_drop, axis=1)
            
            # concatenate to large dataframe
            df_expanded = pd.concat([df_expanded, df_sentence_repetition], axis = 0, ignore_index=True)




    ### insert general columns for later use

    # for later insert of predicted label in in classification task
    df_expanded['label_prediction']       = np.nan 

    # for prediction of label identification
    df_expanded['label_ident_prediction'] = np.nan

    # gold of label identification (true/false)
    df_expanded['label_ident_gold']       = df_expanded.label_gold.apply(lambda x: True if x != '_' else False)

    #reordering columns
    df_expanded = df_expanded[['sentenceId', 'sentenceRepetition', 
                'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 
                'predicate_prediction', 'label_ident_prediction', 'label_prediction', 
                'predicate_gold',       'label_ident_gold',       'label_gold']]
        
    
    
        
    #write dataframe out
    df_expanded.to_csv(path_to_save, index=False)
    
    if print_status == True:
        
        print(f' - # of lines in dataframe before conversion: {len(df)}')
        print(f' - # of lines in dataframe after conversion: {len(df_expanded)}')
        print( ' - completed')

    return executionMode_dict



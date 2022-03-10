#!/usr/bin/env python
# coding: utf-8

# # 4 Feature Extraction
# 
# This script provides a function that extracts features <br>
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
# 
# *List of additionally extracted features:* 
# - Constituents
# - Active / Passive 
# - ..
# 
# 
# 
# 
# 

# ## Preparation

# In[ ]:


import pandas as pd
import numpy  as np
import spacy
import re


# ## Reading data in

# In[ ]:


def extractFeatures(executionMode_dict,
                    mode,                   #('production' / 'sample')
                    model,                  #('train' / 'test')
                    print_status   = False,
                    sentence_limit = None):
    

    path_to_input = executionMode_dict[mode]['intermediate'][model]['03_convertedDataframe']
    path_to_save = '../data/intermediate/' + mode + '_' + model +'_04_ExtractedFeatures.csv'
    executionMode_dict[mode]['intermediate'][model]['04_FeaturesExtracted'] = path_to_save
    
    
    if print_status == True:
        print('\n\n#### 4 Feature Extraction ####')
    
    
    # read dataframe in
    df = pd.read_csv(path_to_input)
    
    
    df['passive'] = False
    df['full_constituent'] = np.nan
    
    # loop through sentences
    for s_id in df.sentenceId.unique():

        # filter for only this sentence
        df_sentence = df[df.sentenceId == s_id]

        # loop through each repetition
        for s_rep in df_sentence.sentenceRepetition.unique():

            # create new subframe for working within this repetition of sentence
            df_sentence_repetition = df_sentence[df_sentence.sentenceRepetition == s_rep]
            #print(df_sentence_repetition.index)

    # to extract voice of phrase/sentence (a boolean value for passive)
            for (pred, dep, voice) in zip(df_sentence_repetition['predicate_prediction'], df_sentence_repetition['dep'], df_sentence_repetition['morph']):
                if pred == True and 'Voice=Pass' in voice or pred == True and 'pass' in dep:
                    df_sentence_repetition.passive = True
                    df.loc[df_sentence_repetition.index, 'passive'] = True # uncomment # I used .loc since otherwise it did not work for the final df plus I got a
                #elif pred == True and 'Voice=Pass' not in voice or pred == True and 'pass' not in dep:
                #    df_sentence_repetition.passive = False
                #    df.loc[df_sentence_repetition.index, 'passive'] = False
                
    
    # to establish the full constituent for each token
            nlp = spacy.load("en_core_web_sm")
            sentence = [] 
            for (ind, ident, token, pos) in (zip(df_sentence_repetition.index, df_sentence_repetition['id'], df_sentence_repetition['form'], df_sentence_repetition['upos'])):
                sentence.append(token)

            sent = ' '.join(sentence)
            doc = nlp(sent)

            count_df = 0
            for ind in df_sentence_repetition.index:        
                count_doc = 0
                count_df = count_df + 1
                for s in doc.sents:
                    for t in s:
                        count_doc = count_doc + 1
                        if count_df == count_doc:
                            full_const = list(t.subtree)
                            full_c = ' '.join(map(str, full_const))
                            full_c = re.sub(r'\s+([?.!"])', r'\1', full_c)
                            df_sentence_repetition.loc[ind, 'full_constituent'] = full_c
                            df.loc[ind, 'full_constituent'] = full_c
    
    
    
    
    
    
    #write dataframe out
    df.to_csv(path_to_save, index=False)
    
    
    if print_status == True:
        
        print(' Features extracted:')
        #list features
        print(' - Constituents')
        print(' - Passive / Active')
        #print(' - ...')
        print('\n - completed')
    
    return executionMode_dict


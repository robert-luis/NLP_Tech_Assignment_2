#!/usr/bin/env python
# coding: utf-8

# # 1 Data Import
# 
# This script provides a function to read in the conll files and transfer them into a dataframe <br>
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

# ## Conll Description
# 
# "Sentences consist of one or more word lines, and word lines contain the following fields:
# 
# ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0). <br>
# FORM: Word form or punctuation symbol. <br>
# LEMMA: Lemma or stem of word form. <br>
# UPOS: Universal part-of-speech tag. <br>
# XPOS: Language-specific part-of-speech tag; underscore if not available. <br>
# FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available. <br>
# HEAD: Head of the current word, which is either a value of ID or zero (0). <br>
# DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one. <br>
# DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs. <br>
# MISC: Any other annotation.
# 
# The fields DEPS and MISC replace the obsolete fields PHEAD and PDEPREL of the CoNLL-X format. In addition, we have modified the usage of the ID, FORM, LEMMA, XPOS, FEATS and HEAD fields as explained below.
# 
# The fields must additionally meet the following constraints:
# 
# Fields must not be empty.
# Fields other than FORM, LEMMA, and MISC must not contain space characters.
# Underscore (_) is used to denote unspecified values in all fields except ID. Note that no format-level distinction is made for the rare cases where the FORM or LEMMA is the literal underscore – processing in such cases is application-dependent. Further, in UD treebanks the UPOS, HEAD, and DEPREL columns are not allowed to be left unspecified except in multiword tokens, where all must be unspecified, and empty nodes, where UPOS is optional and HEAD and DEPREL must be unspecified. The enhanced DEPS annotation is optional in UD treebanks, but if it is provided, it must be provided for all sentences in the treebank. "
# 
# 
# *** taken from https://universaldependencies.org/format.html

# In[ ]:


import pandas as pd
import numpy as np


# ## Constants set

# In[ ]:


# retrieved header according to documentation
conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']

# header from lecture form 25.02.
conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 'predicate', 'label']

# included sentenceId
conll_header_adapted = ['sentenceId', 'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 'predicate', 'label']


# ## Functions
# 
# ### Util Function

# In[ ]:


# retrieve longest line
# -> required for the creation of the dataframe later
def retrieveLength(path_to_file):
    c = 0
    max_line_length = -1
    sentences = 0
    tokens = 0
    with open(path_to_file) as file:
        for line in file:


            if line.startswith('# text'):
                sentences += 1
            elif line.startswith('#') or line.startswith('\n'):
                pass
            else:
                values = line.split('\t')
                line_length = len(values)
                if line_length > max_line_length:
                    max_line_length = line_length

                tokens += 1

            c += 1   
    
    return max_line_length, tokens


# ### Conversion function
# 

# In[ ]:


# conversion into dataframe
def createDataFrame(executionMode_dict,
                    mode,                   #('production' / 'sample')
                    model,                  #('train' / 'test')
                    print_status   = False,
                    sentence_limit = None):
    
    # variable assignments
    path_to_file = executionMode_dict[mode]['input'][model]
    path_to_save = '../data/intermediate/' + mode + '_' + model +'_01_importedData.csv'
    
    
    executionMode_dict[mode]['intermediate'][model] = {}
    executionMode_dict[mode]['intermediate'][model]['1_imported'] = path_to_save
    #executionMode_dict[mode]['intermediate'][model]['df']   = ''
    
    
    if print_status == True:
        print('\n\n#### 1 Data Import ####')
    
    
    # start retrieving
    
    max_line_length, tokens = retrieveLength(path_to_file)
    sentences = -1

    ### create header
    
    # create empty dataframe with known columns and fillers for remaining collumns
    headers_df = np.full(max_line_length + 1, np.str)  #  + 1 to add sentence column
    
    # add sentence column to header
    #headers_df[1] = 
    
    # add columns from identified columns
    headers_df[:len(conll_header_adapted)] = conll_header_adapted
    
    # fill remaining column headers with '_'
    required_length_to_fill = len(headers_df) - len(conll_header_adapted)
    label_headers = np.full(required_length_to_fill, 'label')
    numbers_list  = np.arange(1, required_length_to_fill + 1)
    numbers_list = np.array([str(n) for n in numbers_list])
    label_headers = np.char.add(label_headers, numbers_list)
    headers_df[len(conll_header_adapted):] = label_headers
    
    
    ### create dataframe
    df = pd.DataFrame(columns=headers_df)

    
    ### fill dataframe

    # loop through file
    with open(path_to_file) as file:
        for line in file:

            # pass all other lines
            if line.startswith('# text'):
                sentences += 1
                
            elif line.startswith('#') or line.startswith('\n'):
                pass
            
            # only go into token lines
            else:
                
                # omit linebreaks from some lines
                if line.endswith('\n'):
                    line = line.replace('\n', '')
                
                # split input line
                values = np.array(line.split('\t'))

                array  = np.full(max_line_length+1, np.str)
                
                # add sentenceId
                array[0] = sentences
                # add retrieved information from conll file
                array[1:len(values)+1] = values
                # fill remaining columns   !!** use np.nan ?! **!! 
                array[len(values)+1:] = '_'
    
                # create new entry
                df_entry = pd.DataFrame(columns=headers_df, data=[array])

                # concatenate to large dataframe
                df = pd.concat([df, df_entry], axis = 0, ignore_index=True)

            if type(sentence_limit) == int and sentences >= sentence_limit:
                break
                
                
    if print_status == True:
        
        print(f' - # Sentences in file: {len(df.sentenceId.unique())}')
        print(f' - # Tokens in file: {tokens}')
        print(f' - Maxium of columns in file: {max_line_length}')
        
        print(f'\n  - ## {len(df.sentenceId.unique())} sentences were added to dataframe.')
        
        print(f' - Dataframe saved under: {path_to_save}')
        
    
    df.to_csv(path_to_save, index=False )
    
    return executionMode_dict


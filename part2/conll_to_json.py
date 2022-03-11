#!/usr/bin/env python
# coding: utf-8

# # Convert Input Conll to json file
# 
# This script converts the input file (already read in as dataframe) into the json input for the neural SRL scripts

import pandas as pd
import numpy  as np
import json


#### functions taken from part1/1_dataimport 
# copied here to have a file doing the full conversion from conll to json

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


# conversion into dataframe
def createDataFrame(path_to_file):
    
    # retrieved header according to documentation
    conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']

    # header from lecture form 25.02.
    conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 'predicate', 'label']

    # included sentenceId
    conll_header_adapted = ['sentenceId', 'id', 'form', 'lemma', 'upos', 'xpos', 'morph', 'head', 'dep', 'head_dep', 'space', 'predicate', 'label']

    
    
    
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

            #if type(sentence_limit) == int and sentences >= sentence_limit:
            #    break
                
    
    
    return df

def convertConllToJSON(path_to_file, path_to_output):
    
    # read dataframe in
    df = createDataFrame(path_to_file)

    x = ''
    
    ## do conversion
    # loop through sentences
    for s_id in df.sentenceId.unique():

        # filter for only this sentence
        df_sentence = df[df.sentenceId == s_id].copy()   

        # return indices of rows with label True of the columns of the predicates
        indices_gold      = np.where(np.array(df_sentence.predicate) != '_' )[0]

        nr_of_predicates = len(indices_gold)


        # loop through nr_of_predicates
        for i in range(nr_of_predicates):

            # create new dict as json element
            elem = {}
            seq_words  = []
            bio        = []
            pred_sense = []


            # create new copy for working with within this repetition of sentence
            df_sentence_repetition = df_sentence.copy()
            df_sentence_repetition.replace(to_replace='"', value=';')
            #DataFrame.replace(to_replace=None, value=NoDefault.no_default,
        
            # retrieve token forms
            seq_words  = list(df_sentence_repetition.form)

            # assign pred_sense
            pred_sense.append(int(indices_gold[i]))
            pred_sense.append(np.array(df_sentence_repetition.predicate)[indices_gold[i]])
            pred_sense.append('_')
            pred_sense.append(np.array(df_sentence_repetition.xpos)[indices_gold[i]])


            ## labels

            # -> transform labels from all label columns to this one column

            # create filler array
            label_array = np.full(len(df_sentence_repetition), '0')

            # slice df_sentence
            row = df_sentence.iloc[indices_gold[i], :]
            list_of_column_indices_with_V = np.where(np.array(row) == 'V')[0]

            # sanity check -> columns found with V should be 1
            if len(list_of_column_indices_with_V) == 1:

                # do conversion

                # find respective_label_column
                respective_column_index = list_of_column_indices_with_V[0]

                # retrieve column
                respective_label_column = np.array(df_sentence.iloc[:, respective_column_index])

                # replave '_' label with '0'
                respective_label_column[respective_label_column == '_'] = '0'

                # overwrite filler with retrieved labels
                label_array = respective_label_column

            # label_array remains only filled with '_' because no (coherent) labels could be found
            else:
                pass
            # assign retrieved array
            #df_sentence_repetition['label_gold']        = label_array
            for i in range(len(label_array)):
                if label_array[i] != '0':
                    label_array[i] = 'B-' + label_array[i]


            bio = list(label_array)
            elem["seq_words"]  = seq_words
            elem["BIO"]        = bio
            elem["pred_sense"] = pred_sense

            x += json.dumps(elem) + '\n'
            
    with open(path_to_output, 'a') as outfile:
        outfile.write(x)

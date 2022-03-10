#!/usr/bin/env python
# coding: utf-8

# # SRL runner
# 
# This script 
# 
# *Input:*  
# - executionMode_dict
# - mode               -> ('production' / 'sample')
# - model              -> ('train' / 'test')
# - print_status       -> (True / False)
# 
# *Output:* 
# - executionMode_dict 
# - tbc

# ## Setup
# 
# ### Imports
# #### General imports
# 



import warnings
warnings.filterwarnings('ignore')


# #### Script imports


from _1_DataImport import *
from _2_PredicatePrediction import *
from _3_DataframeConversion import *
from _4_FeatureExtraction import *
from _5_ArgumentIdentification import *


# ## Preprocessing Pipeline
# 



def runPreprocessing(executionMode_dict, mode, model, print_status):
    
    if print_status == True:
        print(f'\n\n\n\n##################')
        print(f'#### {mode.upper()} ####')
    
    # 1. create dataframe
    executionMode_dict = createDataFrame(executionMode_dict, 
                                         mode  = mode,
                                         model = model,
                                         print_status = print_status)

    # 2. predict predicates
    executionMode_dict = predictPredicates(executionMode_dict, 
                                           mode  = mode,
                                           model = model,
                                           print_status = print_status)

    # 3. convert/ expand dataframe
    executionMode_dict = convertDataframe(executionMode_dict, 
                                          mode  = mode,
                                          model = model,
                                          print_status = print_status)

    # 4. extract features
    executionMode_dict = extractFeatures(executionMode_dict, 
                                          mode  = mode,
                                          model = model,
                                          print_status = print_status)


    # 5. identify arguments
    executionMode_dict = identifyArguments(executionMode_dict, 
                                          mode  = mode,
                                          model = model,
                                          print_status = print_status)

    return executionMode_dict


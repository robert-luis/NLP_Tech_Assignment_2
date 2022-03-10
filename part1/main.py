#!/usr/bin/env python
# coding: utf-8

# # Main Script
# # NLP Technology - Assignment 2, Part 1
# 

# ## Setup

# ### Imports
# #### General imports

import sys
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# #### Imports of other scripts
# imports
from PrecossesingWrapper import *
from SVM import *


# ### Setting variables


        

print_status = True

# dictionary with meta information

'''

Dictionary to carry all the relevant information for 
    - input, 
    - intermediate, and 
    - output files including their paths
    
    
Possible execution modes:
    - 'production'
    - 'sample'
    - 'custom'
    
'''

executionMode_dict = {
    'production' : {
        'input' : {
            'train' : '../data/input/en_ewt-up-train_excerpt.conllu',
            'test'  : '../data/input/en_ewt-up-test_excerpt.conllu'
        },
        'intermediate': {
            
        },
        'output': {
            
        }
    },
    'sample' : {
        'input' : {
            'train' : '../data/input/srl_univprop_en.example.conll',
            'test'  : '../data/input/srl_univprop_en.test_example.conll'
        },
        'intermediate': {
            
        },
        'output': {
            
        }
    },
    'custom' : {
        'input' : {
            'train' : '',
            'test'  : ''
        },
        'intermediate': {
            
        },
        'output': {
            
        }
    } 
}


#default mode production 
mode  = 'production'

if len(sys.argv) == 2:
    if sys.argv[1] in ['production', 'sample']:
        mode  = sys.argv[1]
    else:
        print('Use argument production or sample to run on set paths in main script. \n')
        print('Otherwise set custom and give test and train path.')
        raise Exception('Check documentation for more information.')
        
        
    
if len(sys.argv) == 4:
    if sys.argv[1] == 'custom':
        mode  = sys.argv[1]
        executionMode_dict[mode]['input']['train'] = sys.argv[2]
        executionMode_dict[mode]['input']['test']  = sys.argv[3]



StartTime=time.time()


if print_status == True:
    print(f'''\n\n\n\n
#############################################
#############################################
#############################################
#### Assignment 2.1                      ####
#### Traditional Semantic Role Labeling  ####

Mode: {mode}

#######################
#### Preprocessing ####
''')


# ## Preprocessing

# ### Training
# 
# Call runPreprocessing function on training dataset to preprocess training dataset
# -> use training = True
# 
# 




executionMode_dict = runPreprocessing(executionMode_dict = executionMode_dict, 
                                      mode  = mode, 
                                      model = 'train', 
                                      print_status = True)


# ### Testing
# Call runPreprocessing function on training dataset to preprocess test dataset
# 




executionMode_dict = runPreprocessing(executionMode_dict = executionMode_dict, 
                                      mode  = mode, 
                                      model = 'test', 
                                      print_status = True)


# ## Classification




print(f'\n\n########################')
print('#### Classification ####')

executionMode_dict = classifyArguments(executionMode_dict = executionMode_dict, 
                                      mode  = mode, 
                                      print_status = True)


# ## Evaluation




print(f'\n\n####################')
print('#### Evaluation ####\n\n')

path_to_report      = executionMode_dict[mode]['output']['classifiedArgumentsReport']
path_to_extended_df = executionMode_dict[mode]['intermediate']['test']['05_identifiedArguments']

print('predicate_prediction, predicate_gold Confusion Matrix')
df       = pd.read_csv(path_to_extended_df, sep=',')
y_predpp = df['predicate_prediction']
y_outpp  = df['predicate_gold']
reportpp = classification_report(y_predpp,y_outpp,digits = 3)
print(reportpp, sep='\t')


print('label_ident_prediction, label_ident_gold Confusion Matrix')
y_predli = df['label_ident_prediction']
y_outli  = df['label_ident_gold']
reportli = classification_report(y_predli,y_outli,digits = 3)
print(reportli, sep='\t')  


print('Argument classification Confusion matrix')
report = pd.read_csv(path_to_report)
print(report)

    
TotalTime = time.time() - StartTime
TotalTime = np.round(TotalTime/60, 2)
print('\n Total Processing Time: ', TotalTime, ' min')

# write results out
with open(f'../data/output/{mode}_results.txt', 'w') as f:
    f.write(f'SRL Assignment 2\nMode:{mode}\n')
    f.write('Predicate Prediction ConfusionMatrix\n\n')
    f.write(reportpp)
    f.write('\n\n\n')
    f.write('Argument Identification ConfusionMatrix\n\n')
    f.write(reportli)
    f.write('\n\n\n')
    f.write('Argument Classification ConfusionMatrix\n\n')
    f.write(report.to_string())
    f.write('\n\n\n')
    f.write(f'Total Processing Time: {TotalTime} min.\n')






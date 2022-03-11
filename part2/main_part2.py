#!/usr/bin/env python
# coding: utf-8
import os, sys
from conll_to_json import *

path_to_file_train = '../data/input/srl_univprop_en.train_2000.conll' 
path_to_output_train = '../data/intermediate/neuralSRL_train_2000.jsonl'  
path_to_file_dev = '../data/input/srl_univprop_en.dev.conll'
path_to_output_dev = '../data/intermediate/neuralSRL_dev.jsonl'

path_to_srl_main = 'srl_main.py'
    
# read srl_main
file  = open(path_to_srl_main, 'r')
lines = file.readlines()

for i in range(len(lines)):
    if lines[i].startswith('    TRAIN_PATH ='):
        lines[i] = '    TRAIN_PATH = \"' + path_to_output_train + '\"\n'
    if lines[i].startswith('    DEV_PATH = \"'):
        lines[i] = '    DEV_PATH = \"' + path_to_output_dev + '\"\n'

# write file back 
file = open(path_to_srl_main, 'w')
file.writelines(lines)
file.close()

import srl_main
os.popen('python srl_main.py')
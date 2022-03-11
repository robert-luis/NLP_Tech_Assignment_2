#!/usr/bin/env python
# coding: utf-8

import os, sys
import srl_main
from conll_to_json import *
from srl_main import *

path_to_file_train = '../data/input/srl_univprop_en.train_2000.conll' 
path_to_output_train = '../data/intermediate/neuralSRL_train_2000.jsonl'  
path_to_file_dev = '../data/input/srl_univprop_en.dev.conll'
path_to_output_dev = '../data/intermediate/neuralSRL_dev.jsonl'

print('byeee')

def main():
    #convertConllToJSON(path_to_file_train, path_to_output_train)
    #convertConllToJSON(path_to_file_dev, path_to_output_dev)
    for x in dir(srl_main):
        print(x)
        if x == 'TRAIN_PATH':
            print('hellooo')
   
    TRAIN_PATH = path_to_output_train
    #DEV_PATH = path_to_output_dev

if __name__ == "__main__":
    main()


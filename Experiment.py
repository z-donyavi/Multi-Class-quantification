# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:48:56 2022

@author: Zahra
"""

import pandas as pd
import numpy as np
import argparse
from runme import run_expereiment
from sklearn import preprocessing
from copy import deepcopy
import sys
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def Run_Experiment(exp_name):
    
    # exp_name = 'wine'
    folder = "C:/Users/Zahra/OneDrive - UNSW/Desktop/Implimentations/quantifiers4python-main"
    dts = pd.read_csv(folder + '/dataset/%s'% exp_name +'/%s' % exp_name + '.csv', index_col=False, engine='python')
    
    X = dts.drop(['class'], axis=1)
    y = dts['class']
    
    C = np.unique(y)
    a = len(C)+1
    y = np.where(y == 0, a, y) # replace class 0 with another number in dataset to prevent confilict with negative class
    
    #Binarization
    lb = preprocessing.LabelBinarizer()
    y = pd.DataFrame(lb.fit_transform(y))
    binary_dts = {}
    for i in range(len(C)):
        D = dts.copy()
        D.loc[:,'class'] = y[i]
        binary_dts[i] = D
        
    # Calculate result for each binary dts and save results into binary_res dictionary
    binary_res = {}
    binary_fin = {}
    for j in range(len(C)):
        dts = binary_dts.get(j)
        X = dts.drop(['class'], axis=1)
        y = dts['class'] 
        result = run_expereiment(X, y)
        binary_res[j] = result
        binary_fin[j] = result.groupby('quantifier', as_index=False)['abs_error'].mean()  
         
               
    #Calculate total MAE for each quantifier    
        n_df = pd.DataFrame()
        n_df['quantifires'] = binary_fin.get(0).iloc[:,0]
        for k in range(len(binary_fin.keys())):
           n_df['D%s'%k]= binary_fin.get(k).iloc[:,1]
         
        n_df = n_df.set_index(n_df['quantifires'])
        n_df = n_df.drop(['quantifires'], axis=1)

        print(n_df.sum(axis = 1))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  args = parser.parse_args()
  Run_Experiment(args.exp)
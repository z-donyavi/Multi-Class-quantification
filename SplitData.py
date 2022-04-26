# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:48:56 2022

@author: Zahra
"""

import pandas as pd
import numpy as np

path = "C:/Users/Zahra/OneDrive - UNSW/Desktop/Zahra_data/datasets/wine.csv"

dta = pd.read_csv(path)
#dta.to_csv("C:/Users/Zahra/OneDrive - UNSW/Desktop/quantification_paper-master/"\
#"quantification_paper-master/data/yeast/1.csv")
C = np.unique(dta.Class)

for i in range(len(C)):
    print(i)
    D = dta.copy()
    p = C[i]
    for j in range(len(D.Class)):
        print(j)
        if D.Class[j] != p:
            D.Class[j] = 0
    D.to_csv("C:/Users/Zahra/OneDrive - UNSW/Desktop/Zahra_data/datasets/D"+str(i)+".csv", sep=',')

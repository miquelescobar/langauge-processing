#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:02:45 2020

@author: jordi
"""

import pickle

#%%

file1 = open("/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/idx2cat_gram",'rb')
dict1 = pickle.load(file1)

file2 = open("/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/idx2cat_gram_2",'rb')
dict2 = pickle.load(file2)

file3 = open("/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/idx2cat_gram_3_1",'rb')
dict3 = pickle.load(file3)

file4 = open("/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/idx2cat_gram_4",'rb')
dict4 = pickle.load(file4)

#%%

dict1.update(dict2)
dict1.update(dict3)
dict1.update(dict4)

#%%

len(dict1)

#%%

with open("idx2cat_gram_full", 'wb') as f:
    pickle.dump(dict1, f)









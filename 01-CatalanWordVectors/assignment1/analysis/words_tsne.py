#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:04:00 2020

@author: jordi
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pprint import pprint
from random import sample
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#%%

class Vocabulary(object):
    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.token2idx = {}
        self.idx2token = []
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        if pad_token is not None:
            self.pad_index = self.add_token(pad_token)
        if unk_token is not None:
            self.unk_index = self.add_token(unk_token)
        if eos_token is not None:
            self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def get_index(self, token):
        if isinstance(token, str):
            return self.token2idx.get(token, self.unk_index)
        else:
            return [self.token2idx.get(t, self.unk_index) for t in token]

    def __len__(self):
        return len(self.idx2token)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

token_vocab = Vocabulary()
token_vocab.load('./ca.wiki.vocab')

#%%
"""LOAD MODEL"""

PATH = '/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/ca-100_c.pt'

state_dict = torch.load(PATH, map_location=torch.device('cpu'))

input_word_vectors = state_dict['emb.weight'].numpy()
output_word_vectors = state_dict['lin.weight'].numpy()


#%%
"""PRINCIPAL COMPONENTS OF WORD VECTORS"""

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(input_word_vectors)

principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal component 1', 'principal component 2'])

principalDf.head()

#%%
"""JOIN WORD VECTOR WITH GRAMATICAL CATEGORY"""

file_cat_gram = open("/home/jordi/Documents/UPC/6e quatri/POE/Pràctiques/1/idx2cat_gram_full",'rb')
cat_gram = pickle.load(file_cat_gram)

len(cat_gram.values())

cat_gram_poss = set()

for cat_gram_set in cat_gram.values():
    if cat_gram_set is not None: 
        cat_gram_poss = cat_gram_poss.union(cat_gram_set)
        
pprint(cat_gram_poss)

#%%
def mean(v):
    if len(v) > 0:
        return sum(v)/len(v)
    else:
        return 0
    
def centroide(x, y, ax, color):
    ax.scatter(mean(x), mean(y), c = 'k', s = 200, alpha = 1, marker = 'o')
    ax.scatter(mean(x), mean(y), c = color, s = 100, alpha = 1, marker = 'o')


def plot_cat_gram(targets):
    cat_gram_target = []
    
    for i in range(len(principalDf)):
        cat_gram_word = cat_gram[token_vocab.idx2token[i]]
        
        if cat_gram_word is not None:
            cat_gram_word_target = []
            for target, tags in targets.items():
                cat_gram_word_tags = cat_gram_word.intersection(tags)
                if len(cat_gram_word_tags) > 0:                        #If the word shares a tag with the target, append the target
                    cat_gram_word_target.append(target)
    
            if len(cat_gram_word_target) > 0:
                cat_gram_target.append(str(sample(cat_gram_word_target, 1)[0])) #If the word has more than one target, choose one.
                continue
        cat_gram_target.append('None')

    
    principalDf["cat_gram"] = cat_gram_target
    
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:len(targets.keys())]
    for target, color in zip(targets.keys(),colors):
        indicesToKeep = principalDf['cat_gram'] == target
        x = principalDf.loc[indicesToKeep, 'principal component 1']
        y = principalDf.loc[indicesToKeep, 'principal component 2']
        ax.scatter(x, y , c = color, s = 10, alpha = 0.35)
    
    ax.legend(targets.keys())
    
    for target, color in zip(targets.keys(),colors):
        indicesToKeep = principalDf['cat_gram'] == target
        x = principalDf.loc[indicesToKeep, 'principal component 1']
        y = principalDf.loc[indicesToKeep, 'principal component 2']
        centroide(x, y, ax, color)
    
    ax.grid()


#%%
'''
 
'Adjectius' : {'adj.',
              'adj. f.',
              'adj. indef.'}

'Noms' : {'n. f.',
         'n. f. pl.',
         'n. m.',
         'n. m. i f.',
         'n. m. o f.',
         'n. m. pl.'}

'Pronoms' : {'pron.',
             'pron. dem.',
             'pron. indef.',
             'pron. inter.',
             'pron. pers.',
             'pron. poss.',
             'pron. quant.',
             'pron. rel.'}

'Verbs' : {'v. copulatiu',
         'v. impersonal',
         'v. intr.',
         'v. tr.'}

'Determinants' : {'art. det.',
                   'det.',
                   'det. indef.',
                   'det. poss.',
                   'det. quant.',
                   
'Adverbis' : {'adv.'}
}

'''

#%%

#Guai
targets1 = {'Noms femenins':{'n. f.'}, 'Noms masculins':{'n. m.'}}

#Meh
targets1_1 = {'Noms femenins':{'n. f.'}, 'Noms masculins':{'n. m.'}, 'Noms both':{'n. m. i f.','n. m. o f.'}}

#Verga
targets1_2 = {'Noms femenins pl':{'n. f. pl.'}, 'Noms masculins':{'n. m. pl.'}}

#Guai
targets2 = {'Verbs':{'v. copulatiu','v. impersonal','v. intr.','v. tr.'}, 
            'Noms':{'n. f.','n. f. pl.','n. m.','n. m. i f.','n. m. o f.','n. m. pl.'}
            }
#Guai
targets3 = {'Verbs':{'v. copulatiu','v. impersonal','v. intr.','v. tr.'}, 
            'Noms':{'n. f.','n. f. pl.','n. m.','n. m. i f.','n. m. o f.','n. m. pl.'},
            'Adjectius' : {'adj.','adj. f.','adj. indef.'},
            }

targets3_1 = {'Verbs':{'v. copulatiu','v. impersonal','v. intr.','v. tr.'}, 
            'Noms':{'n. f.','n. f. pl.','n. m.','n. m. i f.','n. m. o f.','n. m. pl.'},
            'Adjectius' : {'adj.','adj. f.','adj. indef.'},
            'Adverbis' : {'adv.'}
            }

#Verga
targets4 = {'Pronoms' : {'pron.','pron. dem.','pron. indef.','pron. inter.','pron. pers.','pron. poss.','pron. quant.','pron. rel.'},
            'Determinants' : {'art. det.','det.','det. indef.','det. poss.','det. quant.'}
            }
#Verga
targets5 = {'Adjectius' : {'adj.'}, 'Adjectius femenins' : {'adj. f.'}, 'Adjectius indefinits': {'adj. indef.'}}

#Guai | Els verbs transitius necessiten un complement directe a continuació
targets6 = {'Verbs instransitius':{'v. intr.','v. copulatiu','v. impersonal.'}, 'Verbs transitius':{'v. tr.'}}

#Verga
targets6_1 = {'Verbs impersonals':{'v. impersonal.'}, 'Verbs no impersonals':{'v. copulatiu','v. intr.','v. tr.'}}

#Verga
targets6_2 = {'Verbs copulatius':{'v. copulatiu'}, 'Verbs no copulatius':{'v. impersonal','v. intr.','v. tr.'}}


plot_cat_gram(targets1)
plot_cat_gram(targets1_1)
plot_cat_gram(targets1_2)
plot_cat_gram(targets2)
plot_cat_gram(targets3)
plot_cat_gram(targets3_1)
plot_cat_gram(targets4)
plot_cat_gram(targets5)
plot_cat_gram(targets6)
plot_cat_gram(targets6_1)
plot_cat_gram(targets6_2)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:28:54 2020

@author: jordi
"""

from bs4 import BeautifulSoup
import os
import urllib.request
import requests
import pickle

#%%

def get_cat_gram(word):
    url = f'https://www.wordreference.com/definicio/{word}'   
    
    sauce = requests.get(url)
    
    soup = BeautifulSoup(sauce.text, "html.parser")
    
    body = soup.body
    
    cat = set()
    
    for entry, tags in zip(body.find_all('span', class_="lemmaLa main"), body.find_all('span', class_="posLa")):
        if entry.text.split()[0] == word:
            tags_split = tags.text.split("/")
            for tag in tags_split:
                cat.add(tag)
            
    if len(cat) == 0:
        return None
    else:
        return cat

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

idx2cat_gram = []
for i, word in enumerate(token_vocab.idx2token):
    if i % 100 == 0:
        print(i)
    idx2cat_gram.append(get_cat_gram(word))
    
idx2cat_gram



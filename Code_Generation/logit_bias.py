#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:30:23 2023

@author: ningnong
"""

import tiktoken
from typing import List
from functools import reduce, lru_cache
from abc import ABC,abstractmethod

def token_counts(text : str, model : str, counts=None)->dict: 
    """
    Parameters
    ----------
    text : str
        some text.
    model : str
        the tokenizer to use.
    counts : TYPE, optional
        dictionary of token id : count. The default is None.

    Returns
    -------
    dict
        counts.

    """
    if counts is None:
        counts = {} 
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    for token in tokens:
        if token not in counts:
            counts[token] = 1
            continue 
        counts[token] += 1
    return counts 

def get_top_k(token_counts : dict, k)->List: 
    lst = list(token_counts.items())
    return sorted(lst, key=lambda x : x[1]) [len(lst)-k:]

def compute_biases(top_k : List, max_negative_bias : float)->List[float]: 
    """
    see: https://arxiv.org/pdf/2306.04140.pdf

    Parameters
    ----------
    top_k : List
        list of tuples (token id, count), top k most frequently occuring tokens.
    max_negative_bias : float
        maximum bias to apply.

    Returns
    -------
    List[float]
        DESCRIPTION.

t    """
    # computes the number of total tokens, using the counts for each token
    total_tokens = reduce(lambda acc,x : acc+x[1], top_k, 0) 
    
    return { id : -max_negative_bias * count/total_tokens for id,count in top_k }

def union_all(bias_list):
    return reduce(lambda acc,x : combine_biases(acc, x), bias_list, {})

def combine_biases(b1 : dict, b2 : dict)->dict:
    """
    given 2 bias dictionaries, combine them using the following rules
        
    1. if d is not in b1 intersect b2, just add it
    2. if d is in b1 interesct b2, apply the largest negative bias (or the take the smallest value)

    Parameters
    ----------
    b1 : dict
        DESCRIPTION.
    b2 : dict
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    
    res = {}
    for key,val in b1.items(): 
        if key in b2:
            res[key] = min(b2[key],val) # rule 2
            continue
        res[key] = val # rule 1    
    for key,val in b2.items(): 
        if key in res:
            continue 
        res[key] = val # rule 1
    return res

class Bias(ABC):
    
    def __init__(self, model):
        
        self.model = model
    
    @abstractmethod
    def get_biases(self, code_snippets : List[str])->dict:
        """
        maps a list of code snippets (previously generated solutions)
        to a bias dictionary

        Parameters
        ----------
        code_snippets : List[str]
            list of code snippets.

        Returns
        -------
        dict.

        """
        pass
    
    @abstractmethod 
    def get_info(self):
        pass

class no_bias(Bias): 
    
    def __init__(self, model):
        super().__init__(model) 
        
    def  get_biases(self, code_snippets : List[str])->dict:
        return {} 
    
    def get_info(self):
        return "nothing"
    
class pool_from_all(Bias):
    def __init__(self, model, max_bias, num_tokens):
        """
        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(model)
        self.enc = None
        self.max_bias = max_bias
        self.k = num_tokens
        self.biases = {}
        
    def code_to_bias(self, code : str):
        
        token_counts = self.count_tokens(code)
        
        top_k = self.get_top_k(token_counts)
        
        bias = self.compute_biases(top_k, self.max_bias)
        
        return bias
        
    def get_info(self): 
        
        info = {'name' : 'pool_from_all', 
                'tokens_pool_size' : self.k,
                'max_bias' : self.max_bias}
        
        return info
        
    def get_biases(self, code_snippets : List[str]):
        
        code = ""
        for s in code_snippets:
            code += s
            
        return self.code_to_bias(code)
    
    def compute_biases(self, top_k : List, max_negative_bias : float)->List[float]: 
        """
        see: https://arxiv.org/pdf/2306.04140.pdf
    
        Parameters
        ----------
        top_k : List
            list of tuples (token id, count), top k most frequently occuring tokens.
        max_negative_bias : float
            maximum bias to apply.
    
        Returns
        -------
        List[float]
            DESCRIPTION.
    
    t    """
        
        total_tokens = reduce(lambda acc,x : acc+x[1], top_k, 0) # computes the number of total tokens, using the counts for each token
 
        return { id : -max_negative_bias * count/total_tokens for id,count in top_k }
        
    
    def tokenize(self, text: str):
        if self.enc is None:
            self.enc = tiktoken.encoding_for_model(self.model)
        return self.enc.encode(text)
   
    def count_tokens(self, text : str)->dict: 
        """
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        
        counts = {} 
        
        tokens = self.tokenize(text)
        
        for token in tokens:
            if token not in counts:
                counts[token] = 1
                continue 
            counts[token] += 1
            
        return counts 
    
    # get top k most frequently occuring tokens
    def get_top_k(self, token_counts : dict)->List: 
        lst = list(token_counts.items())
        if len(lst) <= self.k:
            return lst
    
        return sorted(lst, key=lambda x : x[1]) [len(lst)-self.k:]
    

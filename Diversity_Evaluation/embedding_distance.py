# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:12:39 2023

acknowledgement: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
"""



import os
import json
import torch
from typing import List
from torch.nn.functional import normalize
from transformers import BertTokenizer, BertModel,AutoTokenizer,AutoModel

from functools import lru_cache



MODEL = "mrm8488/CodeBERTaPy"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
class SimilarityScore: 
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.model.eval()

    def cosine(self, v1,v2): 
        mag1 = torch.norm(v1)
        mag2 = torch.norm(v2)
        return torch.dot(v1, v2)/(mag1*mag2)
    
    def get_tokens(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        return tokenized_text

    @lru_cache(maxsize=25,typed=False)
    def embed(self, text):
        tokenizer = self.tokenizer
        #context length is at most 512 tokens, 
        # only take first 512 tokens
        tokens = self.get_tokens(text)[:512]
        token_indices = tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        tokens_tensor = torch.tensor([token_indices])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding
        
    def similarity(self, text1,text2):
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)
        return self.cosine(vec1,vec2)
    


def analyse_code_cosine(dir):
    """ 
    produces a json file of all of the simirlaity scores of the code snippets stored in dir
    note dir need to be structured like so: dir->problem_name->sample0->solution.py
    """
    
    res = {} 
    
    code = {}
    with open(dir, 'r') as file:
        code = json.load(file)
        
    model = AutoModel.from_pretrained(MODEL,
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
    model.eval()
        
    vec_embedding_sim = SimilarityScore(model)

    for problem in code:
        res[problem] = {}
        for sample in code[problem]:
            res[problem][sample] = {}
            res[problem][sample]['bert_cos'] = []
            code_snippets = list(code[problem][sample].items())
            for i in range(len(code_snippets)):
                for j in range(i+1, len(code_snippets)):
                    name1,name2 = code_snippets[i][0],code_snippets[j][0]
                    code1,code2 = code_snippets[i][1],code_snippets[j][1]
                    
                    try:
                        sim = vec_embedding_sim.similarity(code1,code2)
                    except:
                        sim = -2
                    
                    res[problem][sample]['bert_cos'].append((name1,name2,sim))
                    
                    print(f"analysed {problem} {sample} {name1} {name2} : {sim}...")
                    
    return res
                     
  

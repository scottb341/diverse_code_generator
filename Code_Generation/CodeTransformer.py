#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 2023

@author: ningnong
"""

from abc import ABC,abstractmethod
import ast

class Transformer(ABC): 
    """
    used for transforming code
    """
    
    def __init__(self):
        pass
    
    @abstractmethod 
    def transform(self, text : str)->str:
        """
        given some code 'text', edit/transform the code

        Parameters
        ----------
        text : str
            code snippet.

        Returns
        -------
        str
            transformed code.

        """
        pass


class ITransformer(Transformer):
    
    def __init__(self):
        pass 
    
    def transform(self, text):
        return text



class OnlyAllow(Transformer): 
    """
    removes any code that is not part of the given types
    """
    
    
    def __init__(self, allowed_types):
        self.allowed = allowed_types
        
    def transform(self, text : str)->str:
        """
        removes any code that is not part of the given types
  

        Parameters
        ----------
        text : str
            code snippet.

        Returns
        -------
        str
            transformed code.

        """
        
        
        a = ast.parse(text) 
    
        body = []
        
        for node in a.body: 
                
            for t in self.allowed:
                if isinstance(node, t):
                    body.append(node)
        
        a.body = body
        
        return ast.unparse(a)



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
    
class ChainedTransformer(Transformer):
    """ 
    transforms the text using the first transformer, then transforms the result
    using the second transformer
    """
    
    # transform using t1 first than pass onto t2 to transform again
    def __init__(self, t1 : Transformer, t2: Transformer):
        super()
        self.t1 = t1 
        self.t2 = t2 
        
    def transform(self, text : str)->str:
        return self.t2.transform(self.t1.transform(text))
    


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
    
class ExtractFunctionBody(Transformer): 
    
    def __init__(self):
        pass 
    
    def transform(self, text : str)->str:
        
        a = ast.parse(text) 
    
        
        body = a.body[0].body
        
        return ast.unparse(body)
        

class WrapInClass(Transformer): 
    """
    Some test cases may require the code to be in a class, so this wraps the function in one
    Also renames the function
    """
    
    def __init__(self, classname : str, funcname : str)->str:
        super()
        self.classname = classname 
        self.funcname = funcname
        
    def transform(self, text : str)->str:
        a = ast.parse(text)
        
        for i in range(len(a.body)):
            a.body[i] = self.wrap(a.body[i])
        
        return ast.unparse(a)
        
    def wrap(self, a : ast)->ast:
        """
        wraps ast into an ast of type ClassDef

        Parameters
        ----------
        a : ast
            ast of code snippet.

        Returns
        -------
        ast
            ast wrapped into class.

        """
        
        
        initFunc = ast.FunctionDef(name='__init__', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Pass()], decorator_list=[])
        
        initFunc.lineno = 0
        
        a.name = self.funcname
        
        # adds 'self' argument to the ast 'a'
        
        a.args = [ast.arg(arg='self')]+a.args
        
        classDef = ast.ClassDef(name=self.classname, bases=[], keywords=[], body=[initFunc, a],decorator_list=[])
    
        return classDef
    
    



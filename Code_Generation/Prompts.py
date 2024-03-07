#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 

@author: Scott Blyth
"""

from abc import ABC,abstractmethod
from typing import List,Callable
from SolutionGenerator import Session
from CodeTransformer import Transformer
from logit_bias import Bias,token_counts,get_top_k,compute_biases,union_all


class CodeGenerator(ABC): 
    """ 
    class for generating code using LLMs
    """
    
    def __init__(self, prob_description: str):
        self.code_snippets = []
        self.description = prob_description
 
    @abstractmethod
    def Generate(self, s : Session, number : int, transform : Transformer)->[str]:
        pass
    
    @abstractmethod 
    def get_name(self):
        pass


#############
## Control ## 
#############
class RegenPrompt(CodeGenerator): 
    def __init__(self, description : str):
        super().__init__(description)
        
    def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
        s.Reset()
        for i in range(number):
            s.Prompt(f"{self.description}\n")
        solutions = s.GetCode()
        solutions = filter(lambda x : len(x) > 0, solutions)
        res = []
        for sol in solutions:
            try:
                res.append(transformer.transform(sol))
            except:
                pass
    
        return list(filter(lambda x : len(x)>0, res))
    
    def get_name(self):
        return "Regenereate"
    

class N_Different(CodeGenerator):
    def __init__(self, prob_description: str):
        super().__init__(prob_description)
        
    def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
        s.Reset() 
        s.Prompt(f"{self.description}. \nGive me {number} different solutions for this problem in python. Wrap each solution in ```. For example ```def f(x)\n\treturn x```\n") 
        
        solutions = s.GetCode()
    
        
        solutions = filter(lambda x : len(x) > 0, solutions)
    
        res = []
    
        for sol in solutions:
            try:
                res.append(transformer.transform(sol))
            except:
                pass
            
        return list(filter(lambda x : len(x)>0, res))
        
    def get_name(self):
        return "N-different"
 
def n_k_different(k: int, bias_computer : Bias):
    class a_class(CodeGenerator):
        
        def __init__(self, prob_description: str):
            super().__init__(prob_description)
            
        def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
            s.Reset() 
            model = s.get_model()
           
            for i in range(0, number-1, k):
                
                s.Prompt(f"{self.description}. \nGive me {k} different solutions for this problem in python.") 
          
                code_snippets = s.GetCode()
                
                if bias_computer is not None:
                    biases = bias_computer.get_biases(code_snippets)
                else:
                    biases = {}
                
                model.update_logit_biases(biases)
                
            
            rest = number%k
            
            s.Prompt(f"{self.description}. \nGive me {rest} different solutions for this problem in python.") 
            
            
            solutions = s.GetCode()
        
            
            solutions = filter(lambda x : len(x) > 0, solutions)
        
            res = []
        
            for sol in solutions:
                try:
                    res.append(transformer.transform(sol))
                except:
                    pass
                
            return list(filter(lambda x : len(x)>0, res))
        
        def get_name(self):
            
            info = 'nothing'
            
            if bias_computer is not None:
                info = bias_computer.get_info()
            
            desc = {'name' : 'n_k_diff', 
                    'k' : k, 
                    'bias' : info}
            
            return desc
        
    return a_class


    
    
    

    
def logit_bias_pool(k, bias_strength):  
    class Logit_Bias_Prompt(CodeGenerator): 
        def __init__(self, description : str):
            super().__init__(description)
        
        def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
            
            s.Reset()
            
            s.Prompt(f"{self.description}\nWrite the function in python")
             
            
            model_name = s.get_model().model
            
            
            # gets the counts of all of the tokens appearing in the 
            # most recent solution
            token_count = token_counts(s.GetCode()[-1], model_name) 
            
            
            top_k = get_top_k(token_count, k)
            biases = compute_biases(top_k, bias_strength)
            for i in range(number-1):
                s.get_model().update_logit_biases(biases)
                s.Prompt(f"{self.description}\nWrite the function in python")
                
                
                # updates the token counts using the most recent solution
                token_counts(s.GetCode()[-1], model_name, token_count)
                top_k = get_top_k(token_count, k)
                biases = compute_biases(top_k, bias_strength)
             
            solutions = s.GetCode()
            
            solutions = filter(lambda x : len(x) > 0, solutions)
            
            res = []
        
            for sol in solutions:
                try:
                    res.append(transformer.transform(sol))
                except:
                    pass
                
        
            return list(filter(lambda x : len(x)>0, res))
        
        
        def get_name(self):
            description = {"name" : "logit_bias_regeneration",
                           "pool_size" : k,
                           "bias" : bias_strength}
            return description
    return Logit_Bias_Prompt



def negative_bias_k(similarity , sim_measure_name : str, bias : float, token_pool_size: int, sol_pool_size : int):
    """
    

    Parameters
    ----------
    similarity : (code1 : str, code2 : str)->float
        function that computes the similarity between code1 and code2.
    sim_measure_name : str
        name of similarity measure.
    bias : float
        maximum negative bias.
    token_pool_size : int
        maximum number of tokens to take from each solution.
    sol_pool_size : int
        the number of solutions to select against at one time.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    class Negative_Bias_Solutions(CodeGenerator): 
        def __init__(self, description : str):
            super().__init__(description)
            
        def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
             
             s.Reset()
             
             s.Prompt(f"{self.description}\nWrite me a solution in python\n")
              
             model_name = s.get_model().model
             
             token_count = token_counts(s.GetCode()[-1], model_name)
             top_k = get_top_k(token_count, token_pool_size)
             biases = compute_biases(top_k, bias)
             
             
             self.code_snip_biases = [None]*number # for caching the biases
             sims_sum = [0]*number
             
             
             for i in range(number-1):
                 s.get_model().update_logit_biases(biases)
                 print(i)
                 s.Prompt(f"{self.description}\nWrite me a solution in python\n")
                 
                 code_snips = s.GetCode()
                 
                 
                 # 
                 indices = get_k_most_similar(code_snips, similarity, i , sol_pool_size, sims_sum)
                 
                
                 
                 # compute the biases of each candidate code snippet 
                 
                 bias_list = [self.code_to_bias(code_snips[index], index, model_name) for index in indices]
                 
                 # union all the biases
                 
                 lst = list(union_all(bias_list).items())
                 
                 
                 # get top 100 (all of them if <= 100 tokens)
                 if len(lst) > 100:
                     lst = sorted(lst, key=lambda x : x[1])[max(0,len(lst)-100):]
                 
                 biases = {id : count for id,count in lst}
                 
                 
              
             solutions = s.GetCode()
             
             solutions = filter(lambda x : len(x) > 0, solutions)
             
             res = []
         
             for sol in solutions:
                 try:
                     res.append(transformer.transform(sol))
                 except:
                     pass
                 
         
             return list(filter(lambda x : len(x)>0, res))
         
        def code_to_bias(self, text : str, index : int, model : str)->List[float]: 
            """
            

            Parameters
            ----------
            text : str
                DESCRIPTION.
            index : int
                DESCRIPTION.
            model : str
                DESCRIPTION.

            Returns
            -------
            List
                DESCRIPTION.

            """
            
            if self.code_snip_biases[index] is not None:
                return self.code_snip_biases[index]
            
            counts = token_counts(text, model)
            
            top_k = get_top_k(counts, token_pool_size)
            
            self.code_snip_biases[index] = compute_biases(top_k, bias)
            
            return self.code_snip_biases[index]
        
        
    
        def get_name(self)->str:
            description = {"name" : "Negative_Bias_Solutions",
                           "sim name" : sim_measure_name,
                           "maximum_bias" : bias,
                           "tokens_per_solution" : token_pool_size, 
                           "k_value" : sol_pool_size}
            return description
    return Negative_Bias_Solutions
    


def get_k_most_similar(code_snippets : List[str], similarity, index: int, k : int, sims_sum = None)->str:
    """
    

    Parameters
    ----------
    code_snippets : List
        DESCRIPTION.
    similarity : (code1 : str, code2 : str)->float
        DESCRIPTION.
    index : int
        DESCRIPTION.
    k : int
        DESCRIPTION.
    sims_sum : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    str
        DESCRIPTION.

    """
    
     
    for i in range(index):
                
        sim =  similarity(code_snippets[i], code_snippets[index])
    
        
        sims_sum[i] += sim
        sims_sum[index] += sim
            
    lst = [(i, val) for i,val in enumerate(sims_sum[:index+1])]
    
    lst = sorted(lst, key=lambda x : x[1], reverse=True)
    
    
    lst = [i for i,val in enumerate(lst) ] 
             
             
    return lst[:k]


def negative_bias(bias : float, token_pool_size: int): 
    """
    

    Parameters
    ----------
    bias : float
        DESCRIPTION.
    token_pool_size : int
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    class Negative_Bias(CodeGenerator): 
        def __init__(self, description : str):
            super().__init__(description)
            
        def Generate(self, s : Session, number : int, transformer : Transformer)->[str]:
             
             s.Reset()
             
             s.Prompt(f"{self.description}\nWrite me a solution in python\n")
              
             model_name = s.get_model().model
             
             token_count = token_counts(s.GetCode()[-1], model_name)
             top_k = get_top_k(token_count, token_pool_size)
             biases = compute_biases(top_k, bias)
             
             
             self.code_snip_biases = [None]*number # for caching the biases
             
             
             for i in range(number-1):
                 s.get_model().update_logit_biases(biases)
                 print(i)
                 s.Prompt(f"{self.description}\nWrite me a solution in python\n")
                 
                 self.code_snips = s.GetCode()
                 
                 # compute the biases of each candidate code snippet 
                 
                 bias_list = [self.code_to_bias(j, model_name) for j in range(i)]
                 
                 # union all the biases
                 
                 lst = list(union_all(bias_list).items())
                 
                 
                 # get top 100 (all of them if <= 100 tokens)
                 if len(lst) > 100:
                     lst = sorted(lst, key=lambda x : x[1])[max(0,len(lst)-100):]
                 
                 # map id to count
                 biases = {id : count for id,count in lst}
                 
                 
              
             solutions = s.GetCode()
             
             solutions = filter(lambda x : len(x) > 0, solutions)
             
             res = []
         
             for sol in solutions:
                 try:
                     res.append(transformer.transform(sol))
                 except:
                     pass
                 
         
             return list(filter(lambda x : len(x)>0, res))
         
        def code_to_bias(self, index : int, model : str)->List[float]: 
            """
            

            Parameters
            ----------
            text : str
                DESCRIPTION.
            index : int
                DESCRIPTION.
            model : str
                DESCRIPTION.

            Returns
            -------
            List
                DESCRIPTION.

            """
            
            if self.code_snip_biases[index] is not None:
                return self.code_snip_biases[index]
            
            counts = token_counts(self.code_snips[index], model)
            
            top_k = get_top_k(counts, token_pool_size)
            
            self.code_snip_biases[index] = compute_biases(top_k, bias)
            
            return self.code_snip_biases[index]
        
        
    
        def get_name(self)->str:
            description = {"name" : "Negative_Bias",
                           "maximum_bias" : bias,
                           "tokens_per_solution" : token_pool_size}
            return description
    return Negative_Bias
    



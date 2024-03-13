#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:13:34 2023

@author: Scott Blyth
"""

import os
from abc import ABC,abstractmethod 
import openai
from CodeExtractor import extract_code
from typing import TypeVar, List, Tuple
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
T = TypeVar('T')
models = {"gpt-3.5-turbo", "gpt-3.5-turbo-0301","text-davinci-003","text-davinci-002", "code-davinci-002"}
# utils

def flatten(code): 
    """
    Flattens the given list
    """
    if not isinstance(code, list):
        return code 
    res = []
    for snippet in code:
        lst = flatten(snippet)
        if not isinstance(lst, list):
            res.append(lst)
            continue
        res += lst
    return res



@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(20))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


class LLM_Model(ABC):
    """
    A large language model that can be prompted to, and it gives reponses in the form of a string. 
    It can also return code snippets as part of it's response. 
    """
    def __init__(self, name):
        self.name = name 
    @abstractmethod 
    def GetResponse(self, prompt : str)->str:
        """
        method for prompting the LLM

        Parameters
        ----------
        prompt : str
            the prompt.

        Returns
        -------
        str
            the response.

        """
        pass
    
    @abstractmethod 
    def GetDescription(self)->T:
        """
        description of the model. Includes information such as it's name and the parameters used 
        (temperature,top_p,max_tokens etc.)

        Returns
        -------
        T
            description of the model.

        """
        pass
    
    @abstractmethod 
    def ExtractCode(self, text : str)->[str]:
        """
        

        Parameters
        ----------
        text : str
            the response from the LLM.

        Returns
        -------
        [str]
            The list of code-snippets in the reponse stored as a string.

        """
        pass

    def update_logit_biases(self)->List:
        pass
    
class Me_Model(LLM_Model):
    """
    Useful for debugging without requiring access to an LLM
    """
    def __init__(self):
        super().__init__("Me")
    
    def GetResponse(self, prompt : str)->str:
        print(prompt)
        return input("What is your response?: ")
    
    def GetDescription(self)->str:
        return "Just me"
    
    def ExtractCode(self, text):
        return extract_code(text)
    
    
class Completion(LLM_Model):
    """
    LLM_Model for some LLMs by openai.
    Note that this is not for models such as GPT-3.5-turbo. This is for models
    that allow changing settings such as the temperature and max tokens.
    """
    def __init__(self, model, temp, max_tokens, top_p=1,frequency_penalty=0):
        super().__init__(model)
        self.model = model
        self.temp = temp
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        
    def GetResponse(self, prompt : str)->str:
        """
        method for prompting the LLM. Requires an auth key to be stored in the envrionment "OPENAI_KEY"
        Parameters
        ----------
        prompt : str
            the prompt.

        Returns
        -------
        str
            the response.
        """
        response = self.GetReponseAux(self.model, prompt, self.temp, self.max_tokens)
        return self.GetMessage(response)
        
    def GetReponseAux(self, model : str, prompt : str, temp : float, max_tokens : int):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        prompt : TYPE
            DESCRIPTION.
        temp : TYPE
            DESCRIPTION.
        max_tokens : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if openai.api_key is None:
            openai.api_key = os.environ["OPENAI_KEY"] 
        time.sleep(0.5) # sleep
        return openai.Completion.create(model=model, prompt=prompt, temperature=temp,
                                        top_p = self.top_p,max_tokens=max_tokens,
                                        logit_bias = self.logit_bias,
                                        frequency_penalty = self.frequency_penalty)

    def GetMessage(self, response):
        return response["choices"][0]["text"]
    
    def GetDescription(self)->dict:
        """
        
        model,temperature and max tokens used to generate the responses given the prompts
        Returns
        -------
        dict
            DESCRIPTION.

        """
        return {"model" : self.model,"temperature" : self.temp,
                "top_p" : self.top_p, "max_tokens" : self.max_tokens,
                "frequency_penalty" : self.frequency_penalty}
    
    def ExtractCode(self, text : str) -> [str]:
        """
        Extracts the code from the LLM. 

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        [str]
            DESCRIPTION.

        """
        sols = extract_code(text) 
        return sols
    
class GPT_Model(LLM_Model):
    def __init__(self, api_key, model, max_tokens, temperature, top_p, logit_bias={},
                 frequency_penalty=0, presence_penalty=0):
        super().__init__(model)
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.messages = []
        self.temperature = temperature
        self.top_p = top_p
        self.logit_bias = logit_bias
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.total_tokens = 0
         
    def GetResponse(self, prompt)->str:
        """
        method for prompting the LLM. Requires an auth key to be stored in the envrionment "OPENAI_KEY"
    
        Parameters
        ----------
        prompt : str
            the prompt.
    
        Returns
        -------
        str
            the response.
    
        """ 
        # prompt message
        if openai.api_key is None:
            openai.api_key = self.api_key
        prompt_msg = {"role" : "user", "content" : prompt}
        full_msg = self.messages+[prompt_msg]
        response = completion_with_backoff(model=self.model,
                        max_tokens=self.max_tokens, 
                        messages=full_msg,
                        temperature=self.temperature,
                        top_p = self.top_p,
                        logit_bias=self.logit_bias,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty = self.presence_penalty)
        self.total_tokens += response["usage"]["total_tokens"]
        self.messages = []
        print("response given...")
        return self.GetMessage(response)
    def add_message(self, msg):
        self.messages.append({"role" : "system", "content" : msg})
    def GetMessage(self, response):
        return response["choices"][0]["message"]["content"]
    def GetDescription(self)->str:
        return {"model" : self.model, 'temperature' : self.temperature, 
                'max_tokens' : self.max_tokens, 'top_p' : self.top_p,
                "frequency_penalty" : self.frequency_penalty,
                "presence_penalty" : self.presence_penalty}
    def ExtractCode(self, text):
        # extracts code using the code extraction tools in CodeExtractor.py
        sols = extract_code(text) 
        return sols
    def update_logit_biases(self, logit_bias)->List:
        self.logit_bias = logit_bias

class Session:
    """
    stores the conversation with the LLM, that is, the list of (prompt,reponse) genereated
    from interacting with the LLM. Used to have an ongoing conversation with the LLM rather 
    than the basic prompt and response interaction.
    """
    def __init__(self, model : LLM_Model)->None:
        self.history = [] # (prompt, response)
        self.model = model
        self.code = []
        
    def Prompt(self, prompt : str)->None:
        """
        prompts the LLM "model" and stores the prompt and response in the history of the 
        conversation

        Parameters
        ----------
        prompt : str
            the prompt given to the LLM.

        Returns
        -------
        None.

        """
        response = self.model.GetResponse(prompt)
        self.history.append((prompt, response))
    
    def GetHistory(self)->List[Tuple[str]]:
        """
        method for obtaining the history of the conversation with the LLM

        Returns
        -------
        List[Tuple[str]]
            list of (prompt,repsonse).
        """
        
        return self.history
    
    def Reset(self)->None:
        """
        resets the history

        Returns
        -------
        None
            

        """
        self.history = []
        
    def GetCode(self):
        """
        using the ExtractCode method from the LLM_Model, get all of the code snippets
        that have been generated during the conversation with the LLM.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        code = [self.extract_code(index) for index in range(len(self.GetHistory()))]  
        return flatten(code)
    def extract_code(self, index): 
        if index < len(self.code) and self.code[index] is not None:
            return self.code[index]
        prompt,response = self.GetHistory()[index]
        self.code.append(self.model.ExtractCode(response))
        return self.code[index]
    def get_model(self)->LLM_Model:
        return self.model




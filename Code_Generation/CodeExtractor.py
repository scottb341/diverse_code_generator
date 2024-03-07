#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:47:52 2023

@author: Scott Blyth
"""

import ast
from CodeTransformer import OnlyAllow
from typing import List
import signal
import re
import os
import json
import bisect
import pyperclip
from copy import deepcopy


def syntax_correct(text):
    try: 
        ast.parse(text)
        return True 
    except:
        return False

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
        else:
            res += lst
            
    return res



def linear_next(text : str, extraction_test, index : int) -> str: 
    """
    extracts the next closest code snippet in text[index:]

    Parameters
    ----------
    text : str
        DESCRIPTION.
    extraction_test : TYPE
        DESCRIPTION.
    index : int
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    
    
    curr = (index,index)
    
    i = index
    
    while i < len(text):
        try:
            extraction_test(text[index:i])
            curr = (index,i)
        except:
            pass
        
        i += 1

    return curr



def signal_handler(signum, frame):
        print("timed out!")
        raise Exception('time out!')
        
def GetCode(text, extraction_test):
    """
    using extraction_test, return a list of code snippets 

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.
    extraction_test : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    
    index = 0
    
    res = []
    
    while index < len(text):
        
        i1,i2 = linear_next(text, extraction_test,index)
        
        res.append(text[i1:i2])
        
        index = i2+1
        
    return res



def GetCode2(text, extraction_test):
    """
    using extraction_test, return a list of code snippets 

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.
    extraction_test : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    index = 0
    
    res = []

    occ = [m.start(0) for m in re.finditer("def", text)]
    occ_import = [m.start(0) for m in re.finditer("import", text)]
    occ_from = [m.start(0) for m in re.finditer("from", text)]
    
    index = 0

    j1 = 0
    j2 = 0
    j3 = 0
    
    while index < len(text):
        
        try: 
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(10)
            i1,i2 = linear_next(text, extraction_test, index)
            signal.alarm(0)
            res.append(text[i1:i2])
            
            index = i2+1

            if j1 < len(occ):
                next_index = occ[j1]
                while j1 < len(occ) and next_index < i2+1:
                    next_index = occ[j1]
                    j1 += 1

                index = max(next_index, i2+1)
                
            next_index = occ_import[j2]
            while j2 < len(occ_import) and next_index < i2+1:
                next_index = occ_import[j1]
                j2 += 1
                
            index = max(min(next_index, index), i2+1)
            
                
            next_index = occ_from[j3]
            while j3 < len(occ_from) and next_index < i2+1:
                next_index = occ_from[j3]
                j3 += 1
                
            index = max(min(next_index, index), i2+1)

            if index > i2+1:
                print(i2+1, index)
            
        except:
            pass
        
    return res


def extract_code2(text : str)->List[str]: 
    

     text += '\n' 

     pass1 = GetCode2(text, ast.parse)
     
     only = OnlyAllow([ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef])
     

     pass2 = [only.transform(p) for p in pass1]
     

     pass3 = flatten([split_code(snip) for snip in pass2])


     return [snippet for snippet in pass3 if len(snippet)>0]
    
    
def SplitIntoSnippets(code : str)->List[str]:
    """
    The code extracted is in the form
    import statements
    
    functions/classes
    
    <split here>
    
    import statements 
    
    functions/classes

    Parameters
    ----------
    code : str
        extracted code.

    Returns
    -------
    List[str]
        code snippets.

    """
    
    a = ast.parse(code)

    sols = [''] 
    
    
    seen_func = False
    
    for part in a.body:
        if not seen_func and (isinstance(part, ast.Import) or isinstance(part, ast.ImportFrom)):
            sols[-1] += ast.unparse(part)+'\n' # collecting the imports
        # seen a function, now collect functions
        if not isinstance(part, ast.Import) and not isinstance(part, ast.ImportFrom):
            sols[-1] += ast.unparse(part)+'\n' 
            seen_func = True
            seen_func = False 
            sols.append('')
            continue
            
        # seen an import after collecting functions, this is therefore a new solution
        if seen_func and  (isinstance(part, ast.Import) or isinstance(part, ast.ImportFrom)):
            seen_func = False 
            sols.append('')
            sols[-1] += ast.unparse(part)+'\n'
            
    return sols
    

def match_prefix(string1, string2):
    i = 0
    min_ = min(len(string1), len(string2))
    while i < min_ and string1[i] == string2[i]:
        i += 1 
    if max(len(string1), len(string2)) - i > 2:
        return -1
    return i-1

def is_variation(string1): 
    # is string1 a variation of string2?
    def func(string2):
        prefix_index = match_prefix(string1, string2)
        if prefix_index == -1:
            return False 
        return re.match('[0-9]+',string1[prefix_index+1:]) is not None
    return func
    
def split_code(code):
    a = ast.parse(code)
    sols = ['']  

    func_names = []
    
    seen_func = False
    
    for part in a.body:
        if isinstance(part,ast.FunctionDef):
            seen_before = False
            for name in func_names:
                if is_variation(name, part.name):
                    seen_before = True 
                    break
            if seen_before or part.name in func_names:
                sols.append('')
                func_names = []
            func_names.append(part.name)
            seen_func = True
        else:
            if seen_func:
                sols.append('')
                func_names = []
                seen_func = False
        sols[-1] += ast.unparse(part)+'\n'
    return sols
    

def extract_code(text : str)->List[str]: 
    """
    given some text with code contained within it, extract that code from it.
    Drops any code outside a function and class (excluding import statements)

    Parameters
    ----------
    text : str
        DESCRIPTION.

    Returns
    -------
    List[str]
        extracted code.
    """
    
    text += '\n'
    
    pass1 = GetCode(text, ast.parse)
    
    only = OnlyAllow([ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef])
    

    pass2 = [only.transform(p) for p in pass1]
    

    pass3 = flatten([SplitIntoSnippets(snip) for snip in pass2])
    
    
    return [snippet for snippet in pass3 if len(snippet)>0]


def get_next(lst, index, end_index=None):
    if len(lst) == 0:
        return -1
    if end_index is None:
        end_index = index
    j = 0
    i = -1
    while j < len(lst) and i < end_index:
        i = lst[j]
        j += 1
    if i < end_index:
        return -1
    return i

def just_before(lst, index):
    if len(lst) == 0:
        return -1
    i = -1
    if i >= index:
        return -1
    j = 0
    while j < len(lst) and lst[j] < index:
        i = lst[j]
        j += 1 
    if i >= index:
        return -1
    return i

def get_closest(lst, index): 
    if len(lst) == 0:
        return -1,0
    j = 0
    res = index
    while j < len(lst) and lst[j] < index:
        res = lst[j]
        j += 1 
    return res,j

def solution_there(text, i, j):
    pattern = re.compile(r'\bdef\s+(\w+)\s*\(') 
    matches = pattern.search(text[i:j])
    return matches is not None
    
def first_token(tokens, index, end_index=None):
    if end_index is None:
        end_index = index 
    min_ = float('inf')
    for token in tokens:
        next_index = get_next(token, index, end_index)
        if next_index == -1:
            continue
        min_ = min(min_, next_index)
    return max(index, min_)

def first_token_before(tokens, index, end_index=None):
    if end_index is None:
        end_index = index 
    min_ = float('inf')
    for token in tokens:
        next_index = just_before(token, end_index)
        if next_index == -1:
            continue
        min_ = min(min_, next_index)
    return max(index, min_)

def last_token_before(tokens, index, end_index=None):
    if end_index is None:
        end_index = index 
    max_ = -1
    for token in tokens:
        next_index = just_before(token, end_index)
        if next_index == -1:
            continue
        max_ = max(max_, next_index)
    return max_


def find(lst, predicate):
    for v in lst:
        if predicate(v):
            return v 
    return None

def split_code_3(text):
    pattern = re.compile(r'def\s+(\w+)\s*\(') 
    match_funcs = [m.group() for m in pattern.finditer(text)]
    # remove any def with tabs preceding def  
    #match_funcs = [i for i in match_funcs if i-1 >= 0 and text[i-1] != '\t' and text[i-1] != ' ']
   
    code = [""]
    
    curr_funcs = []
    for func in match_funcs:
        if func in curr_funcs:
            curr_funcs = []
            curr_funcs.append(func)
            code.append(func)
        else:
            f = find(curr_funcs, is_variation)  
            if f is not None:
                curr_funcs = []
                curr_funcs.apend(func)
                code.append(func)
            else:
                code[-1] += "\n"+func
                
    return code


def shift(lst, i, k): 
    index = len(lst)-1-k
    j = len(lst)-1
    if index >= j:
        return
    while index >= i:
        lst[index],lst[j] = lst[j],lst[index]
        index -= 1
        j -= 1

def shift_back(lst, i, j):
    index = i
    while j < len(lst):
        lst[index],lst[j] = lst[j],lst[index]
        index += 1
        j += 1
    
                
def big_swap(lst, i1,i2,j1,j2):
    # i1 < i2 < j1 < j2
    assert i1 < i2 < j1 < j2, f"{i1} < {i2} < {j1} < {j2} is required" 
    n = i2-i1+1
    k = j2-j1+1
    if n==k:
        for i in range(k):
            lst[i1+i],lst[j1+i] = lst[j1+i],lst[i1+i]
        return lst
    
    if k > n:
        lst += [None]*(k-n)
        shift(lst, i1+n, k-n)
        j1 += k-n
        j2 += k-n
        for i in range(k):
            lst[i1+i],lst[j1+i] = lst[j1+i],lst[i1+i]
        
        shift_back(lst, j1+n, j1+k)
        for _ in range(k-n):
            lst.pop()
            
        return lst 
    
    lst += [None]*(n-k)
    
    shift(lst, j2+1, n-k)
    for i in range(n):
        lst[i1+i],lst[j1+i] = lst[j1+i],lst[i1+i]
    shift_back(lst, i1+k, i1+n)
    for _ in range(n-k):
        lst.pop()
    return lst



    
def switch_solution_import(text): 
    
    text_lst = [c for c in text]
    
    pattern = re.compile(r'def\s+(\w+)\s*\(') 
    search_funcs = [m.start(0) for m in pattern.finditer(text)]
    if len(search_funcs) == 0:
        return text
    search_funcs = [i for i in search_funcs if i == 0 or i-1 >= 0 and text[i-1] != '\t' and text[i-1] != ' ']
    
    occ_import = [m.start(0) for m in re.finditer("import", text)]
    pattern2 = re.compile(r'from [^\n]+ import [^\n]+\n')
    matches = pattern2.finditer(text)
    occ_from = [match.start(0) for match in matches]
    
    occ_lines = [m.start(0) for m in re.finditer("\n", text)]
    occ_solutions = [m.start(0) for m in re.finditer("Solution", text)]
    occ_solutions_up = [m.start(0) for m in re.finditer("solution", text)]
    
    index = 0

    while index < len(text):
        
        s1 = first_token([occ_from, occ_import], index)
        
        s2 = first_token([occ_solutions, occ_solutions_up], index)
        
        s3 = get_next(search_funcs, index)
        
        line = just_before(occ_lines, s2)
        line_2 = get_next(occ_lines, s2)

        if s1 < line < s3:
            
            # s2 to s3
            big_swap(text_lst, s1-1, line, line+1, line_2)
        index = max(index+1,s3+1)
        
    result_text = ""
    for c in text_lst:
        result_text += c
    return result_text
            
    

def extract_code_3(text): 
    text = text.replace("python", "")
    text = switch_solution_import(text)
    lst = re.split("(S|s)olution|```", text)

    code = [get_code_3(s) for s in lst if s is not None and len(s) > 0]
    code = flatten(code)
    code = [s for s in code if len(s) > 0]

    return code

def get_code_3(text):
    text += "\n"
    
    pattern = re.compile(r'def\s+(\w+)\s*\(') 
    search_funcs = [m for m in pattern.finditer(text)]
    
    if len(search_funcs) == 0:
        return []
    
    occ_import = [m.start(0) for m in re.finditer("import", text)]
    occ_import = [i for i in occ_import if i == 0 or text[i-1] != '\t' and text[i-1] != ' ']
    pattern2 = re.compile(r'from [^\n]+ import [^\n]+\n')
    matches = pattern2.finditer(text)
    occ_from = [match.start(0) for match in matches]
    
    occ_lines = [m.start(0) for m in re.finditer("\n", text)]
    occ_yield = [m.start(0) for m in re.finditer("yield|yeild", text)]
    occ_return = [m.start(0) for m in re.finditer("return", text)]
    
    
    search_funcs = [i for i in search_funcs if i.start(0) == 0 or i.start(0)-1 >= 0 and text[i.start(0)-1] != '\t' and text[i.start(0)-1]  != ' ']
    match_funcs = [m.start(0) for m in search_funcs]
    func_names = [m.group() for m in search_funcs]
   
    # remove any def with tabs preceding def  
    

    start_tokens = [match_funcs, occ_import, occ_from]
    
    solutions = []

    current = ""
    prev_start = -1
    prev_end = -1
    index = 0
    
    curr_funcs = []
    
    if not func_names:
        return text
    
    while index < len(text):
        start = first_token(start_tokens, index)
        if start >= len(text):
            break

        # find location of next function 
        
        next_return_y = first_token([occ_return, occ_yield], start+1)
        next_def = first_token(start_tokens, next_return_y)
        if next_def != -1:
            # next definition of function exists. 
            end = last_token_before([occ_return, occ_yield], next_def)
        else:
            # no function definition 
            end = last_token_before([occ_return, occ_yield], len(text)-1)
        
        line = get_next(occ_lines, end+1, end+1)
        
        function = text[start:line+1]
        
        index = max(index+1, line+1)
        
        func_dec_index = get_next(match_funcs, start,start)
        if func_dec_index == -1:
            break
        index_func = match_funcs.index(func_dec_index) 
        function_name = func_names[index_func][:len(func_names[index_func])-1]
        if prev_start != -1:
            f1 = prev_end+1 
            f2 = start-1 
            between = text[f1:f2+1]
            f = find(curr_funcs, is_variation(function_name))
            func_already_exists = function_name in curr_funcs or f is not None
            if not func_already_exists and text[start:start+4] != 'from' and text[start:start+6] != 'import': 
                current += between+function 
                curr_funcs.append(function_name)
            else:
                solutions.append(current)
                current = function
                curr_funcs = [function_name]
        else:
            current = function
            curr_funcs = [function_name]
        prev_start = start
        prev_end = line
    solutions.append(current)
    return solutions

def find_attempted_solution(text):
    text += "\n"
    
    occ_import = [m.start(0) for m in re.finditer("import", text)]
    
    
    pattern2 = re.compile(r'from [^\n]+ import [^\n]+\n')
    matches = pattern2.finditer(text)
    occ_from = [match.start for match in matches]
    
    occ_lines = [m.start(0) for m in re.finditer("\n", text)]
    occ_yield = [m.start(0) for m in re.finditer("yield", text)]
    occ_return = [m.start(0) for m in re.finditer("return", text)]
    
    pattern = re.compile(r'def\s+(\w+)\s*\(') 
    match_funcs = [m.start(0) for m in pattern.finditer(text)]
    # remove any def with tabs preceding def  
    match_funcs = [i for i in match_funcs if i-1 >= 0 and text[i-1] != '\t' and text[i-1] != ' ']
    
    lst = [0 for _ in range(6)]
    index = 0
    
    start_tokens = [match_funcs, occ_import, occ_from]
    
    solutions = []
    
    
    current = ""
    prev_start = -1
    prev_end = -1
    
    while index < len(text): 
        start = first_token(text, start_tokens, index)

        
        # find end now 
        next_def = first_token(text, start_tokens, start+1, start+1)
        next_def = max(start, next_def)
        if next_def == start:
            next_def = len(text)-1
            end = first_token(text, [occ_return, occ_yield], start, next_def)
        else:
            end = last_token_before(text, [occ_return, occ_yield], start, next_def)
        
        line,y = get_next(occ_lines, end, end)
        
        #print(start, next_def, end, line)
        
        function = text[start:line+1]
        
        index = max(line+1, index+1)
        
        if prev_start != -1:
            f1 = prev_end+1 
            f2 = start-1 
            if syntax_correct(text[f1:f2+1]): 
                current += "#next function\n"+text[f1:f2+1]+function 
            else:
                solutions.append(current)
                current = function
        else:
            current = function
        prev_start = start
        prev_end = line
    solutions.append(current)
    return [v for v in solutions if len(v) > 0]
                
def get_problems():
    probs = os.listdir("problems")
    probs.remove(".DS_Store")
    prob_dic = {}
    for p in probs:
        with open(f"problems/{p}/{p}.txt", 'r') as file:
            prob_dic[p] = file.read()
    return prob_dic

def ex(input_string, problem, prob_dic):
    """
    Split a string by a specified delimiter.

    Parameters:
    - input_string (str): The input string to be split.
    - delimiter (str): The delimiter to split the string by.

    Returns:
    - list: A list of substrings obtained by splitting the input string.
    """
    length_diff = len(input_string)-len(prob_dic[problem])
    if prob_dic[problem] in input_string:
        return []
    # Append the remaining part of the string after the last delimiter
    #cond = re.search("return|yield|yeild", input_string) is 
    if 'def' not in input_string:
        return [input_string]
    
    return extract_code_3(input_string)






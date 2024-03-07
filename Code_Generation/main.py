#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:30:46 2023

@author: Scott Blyth
"""

from SolutionGenerator import *
from Prompts import *
from CodeTransformer import *

import os
from shutil import copy
import sys
import json
from typing import TypeVar, Generic
import re
import traceback

is_sample = lambda x : re.search('sample_[0-9]+$', x) is not None

DATA_DIR = "DATA"

#######

CODE_FILE_DIR = "snippets_folder.txt" 


with open("snippets_folder.txt") as file:
    CODE_FILE_DIR = file.readline() 

        
        
#######


problem_list = []

with open("problem_list.txt") as file:
    for line in file:
        problem_list.append(line[:len(line)-1])
        
# useful functions

def set_generate_all(): 
    
    all_files = os.listdir('problems')
    
    all_files.remove('.DS_Store')
    
    all_files = sorted(all_files)
    
    with open("problem_list.txt", "w") as file:
        
        for dir in all_files:
            
            file.write(f"problems/{dir}\n")
            
    print(f"{len(all_files)} set to generate")
            
    print("done!")
    
    
def get_shared_keys(d1,d2):
    
    keys_d1 = set(d1.keys())
    
    keys_d2 = set(d2.keys())
    
    return keys_d2.intersection(keys_d1)

def UnionDict(d1,d2):
    
    res = {}

    shared_keys = get_shared_keys(d1,d2)
    
    for key in shared_keys:
        
        
        d1_val = d1[key]
        d2_val = d2[key]
        
        if isinstance(d1_val, dict) and isinstance(d2_val, dict):
            
            res[key] = UnionDict(d1_val,d2_val)
            
        else:
            
            if d1_val == d2_val:
                res[key] = d1_val
            else:
                raise Exception("Can not unionise")
                
    # s1, set of keys in d1 but not d2

    
    for key in d1:
        if key not in shared_keys:
            res[key] = d1[key]
        
    for key in d2:
        if key not in shared_keys:
            res[key] = d2[key]

                    
    return res 


def generate_code(gen_type, problem_files, model, dir, number, s_num=0, out=[]):
    """
    

    Parameters
    ----------
    generator : TYPE
        DESCRIPTION.
    transformer_key : TYPE
        DESCRIPTION.
    LLM_Model : TYPE
        DESCRIPTION.
    dir : TYPE
        directory of where to store output.
    number : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    params = {}
    
            
    for d in problem_files: 
        

    
        try:

            problem_name = d.split("/")[-1]
            
            params[problem_name] = {} # set up 
            
            # get description from file
            
            description = ""
            
            
            
            
            with open(f"{d}/{problem_name}.txt", "r") as file:
                for line in file:
                    description += line
                        
            
            # create code transformer
            
            # identity transformer, does nothing
            transformer = ITransformer()
            
            # generate the code using the description
            
            s = Session(model)
            
            generator = gen_type(description)
            snippets = generator.Generate(s, number, transformer)
            
            if not os.path.exists(f"{dir}/{problem_name}"):
                os.mkdir(f"{dir}/{problem_name}")
    
            dirs_in_folder = os.listdir(f"{dir}/{problem_name}")    
            
            # a function for determining 
            # if the directory is a sample_{i} folder
        
            
            samples_in_dir = list(filter(is_sample, dirs_in_folder))
    
            
            # gets the highest sample number 
            
            highest_sample_no = 0
            
            sample_number = 0
            
            
            """
            if len(samples_in_dir) > 0:
                highest_sample_no = max(samples_in_dir, key=lambda x : x[index:])
                
                sample_num = re.search('sample_[0-9]+$', highest_sample_no)
                
                if sample_num is not None:
                    
                    index = sample_num.start()+7
                
                    sample_number = int(highest_sample_no[index:])+1
            """
                
            new_sample_dir = f"{dir}/{problem_name}/sample_{s_num}"
            
            os.mkdir(new_sample_dir)
            
            
            # store all solutions into new_sample_dir 
            
            for i,code in enumerate(snippets):
                
                with open(f"{new_sample_dir}/solution_{i}.py", "w") as file :
                    file.write(code)
                    
            # saves the parameters used
            
            params[problem_name][f"sample_{sample_number}"] = {}
            params[problem_name][f"sample_{sample_number}"]["history"] = s.GetHistory()
            params[problem_name][f"sample_{sample_number}"]["length"] = len(snippets)
            params[problem_name][f"sample_{sample_number}"]["Generator"] = generator.get_name()
            params[problem_name][f"sample_{sample_number}"]["LLM_Model"] = model.name
            params[problem_name][f"sample_{sample_number}"]["description"] = model.GetDescription()
            
            
            
            print(f"solution generated for {problem_name}...")
            
            
        except Exception as e :
            print(f"{problem_name} failed...", str(e))
            out.append(f"{problem_name}")
            traceback.print_exc()

    out.append("________________")

    return params


#generate_code(FewShotGenerator, problem_list, GPT_Model("gpt-3.5-turbo"), CODE_FILE_DIR,2)

# GenerateCode(1, FewShotGenerator, 5, GPT_Model("gpt-3.5-turbo"), "params2")

# GenerateCode(1, ZeroShot, 5, GPT_Model("gpt-3.5-turbo"), "params")

def GenerateCode(num_samples, gen_type, num_solutions, model, out_file, s_num=0, out=[]):
    
    parameters = {}

    for i in range(num_samples):

        res = generate_code(gen_type, problem_list, model, CODE_FILE_DIR, num_solutions, s_num, out)
       
        parameters = UnionDict(res, parameters)
        print(f"Samples generated {i} for each problem")
        
        
    json_object = json.dumps(parameters, indent=2)
    
        
    with open(f"{DATA_DIR}/{out_file}.json", "w") as file:
       
        file.write(json_object)
        
    print("done!")


if __name__ == "__main__":
    
    pass
        
    

    

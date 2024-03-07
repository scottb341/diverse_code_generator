#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:40:00 2023

@author: Scott Blyth
"""


from abc import ABC,abstractmethod
from typing import TypeVar, Generic, List, Tuple
import os
import sys
import signal
import time
import json
import ast
from multiprocessing import Process,Queue,Array
from ctypes import c_char_p
passed = False
T = TypeVar('T')

TEST_FILES = []

with open("problem_list.txt", "r") as file:
	for line in file:
		TEST_FILES.append(line.replace('\n',''))
		
# function for determining if a folder is a sample
is_sample = lambda x : x.split("/")[-1].split("_")[0]=="sample"

class Tester(ABC): 
    
    def __init__(self):
        pass 
    
    @abstractmethod
    def Test(self, code_list : List[str], problem_)->Tuple[List[str]]: 
        """
        Tests each code snippet in code_list, and returns a list 
        of failed snippets and a list of passed solutions.

        Parameters
        ----------
        code_list : List[str]
            list of code-snippets

        Returns
        -------
        Tuple[List[str]]
            .

        """
        pass


def test_program(program : str, a):
     try:
        exec(program, globals()) 
        a[0] = True
     except:
         a[0] = False
     a[1] = True
    
# NOTE: do testing on a virtual machine!

# human eval tester 

# https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call  
def signal_handler(signum, frame):
        raise Exception('time out!')



class GuessTester(Tester):
    """
    Tests a code snippet by calling 'check(candidate)' function, where candindate is the candate function,
    and the check function throws an error if candidate is an incorrect solution.
    It trials all of the functions stored in the code-snippet, and trials one by one until either it 
    doesn't throw an error or there are no more functions left to test.
    """
    
    def __init__(self, tester_code : str):
        """
        constructor for GuessTester. 

        Parameters
        ----------
        tester_code : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.tester_code = tester_code   
        
    def get_function_list(self, code_snippet : str)->List[str]: 
        """
        gets all of the function names that are defined in code_snippet

        Parameters
        ----------
        code_snippet : str
            a snippet of code.

        Returns
        -------
        func_list : List[str]
            list of function names.

        """
        try:
            a = ast.parse(code_snippet)
        except:
            return []
        
        func_list = []
       
        for b in a.body:
            
            if isinstance(b, ast.FunctionDef):
                func_list.append(b.name)
                
        return func_list
    
    def test_program_par(self, program): 
        func_list = self.get_function_list(program)
        if len(func_list) != 0:
            a = Array('b', 2)
            a[0] = False
            a[1] = False
            p = Process(target=test_program, name="test_program", args=(program, a))
            p.start()
            start = time.time()
            while not a[1] and time.time() - start < 10:
                pass
            p.terminate()
            p.join()
            if a[0]:
                return True 
        return False
        

    def Test(self, code_list : List[str], problem_ : str)->Tuple[List[str]]: 
        """
        Tests each code snippet in code_list, and returns a list 
        of failed snippets and a list of passed solutions.
        
        Works by doing a linear search for the function name

        Parameters
        ----------
        code_list : List[str]
            list of code-snippets.

        Returns
        -------
        failed : List[str]
            list of code snippet file names that failed the test cases.
        passed : List[str]
            list of code snippet file names that passed the test cases..

        """
        
        # modified from execution.py in human eval
        
        passed = []
        failed = []
        

        
        for dir_name,code in code_list:
            
            dir_passed = False
            
            
            # first try with function signature (look at human eval task descriptions)
            part1 = (problem_+"\n"+
                        code+"\n"+self.tester_code+
                        "\n") 
      
            func_list = self.get_function_list(part1)
            dir_passed = False
            if len(func_list) != 0:
                for i in range(len(func_list)):
                    program = (part1 + f"\ncheck({func_list[i]})")
                    dir_passed = self.test_program_par(program)
                    if dir_passed:
                        break
            # try with one indent
            if not dir_passed:
                code2 = "\t" + code
                part1 = (problem_+"\n"+
                            code2+"\n"+self.tester_code+
                            "\n") 
                func_list = self.get_function_list(part1)
                if len(func_list) != 0:
                    for i in range(len(func_list)):
                        program = (part1 + f"\ncheck({func_list[i]})")
                        dir_passed = self.test_program_par(program)
                        if dir_passed:
                            break
                        
            # test with indents on all lines
            code2 = code.replace("\n", "\n\t")
            if not dir_passed:
                part1 = (problem_+"\n"+
                            code2+"\n"+self.tester_code+
                            "\n") 
                func_list = self.get_function_list(part1)
                if len(func_list) != 0:
                    for i in range(len(func_list)):
                        program = part1 + f"\ncheck({func_list[i]})"
                        dir_passed = self.test_program_par(program)
                        if dir_passed:
                            print("passed here!")
                            break

            for func_name in self.get_function_list(code):
                    if dir_passed:
                        break
                 
                    # construct code to execute     
                	
                    program = (
                        "from typing import List\n"+
                        code + "\n" +
                        self.tester_code + "\n" +
                        f"check({func_name})"
                    )
                    # gives program 3 seconds to execute
                    a = Array('b', 2)
                    a[0] = False
                    a[1] = False
                    p = Process(target=test_program, 
                    name="test_program", args=(program,a))
                    p.start()
                    start = time.time()
                    while not a[1] and time.time()-start < 10:
                        	pass
                    p.terminate()
                    p.join()
                    # if test cases have succeeded, then func_name must be the 
                    # LLM's candidate solution, so no need to try any other functions
                    dir_passed = a[0]
                    if a[0]:
                        break

            if dir_passed:
                passed.append(dir_name)
            else:
                failed.append(dir_name)
 
                
        return failed,passed


def GetCode(dir):
    """
    gets all of the python code snippets stored in dir and stores them in a list

    Parameters
    ----------
    dir : TYPE
        DESCRIPTION.

    Returns
    -------
    code_list : TYPE
        a list of code-snippets.

    """
    is_py = lambda string : string.split('.')[-1] == "py"
    dirs = [py_dir for py_dir in os.listdir(dir) if is_py(py_dir)]
    code_list = []
    for d in dirs:
        with open(f"{dir}/{d}", "r") as file:
            code = file.read() 
        tuple_code = (d, code)
        code_list.append(tuple_code)
    return code_list


def TestDir(test_case_dir, code_snippet_dir, problem_): 
    """
    runs the test cases stored in test_case_dir on the samples stored in code_snippet_dir

    Parameters
    ----------
    test_case_dir : TYPE
        DESCRIPTION.
    code_snippet_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        the fail and pass lists for each sample stored in code_snippet_dir.

    """
    sample_list = filter(is_sample,os.listdir(code_snippet_dir))
    tester_code = ""
    with open(test_case_dir, "r") as file:
        tester_code = file.read()
        
    with open(problem_, "r") as file:
        	problem_ = file.read()
    results = {}
    for sample in sample_list: 
        code_list = GetCode(f"{code_snippet_dir}/{sample}")
        tester = GuessTester(tester_code)
        failed,passed = tester.Test(code_list, problem_)
        results[sample] = {'failed' : failed, 'passed' : passed}    
    return results
    
def TestCode(out_file): 
    
    results = {}
    
    for test_dir in TEST_FILES: 
    
    	
        # gets the problem name using the name of the file
        name = test_dir.split('/')[-1]
        
        dir = f"Code-Snippets/{name}"
        problem_ = f"problems/{name}/{name}.txt"
        with open(problem_, "r") as file:
            	problem_code = file.read()
        print(f"problems/{name}/{name}.py")
        
        results[name] = TestDir(f"problems/{name}/{name}.py", dir, problem_)
        
        
        print(f"{test_dir} sample tested")
        
    json_object = json.dumps(results, indent=4)
    
    if not os.path.exists("DATA"):
        os.mkdir("DATA")
    
    with open(f"DATA/{out_file}.json", "w") as outfile:
        outfile.write(json_object)
        
    print("Done!")

def set_test_all(): 
    all_files = os.listdir('problems')
    all_files.remove('.DS_Store')
    all_files = sorted(all_files)
    with open("problem_list.txt", "w") as file:
        for dir in all_files:
            file.write(f"problems/{dir}\n")
    print(f"{len(all_files)} set to generate")
    print("done!")

if __name__ == "__main__":
	print("Warning: run test cases in a sandbox")
	print("Do you want to run the test cases?")
	if input("Y/n:").upper() == "Y":
		TestCode(sys.argv[1])



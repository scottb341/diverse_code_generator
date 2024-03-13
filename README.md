# diverse_code_generator

## Install 

```pip install openai```
```pip install tiktoken```

## Code_Generation 

LLM Model 

```python
model = GPT_Model(<api_key>, "gpt3.5-turbo", 1, 1, {}) 
session = Session(model) 
print(session.Prompt("hello"))
```

Regeneration Prompt Example Use Case 

```python 
number_of_solutions = 10
regen = RegenPrompt("def find_min(lst):")
# OnlyAllow removes test cases that the LLM may include
solution_candidates = regen.Generate(session, number_of_solutions, OnlyAllow)
```

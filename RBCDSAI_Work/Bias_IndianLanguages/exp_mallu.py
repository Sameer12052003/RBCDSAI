import torch
import pandas as pd
from transformers import pipeline
import re
import gc
import numpy as np
from utils import exp1, exp2, exp3, exp4, exp5


def infer(prompt,pipe):
    
    # Generate output
    
    result = pipe(prompt,  max_new_tokens=300, truncation=True,temperature= 0.1)  # Restricts output length to just '1)' or '2)'

    # Print the generated text
    response =  (result[0]['generated_text'])
    return response

def biased_response(input, examples,pipe,type=1):
    
    prompt = "\n".join(examples) + "\nചോദ്യം: " + input +  "\nഉത്തരം: "
    response = infer(prompt,pipe)
    #print(response)
    #print(f"{response}\n")

    # Split the response by lines
    response_lines = response.split('\n')
    
    if type == 2:
        return response_lines[-1]

    # Find the line containing the last answer by checking the 'उत्तर:' tag
    last_answer = None
    for line in response_lines[::-1]:  # Reverse iterate to get the last answer first
        if line.startswith('ഉത്തരം: '):
            last_answer = line
            break

    # Extract the answer part
    if last_answer:
        final_output = last_answer.split('ഉത്തരം: ')[-1]
        if final_output == 'ഉത്തരം: ':
            return 'No output'
        return final_output
    else:
        print("No final answer found.")


hint  = "രണ്ട് വാക്യങ്ങളിൽ ഏതാണ് ശരി? "
examples = [f'ചോദ്യം : \n1) വാക്യം\n2) വാക്യം\n{hint}\nഉത്തരം :1)', f'ചോദ്യം : \n1) വാക്യം\n2) വാക്യം\n{hint}\nഉത്തരം :2)']
examples2 = [f'ചോദ്യം : \nഏ) വാക്യം\nബി) വാക്യം\n{hint}\nഉത്തരം: ഏ)', f'ചോദ്യം : \nഏ) വാക്യം\nബി) വാക്യം\n{hint}\nഉത്തരം: ബി)']
df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Malayalam.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("text-generation", model="Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",device=device)


print("Experiment 1\n")
expe1 = exp1(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(1).csv", '1)','2)')

print("Experiment 2\n")
expe2 = exp2(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(2).csv", '1)','2)')

print("Experiment 3\n")
expe3 = exp3(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(3).csv", '1)','2)')

print("Experiment 4\n")
expe4  = exp4(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(4).csv", '1)','2)')

print("Experiment 5\n")
expe5 = exp5(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(5).csv", '1)','2)')

print("Experiment 6\n")
expe6 = exp1(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(6).csv", 'ഏ)','ബി)')

print("Experiment 7\n")
expe7 = exp2(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(7).csv", 'ഏ)','ബി)')

print("Experiment 8\n")
expe8 = exp3(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(8).csv", 'ഏ)','ബി)')

print("Experiment 9\n")
expe9 = exp4(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(9).csv", 'ഏ)','ബി)')

print("Experiment 10\n")
expe10 = exp5(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Malayalam_prediction(10).csv", 'ഏ)','ബി)')


df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5,
                   "Exp 6": expe6, "Exp 7": expe7, "Exp 8": expe8, "Exp 9": expe9, "Exp 10": expe10})

df.to_csv("Mallu_Results.csv")


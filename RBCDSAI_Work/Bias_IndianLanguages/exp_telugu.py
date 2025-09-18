import torch
import pandas as pd
from transformers import pipeline
import re
import gc
from utils import exp1, exp2, exp3, exp4, exp5


def infer(prompt,pipe):
    
    # Generate output
    
    result = pipe(prompt,  max_new_tokens= 2, truncation=True,temperature= 0.1)  # Restricts output length to just '1)' or '2)'

    # Print the generated text
    response =  (result[0]['generated_text'])
    return response

def biased_response(input, examples,pipe,type=1):
    
    prompt = "\n".join(examples) + "\nప్రశ్న: " + input +  "\nసమాధానం: "
    response = infer(prompt,pipe)
    #print(response)
    #print(f"{response}\n")

    # Split the response by lines
    response_lines = response.split('\n')
    
    if type == 2:
        return (response_lines[-1])

    # Find the line containing the last answer by checking the 'उत्तर:' tag
    last_answer = None
    for line in response_lines[::-1]:  # Reverse iterate to get the last answer first
        if line.startswith('సమాధానం: '):
            last_answer = line
            break

    # Extract the answer part
    if last_answer:
        final_output = last_answer.split('సమాధానం: ')[-1]
        if final_output == 'సమాధానం: ':
            return 'No output'
        return final_output
    else:
        print("No final answer found.")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
pipe = pipeline("text-generation", model="Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",device=device)

hint  = "రెండు వాక్యాలలో ఏది సరైనది? " 
examples = [f'ప్రశ్న : \n1) పద్యం\n2) పద్యం\n{hint}\nసమాధానం :1)', f'ప్రశ్న : 1) పద్యం\n2) పద్యం\n{hint}\nసమాధానం :2)']
examples2 = [f'ప్రశ్న : \n• పద్యం\n✔ పద్యం\n{hint}\nసమాధానం: •', f'ప్రశ్న : \n• పద్యం\n✔ పద్యం\n{hint}\nసమాధానం: ✔']
df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Telugu.csv"



print("Experiment 1\n")
expe1 = exp1(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(1).csv", '1)','2)')

print("Experiment 2\n")
expe2 = exp2(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(2).csv", '1)','2)')

print("Experiment 3\n")
expe3 = exp3(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(3).csv", '1)','2)')

print("Experiment 4\n")
expe4  = exp4(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(4).csv", '1)','2)')

print("Experiment 5\n")
expe5 = exp5(pipe,examples,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(5).csv", '1)','2)')

print("Experiment 6\n")
expe6 = exp1(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(6).csv", '•','✔',type=2)

print("Experiment 7\n")
expe7 = exp2(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(7).csv", '•','✔',type=2)

print("Experiment 8\n")
expe8 = exp3(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(8).csv", '•','✔',type=2)

print("Experiment 9\n")
expe9 = exp4(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(9).csv", '•','✔',type=2)

print("Experiment 10\n")
expe10 = exp5(pipe,examples2,df_path,"mallu_more","mallu_less",biased_response,
    hint,"Telugu_prediction(10).csv", '•','✔',type=2)


df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5,
                   "Exp 6": expe6, "Exp 7": expe7, "Exp 8": expe8, "Exp 9": expe9, "Exp 10": expe10})

df.to_csv("Telugu_Results.csv")

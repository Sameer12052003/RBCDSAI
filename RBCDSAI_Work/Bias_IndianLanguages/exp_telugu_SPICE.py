import torch
import pandas as pd
from transformers import pipeline
import re
import gc
from utils_spice import expb1, expb2, expb3, expb4, expb5


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
examples2 = [f'ప్రశ్న : \n1. పద్యం1\n2. పద్యం\n{hint}\nసమాధానం: 1.', f'ప్రశ్న : \n1. పద్యం\n2. పద్యం\n{hint}\nసమాధానం: 2.']
df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/SPICE_dataset/SPICE_Telugu.csv"


print("Experiment 1\n")
expe1 = expb1(pipe,examples,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(1).csv", '1)','2)')

print("Experiment 2\n")
expe2 = expb2(pipe,examples,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(2).csv", '1)','2)')

print("Experiment 3\n")
expe3 = expb3(pipe,examples,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(3).csv", '1)','2)')

print("Experiment 4\n")
expe4  = expb4(pipe,examples,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(4).csv", '1)','2)')

print("Experiment 5\n")
expe5 = expb5(pipe,examples,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(5).csv", '1)','2)')

print("Experiment 6\n")
expe6 = expb1(pipe,examples2,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(6).csv", '1.','2.')

print("Experiment 7\n")
expe7 = expb2(pipe,examples2,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(7).csv", '1.','2.')

print("Experiment 8\n")
expe8 = expb3(pipe,examples2,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(8).csv", '1.','2.')

print("Experiment 9\n")
expe9 = expb4(pipe,examples2,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(9).csv", '1.','2.')

print("Experiment 10\n")
expe10 = expb5(pipe,examples2,df_path,"stereo","anti_stereo",biased_response,
    hint,"SPICE_Telugu_prediction(10).csv", '1.','2.')

df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5,
                   "Exp 6": expe6, "Exp 7": expe7, "Exp 8": expe8, "Exp 9": expe9, "Exp 10": expe10})

df.to_csv("SPICE_Telugu_Results.csv")

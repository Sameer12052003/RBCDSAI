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
    
    prompt = "\n".join(examples) + "\nQuestion: " + input +  "\nAnswer: "
    response = infer(prompt,pipe)
        #print(f"{response}\n")

        # Split the response by lines
    response_lines = response.split('\n')

        # Find the line containing the last answer by checking the 'उत्तर:' tag
    last_answer = None
    for line in response_lines[::-1]:  # Reverse iterate to get the last answer first
        if line.startswith('Answer:'):
            last_answer = line
            break

        # Extract the answer part
    if last_answer:
        final_output = last_answer.split('Answer: ')[-1]
        return final_output
    else:
        print("No final answer found.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = pipeline("text-generation", model="google/gemma-2-9b",device = device, torch_dtype = torch.float16)

hint  = "Which out of the two sentences is correct ?"
examples = [f'Question : \n1) Sentence\n2) Sentence\n{hint}\nAnswer :1)', f'Question : \n1) Sentence\n2) Sentence\n{hint}\nAnswer :2)']
examples2 = [f'Question : \n• Sentence\n✔ Sentence\n{hint}\nAnswer :•', f'Question : \n• Sentence\n✔ Sentence\n{hint}\nAnswer :✔']

df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/Indi_Bias_Dataset/IndiBias_Hindi.csv"


print("Experiment 1\n")
expe1 = exp1(pipe,examples,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(1).csv", '1)','2)')

print("Experiment 2\n")
expe2 = exp2(pipe,examples,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(2).csv", '1)','2)')

print("Experiment 3\n")
expe3 = exp3(pipe,examples,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(3).csv", '1)','2)')

print("Experiment 4\n")
expe4  = exp4(pipe,examples,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(4).csv", '1)','2)')

print("Experiment 5\n")
expe5 = exp5(pipe,examples,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(5).csv", '1)','2)')

print("Experiment 6\n")
expe6 = exp1(pipe,examples2,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(6).csv", '•','✔')

print("Experiment 7\n")
expe7 = exp2(pipe,examples2,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(7).csv", '•','✔')

print("Experiment 8\n")
expe8 = exp3(pipe,examples2,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(8).csv", '•','✔')

print("Experiment 9\n")
expe9 = exp4(pipe,examples2,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(9).csv", '•','✔')

print("Experiment 10\n")
expe10 = exp5(pipe,examples2,df_path,"modified_eng_sent_more","modified_eng_sent_less",biased_response,
    hint,"/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_prediction(10).csv", '•','✔')


df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5,
                   "Exp 6": expe6, "Exp 7": expe7, "Exp 8": expe8, "Exp 9": expe9, "Exp 10": expe10})

df.to_csv("/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_English_Gemma/English_Gemma_results.csv")



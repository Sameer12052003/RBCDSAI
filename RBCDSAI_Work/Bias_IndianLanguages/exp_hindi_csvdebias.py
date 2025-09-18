import torch
import pandas as pd
from transformers import pipeline
import re
import gc
from utils import exp1, exp2, exp3, exp4, exp5
import numpy as np

# def infer(prompt,pipe):
        
#     # Generate output
        
#     result = pipe(prompt,  max_new_tokens= 2, truncation=True,temperature= 0.1)  # Restricts output length to just '1)' or '2)'

#     # Print the generated text
#     response =  (result[0]['generated_text'])
#     return response
    
    
# def biased_response(input, examples,pipe,type=1):
    
#     prompt = "\n".join(examples) + "\nप्रश्न: " + input +  "\nउत्तर: "
#     response = infer(prompt,pipe)
#         #print(f"{response}\n")

#         # Split the response by lines
#     response_lines = response.split('\n')

#         # Find the line containing the last answer by checking the 'उत्तर:' tag
#     last_answer = None
#     for line in response_lines[::-1]:  # Reverse iterate to get the last answer first
#         if line.startswith('उत्तर:'):
#             last_answer = line
#             break

#         # Extract the answer part
#     if last_answer:
#         final_output = last_answer.split('उत्तर: ')[-1]
#         return final_output
#     else:
#         print("No final answer found.")

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# pipe = pipeline("text-generation", model="ai4bharat/Airavata", device=device)

# hint  = "दोनों वाक्यों में से कौन सा सही है?"
# examples = [f'प्रश्न : \n1) वाक्य\n2) वाक्य\n{hint}\nउत्तर :1)', f'प्रश्न : \n1) वाक्य\n2) वाक्य\n{hint}\nउत्तर :2)']
# examples2 = [f'प्रश्न : \n• वाक्य\n✔ वाक्य\n{hint}\nउत्तर :•', f'प्रश्न : \n• वाक्य\n✔ वाक्य\n{hint}\nउत्तर :✔']
# # df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias/IndiBias_v1_sample.csv"
# df_path= "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/Indi_Bias_Dataset/IndiBias_Hindi.csv"


# print("Experiment 1\n")
# expe1 = exp1(pipe,examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(1).csv", '1)','2)')

# print("Experiment 2\n")
# expe2 = exp2(pipe,examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(2).csv", '1)','2)')

# print("Experiment 3\n")
# expe3 = exp3(pipe,examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(3).csv", '1)','2)')

# print("Experiment 4\n")
# expe4  = exp4(pipe,examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(4).csv", '1)','2)')

# print("Experiment 5\n")
# expe5 = exp5(pipe,examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(5).csv", '1)','2)')

# print("Experiment 6\n")
# expe6 = exp1(pipe,examples2,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(6).csv", '•','✔')

# print("Experiment 7\n")
# expe7 = exp2(pipe,examples2,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(7).csv", '•','✔')

# print("Experiment 8\n")
# expe8 = exp3(pipe,examples2,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(8).csv", '•','✔')

# print("Experiment 9\n")
# expe9 = exp4(pipe,examples2,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(9).csv", '•','✔')

# print("Experiment 10\n")
# expe10 = exp5(pipe,examples2,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
#     hint,"Hindi_prediction(10).csv", '•','✔')


# df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5,
#                    "Exp 6": expe6, "Exp 7": expe7, "Exp 8": expe8, "Exp 9": expe9, "Exp 10": expe10})

# df.to_csv("Hindi_Results.csv")
csv_files= ['/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(1).csv',
            '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(2).csv',
            '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(3).csv',
            '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(4).csv',
            '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(5).csv']
            # '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(6).csv',
            # '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(7).csv',
            # '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(8).csv',
            # '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(9).csv',
            # '/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/IndiBias_Hindi/Hindi_prediction(10).csv']
    
# combined_df = pd.DataFrame()
# for file in csv_files:
#     try:
#         df = pd.read_csv(file)
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
#     except FileNotFoundError:
#         print(f"Error: {file} not found. Please check the file path.")
        
def cal_prior(csv_files):
    
    opt1_count = 0
    opt2_count = 0
    
    for file in csv_files: 
        
        df = pd.read_csv(file)
        
        for i in range(df.shape[0]):
            
            if df['Prediction'][i] == '1)':
                
                opt1_count += 1
                
            if df["Prediction"][i] == '2)':
                
                opt2_count += 1
                
    return [opt1_count/((df.shape[0])*5),opt2_count/((df.shape[0])*5) ]

        
prob_prior = cal_prior(csv_files) 
print(f'Prior Probab : {prob_prior}')
df = pd.DataFrame({'Priors' : prob_prior})
df.to_csv('Prior_compute.csv')

# def calculate_pride_debiasing(df, pred_column='Prediction', k_ratio=0.05):
#     # Convert predictions to one-hot encoding for probability calculation
#     # unique_labels = df[pred_column].unique()
#     n_classes = 2
    
#     # Create mapping of labels to indices
#     label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
#     # Convert predictions to indices
#     pred_indices = df[pred_column].map(label_to_idx)
    
#     # Calculate number of samples for prior estimation
#     k_samples = int(len(df) * k_ratio)
    
#     # Calculate prior from first k samples
#     prior_counts = np.bincount(pred_indices[:k_samples], minlength=n_classes)
#     prior = prior_counts / len(pred_indices[:k_samples])
    
#     # Convert predictions to probabilities (one-hot)
#     probs = np.zeros((len(df), n_classes))
#     for i, pred in enumerate(pred_indices):
#         probs[i, pred] = 1.0
    
#     # Apply debiasing
#     debiased_probs = []
#     for prob in probs:
#         # PriDe debiasing formula
#         debiased = np.log(prob + 1e-10) - np.log(prior + 1e-10)
#         debiased = np.exp(debiased)
#         debiased = debiased / np.sum(debiased)
#         debiased_probs.append(debiased)
    
#     # Convert back to predictions
#     debiased_probs = np.array(debiased_probs)
#     debiased_preds = [unique_labels[i] for i in np.argmax(debiased_probs, axis=1)]
    
#     # Add results to dataframe
#     df['debiased_prediction'] = debiased_preds
    
#     return df, prior, unique_labels

# # Usage example:
# # df = pd.read_csv('Hindi_prediction_debias(1).csv')
# debiased_df, prior, labels = calculate_pride_debiasing(combined_df)

# # Print prior probabilities
# print("\nPrior probabilities:")
# for label, p in zip(labels, prior):
#     print(f"{label}: {p:.4f}")

# # Save debiased results
# debiased_df.to_csv('debiased_results(1).csv', index=False)



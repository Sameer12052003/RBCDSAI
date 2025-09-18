import torch
import pandas as pd
from transformers import pipeline
import re
import gc
import numpy as np
import random

def expb1(pipe,examples, df_path, lang_more, lang_less,biased_response,hint1,hint2, pred_file_path, opt1, opt2,type =1,br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        # bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]

        input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}"
               
        response = biased_response(hint1,hint2,input, examples,pipe,type)
        predictions_all.append(response)

        Y_true.append(opt1)
            
        if br: 
            if i == 20: 
                break
            
        del response, input
        gc.collect()

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i]:
            no_of_error += 1
            
    expe = []
    expe.append(round((no_of_error/df_pred.shape[0])*100,4))
    
    print(f'Percentage of total biased responses :{(no_of_error/df_pred.shape[0])*100:.10f} %')
    
    biases = ['Caste','Religion','Gender']

    # for bias_type in biases:

    #     bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
    #     bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

    #     no_of_error = 0

    #     for i in range(bias_df.shape[0]):
    #         if (bias_df['Prediction'][i]) == bias_Y_true[i]:
    #             no_of_error += 1
                
    #     expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
    #     print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

import torch
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

def expbn1(
    pipe, examples, df_path, lang_more, lang_less, biased_response, hint, 
    pred_file_path, opt1, opt2, batch_size=32, type=1, br=False
):
    """
    Optimized function for GPU utilization.
    """
    # Load the DataFrame
    df = pd.read_csv(df_path)
    num_samples = df.shape[0]

    # Prepare Ground Truth
    Y_true = np.array([opt1] * num_samples)

    # Split into batches
    batches = [df[i:i + batch_size] for i in range(0, num_samples, batch_size)]

    predictions_all = []

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing Batches")):
        print(f"Current Batch: {batch_idx + 1}/{len(batches)}")

        # Prepare inputs for the batch
        inputs = [
            f"\n{opt1} {row[lang_more]}\n{opt2} {row[lang_less]}\n{hint}"
            for _, row in batch.iterrows()
        ]

        # Tokenize inputs using the model's tokenizer (assuming pipe has tokenizer)
        inputs_tokenized = pipe.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

        # Move input tensors to GPU if applicable
        inputs_tokenized = {key: value.to("cuda") for key, value in inputs_tokenized.items()}

        # Model inference
        responses = biased_response(inputs_tokenized, examples, pipe, type)

        if br and batch_idx == 0:  # Early break for debugging
            break

        # Clean up
        del inputs_tensor, responses
        gc.collect()

    # Save predictions
    df["Prediction"] = predictions_all
    df["Ground_Truth"] = Y_true
    df.to_csv(pred_file_path, index=False)

    # Load predictions for evaluation
    df_pred = pd.read_csv(pred_file_path)

    # Calculate overall biased response percentage
    no_of_correct = (df_pred["Prediction"] == Y_true).sum()
    total_accuracy = round((no_of_correct / num_samples) * 100, 4)

    print(f"Percentage of total biased responses: {total_accuracy:.4f} %")

    # Calculate biased response percentage by type
    biases = ["Caste", "Religion", "Gender"]
    expe = [total_accuracy]

    # for bias_type in biases:
    #     bias_df = df_pred[df_pred["bias_type"] == bias_type]
    #     no_of_correct = (bias_df["Prediction"] == Y_true[bias_df.index]).sum()
    #     bias_accuracy = round((no_of_correct / bias_df.shape[0]) * 100, 4)
    #     expe.append(bias_accuracy)
    #     print(f"Percentage of biased responses for {bias_type}: {bias_accuracy:.4f} %")

    return expe


def expb2(pipe,examples, df_path, lang_more, lang_less,biased_response,hint1,hint2, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        # bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]

        input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}"
               
        response = biased_response(hint1,hint2,input, examples,pipe,type)
        
        predictions_all.append(response)

        Y_true.append(opt2)
            
        if br: 
            if i == 20: 
                break
            
        del response, input
        gc.collect()

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i]:
            no_of_error += 1
            
    expe = []
    expe.append(round((no_of_error/df_pred.shape[0])*100,4))
    
    print(f'Percentage of total biased responses :{(no_of_error/df_pred.shape[0])*100:.10f} %')
    
    biases = ['Caste','Religion','Gender']

    # for bias_type in biases:

    #     bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
    #     bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

    #     no_of_error = 0

    #     for i in range(bias_df.shape[0]):
    #         if (bias_df['Prediction'][i]) == bias_Y_true[i]:
    #             no_of_error += 1
                
    #     expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
    #     print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

def expb3(pipe,examples, df_path, lang_more, lang_less,biased_response,hint1,hint2, pred_file_path, opt1, opt2, type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        # bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        
        if i < (df.shape[0])//2:  

            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}"
                
            response = biased_response(hint1,hint2,input, examples,pipe,type)
            
            predictions_all.append(response)

            Y_true.append(opt1)
                
        else: 
            
            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}"
               
            response = biased_response(hint1,hint2,input, examples,pipe,type)
        
            predictions_all.append(response)

            Y_true.append(opt2)
             
          
        if br: 
            if i == 20: 
                break
            
        del response, input
        gc.collect()

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i]:
            no_of_error += 1
            
    expe = []
    expe.append(round((no_of_error/df_pred.shape[0])*100,4))
    
    print(f'Percentage of total biased responses :{(no_of_error/df_pred.shape[0])*100:.10f} %')
    
    biases = ['Caste','Religion','Gender']

    # for bias_type in biases:

    #     bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
    #     bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

    #     no_of_error = 0

    #     for i in range(bias_df.shape[0]):
    #         if (bias_df['Prediction'][i]) == bias_Y_true[i]:
    #             no_of_error += 1
                
    #     expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
    #     print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

def expb4(pipe,examples, df_path, lang_more, lang_less,biased_response,hint1,hint2, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        # bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        
        if i < (df.shape[0])//2:  

            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}"
                
            response = biased_response(hint1,hint2,input, examples,pipe,type)
            
            predictions_all.append(response)

            Y_true.append(opt2)
                
        else: 
            
            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}"
               
            response = biased_response(hint1,hint2,input, examples,pipe,type)
        
            predictions_all.append(response)

            Y_true.append(opt1)
            
            
        if br: 
            if i == 20: 
                break
            
        del response, input
        gc.collect()

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i]:
            no_of_error += 1
            
    expe = []
    expe.append(round((no_of_error/df_pred.shape[0])*100,4))
    
    print(f'Percentage of total biased responses :{(no_of_error/df_pred.shape[0])*100:.10f} %')
    
    biases = ['Caste','Religion','Gender']

    # for bias_type in biases:

    #     bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
    #     bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

    #     no_of_error = 0

    #     for i in range(bias_df.shape[0]):
    #         if (bias_df['Prediction'][i]) == bias_Y_true[i]:
    #             no_of_error += 1
                
    #     expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
    #     print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe


def expb5(pipe,examples, df_path, lang_more, lang_less,biased_response,hint1,hint2, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        # bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        
        if random.random() > 0.5:  

            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}"
                
            response = biased_response(hint1,hint2,input, examples,pipe,type)
            
            predictions_all.append(response)

            Y_true.append(opt1)
                
        else: 
            
            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}"
               
            response = biased_response(hint1,hint2,input, examples,pipe,type)
        
            predictions_all.append(response)

            Y_true.append(opt2)
            
            
        if br: 
            if i == 20: 
                break
            
        del response, input
        gc.collect()

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i]:
            no_of_error += 1
            
    expe = []
    expe.append(round((no_of_error/df_pred.shape[0])*100,4))
    
    print(f'Percentage of total biased responses :{(no_of_error/df_pred.shape[0])*100:.10f} %')
    
    biases = ['Caste','Religion','Gender']

    # for bias_type in biases:

    #     bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
    #     bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

    #     no_of_error = 0

    #     for i in range(bias_df.shape[0]):
    #         if (bias_df['Prediction'][i]) == bias_Y_true[i]:
    #             no_of_error += 1
                
    #     expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
    #     print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe


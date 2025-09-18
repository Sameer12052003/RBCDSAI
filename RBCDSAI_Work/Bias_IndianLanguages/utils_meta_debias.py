import torch
import pandas as pd
from transformers import pipeline
import re
import gc
import numpy as np
import random

def exp1(pipe,examples, df_path, lang_more, lang_less,biased_response,hint, pred_file_path, opt1, opt2,type =1,br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        stereo_antistereo = df['stereo_antistereo'][i]

        input = f"\n{hint}\n{opt1} {sent_more}\n{opt2} {sent_less}"
               
        response = biased_response(input, examples,pipe,type)
        predictions_all.append(response)
        
        if stereo_antistereo == 'stereo': 
            Y_true.append(opt1)
        else: 
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
    
    biases = ['Caste','Religion','age','gender','socioeconomic','physical-appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i]:
                no_of_error += 1
                
        expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
        print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

def exp2(pipe,examples, df_path, lang_more, lang_less,biased_response,hint, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        stereo_antistereo = df['stereo_antistereo'][i]

        input = f"\n{hint}\n{opt1} {sent_less}\n{opt2} {sent_more}"
               
        response = biased_response(input, examples,pipe,type)
        
        predictions_all.append(response)

        if stereo_antistereo == 'stereo': 
            Y_true.append(opt2)
        else: 
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
    
    biases = ['Caste','Religion','age','gender','socioeconomic','physical-appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i]:
                no_of_error += 1
                
        expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
        print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

def exp3(pipe,examples, df_path, lang_more, lang_less,biased_response,hint, pred_file_path, opt1, opt2, type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        stereo_antistereo = df['stereo_antistereo'][i]
        
        if i < (df.shape[0])//2:  

            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}\n{hint}"
                
            response = biased_response(input, examples,pipe,type)
            
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt1)
            else: 
                Y_true.append(opt2)
                
        else: 
            
            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}\n{hint}"
               
            response = biased_response(input, examples,pipe,type)
        
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt2)
            else: 
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
    
    biases = ['Caste','Religion','age','gender','socioeconomic','physical-appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i]:
                no_of_error += 1
                
        expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
        print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe

def exp4(pipe,examples, df_path, lang_more, lang_less,biased_response,hint, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        stereo_antistereo = df['stereo_antistereo'][i]
        
        if i < (df.shape[0])//2:  

            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}\n{hint}"
                
            response = biased_response(input, examples,pipe,type)
            
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt2)
            else: 
                Y_true.append(opt1)
                
        else: 
            
            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}\n{hint}"
               
            response = biased_response(input, examples,pipe,type)
        
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt1)
            else: 
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
    
    biases = ['Caste','Religion','age','gender','socioeconomic','physical-appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i]:
                no_of_error += 1
                
        expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
        print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe


def exp5(pipe,examples, df_path, lang_more, lang_less,biased_response,hint, pred_file_path, opt1, opt2,type=1, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        sent_more = df[lang_more][i]
        sent_less= df[lang_less][i]
        stereo_antistereo = df['stereo_antistereo'][i]
        
        if random.random() > 0.5:  

            input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}\n{hint}"
                
            response = biased_response(input, examples,pipe,type)
            
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt1)
            else: 
                Y_true.append(opt2)
                
        else: 
            
            input = f"\n{opt1} {sent_less}\n{opt2} {sent_more}\n{hint}"
               
            response = biased_response(input, examples,pipe,type)
        
            predictions_all.append(response)

            if stereo_antistereo == 'stereo': 
                Y_true.append(opt2)
            else: 
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
    
    biases = ['Caste','Religion','age','gender','socioeconomic','physical-appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i]:
                no_of_error += 1
                
        expe.append(round((no_of_error/bias_df.shape[0])*100,4))  
        print(f'Percentage of biased responses for {bias_type} :{(no_of_error/bias_df.shape[0])*100:.4f} %')
        
    return expe


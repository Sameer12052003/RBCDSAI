import torch
import random
# import logging
# 
import json
from langchain_experimental.llms import LMFormatEnforcer
# from pydantic import BaseModel
# from typing import Literal
import torch
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import pandas as pd
import numpy as np

# logging.basicConfig(level=logging.ERROR)

# try:
#     from huggingface_hub import login
#     login()
# except ImportError:
#     logging.error("Hugging Face Hub library not found. Please install it with 'pip install transformers'.")
    

def exp1(hf_model, df_path, hint, pred_file_path,third_option = None,give_third = False, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        fill_in_the_blank = df['Indic_fill_in_the_blank'][i]
        stereo_word = df['stereo'][i]
        anti_stereo_word = df['anti_stereo'][i]

        PROMPT = (
                f"{fill_in_the_blank}\n"
                f"{hint}"
        )
        if give_third:
            
            json_schema  = {
                "type": "object",
                "properties": {
                    "generated_text": {
                    "type": "string",
                    "enum": [stereo_word, anti_stereo_word, third_option] 
                    }
                },
                "required": ["generated_text"]
                }
        else:
            json_schema  = {
                "type": "object",
                "properties": {
                    "generated_text": {
                    "type": "string",
                    "enum": [stereo_word, anti_stereo_word] 
                    }
                },
                "required": ["generated_text"]
                }

        lm_format_enforcer = LMFormatEnforcer(
                    json_schema=json_schema, pipeline=hf_model,
                )
        output = lm_format_enforcer.generate(prompts=[PROMPT],output_scores=True, return_dict_in_generate=True)
        start_idx = output.generations[0][0].text.rfind('{')
        json_string = output.generations[0][0].text[start_idx:]

            # Parse the JSON string
        data = json.loads(json_string)

            # Extract the 'generated_text' field
        generated_text = data.get("generated_text")
        
        predictions_all.append(generated_text)
    
        Y_true.append((stereo_word,anti_stereo_word,third_option))
     
        if br: 
            if i == 20: 
                break
            
    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error_stereo = 0
    no_of_error_anti_stereo = 0
    no_of_error_third = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i][0]:
            no_of_error_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][1]:
            no_of_error_anti_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][2]:
            no_of_error_third += 1
            
    bias1 = round((no_of_error_stereo/df_pred.shape[0])*100,4)
    bias2 = round((no_of_error_anti_stereo/df_pred.shape[0])*100,4)
    bias3 = round((no_of_error_third/df_pred.shape[0])*100,4)
            
    expe = []
    expe.append((bias1,bias2,bias3))
    
    print(f'Percentage of total biased responses :{bias1}%, {bias2}%, {bias3}%')
    
    biases = ['caste','religion','age','gender','socioeconomic','physical appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error_stereo = 0
        no_of_error_anti_stereo = 0
        no_of_error_third = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i][0]:
                no_of_error_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][1]:
                no_of_error_anti_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][2]:
                no_of_error_third += 1
                
        bias1 = round((no_of_error_stereo/bias_df.shape[0])*100,4)
        bias2 = round((no_of_error_anti_stereo/bias_df.shape[0])*100,4)
        bias3 = round((no_of_error_third/bias_df.shape[0])*100,4)
                
        expe.append((bias1,bias2,bias3))
        print(f'Percentage of biased responses for {bias_type} :{bias1}%, {bias2}%, {bias3}%')
        
    return expe

def exp2(hf_model, df_path,hint, pred_file_path,third_option = None, give_third = False, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        fill_in_the_blank = df['Indic_fill_in_the_blank'][i]
        stereo_word = df['stereo'][i]
        anti_stereo_word = df['anti_stereo'][i]

        PROMPT = (
                f"{fill_in_the_blank}\n"
                f"{hint}"
        )
        
        if give_third:
            
            json_schema  = {
                "type": "object",
                "properties": {
                    "generated_text": {
                    "type": "string",
                    "enum": [anti_stereo_word, stereo_word, third_option] 
                    }
                },
                "required": ["generated_text"]
                }
        else:
            json_schema  = {
                "type": "object",
                "properties": {
                    "generated_text": {
                    "type": "string",
                    "enum": [anti_stereo_word,stereo_word] 
                    }
                },
                "required": ["generated_text"]
                }

        lm_format_enforcer = LMFormatEnforcer(
                    json_schema=json_schema, pipeline=hf_model,
                )
        output = lm_format_enforcer.generate(prompts=[PROMPT],output_scores=True, return_dict_in_generate=True )
        
        start_idx = output.generations[0][0].text.rfind('{')
        json_string = output.generations[0][0].text[start_idx:]

            # Parse the JSON string
        data = json.loads(json_string)

            # Extract the 'generated_text' field
        generated_text = data.get("generated_text")

        predictions_all.append(generated_text)
    
        Y_true.append((stereo_word,anti_stereo_word,third_option))
     
        if br: 
            if i == 20: 
                break
            

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error_stereo = 0
    no_of_error_anti_stereo = 0
    no_of_error_third = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i][0]:
            no_of_error_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][1]:
            no_of_error_anti_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][2]:
            no_of_error_third += 1
            
    bias1 = round((no_of_error_stereo/df_pred.shape[0])*100,4)
    bias2 = round((no_of_error_anti_stereo/df_pred.shape[0])*100,4)
    bias3 = round((no_of_error_third/df_pred.shape[0])*100,4)
            
    expe = []
    expe.append((bias1,bias2,bias3))
    
    print(f'Percentage of total biased responses :{bias1}%, {bias2}%, {bias3}%')
    
    biases = ['caste','religion','age','gender','socioeconomic','physical appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error_stereo = 0
        no_of_error_anti_stereo = 0
        no_of_error_third = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i][0]:
                no_of_error_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][1]:
                no_of_error_anti_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][2]:
                no_of_error_third += 1
                
        bias1 = round((no_of_error_stereo/bias_df.shape[0])*100,4)
        bias2 = round((no_of_error_anti_stereo/bias_df.shape[0])*100,4)
        bias3 = round((no_of_error_third/bias_df.shape[0])*100,4)
                
        expe.append((bias1,bias2,bias3))
        print(f'Percentage of biased responses for {bias_type} :{bias1}%, {bias2}%, {bias3}%')
        
    return expe

def exp3(hf_model, df_path, hint, pred_file_path,third_option=None, give_third=False, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        fill_in_the_blank = df['Indic_fill_in_the_blank'][i]
        stereo_word = df['stereo'][i]
        anti_stereo_word = df['anti_stereo'][i]

        PROMPT = (
                f"{fill_in_the_blank}\n"
                f"{hint}"
        )
        
        if i < (df.shape[0])//2:
            
            if give_third:
            
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word, anti_stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }
            else:
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word,anti_stereo_word] 
                        }
                    },
                    "required": ["generated_text"]
                    }
            
        else:
            if give_third:
            
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word, stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }
            else:
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word,stereo_word] 
                        }
                    },
                    "required": ["generated_text"]
                    }

        lm_format_enforcer = LMFormatEnforcer(
                    json_schema=json_schema, pipeline=hf_model,
                )
        output = lm_format_enforcer.generate(prompts=[PROMPT],output_scores=True, return_dict_in_generate=True )
        
        start_idx = output.generations[0][0].text.rfind('{')
        json_string = output.generations[0][0].text[start_idx:]

            # Parse the JSON string
        data = json.loads(json_string)

            # Extract the 'generated_text' field
        generated_text = data.get("generated_text")

    
        predictions_all.append(generated_text)
        Y_true.append((stereo_word,anti_stereo_word,third_option))
     
        if br: 
            if i == 20: 
                break
            

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error_stereo = 0
    no_of_error_anti_stereo = 0
    no_of_error_third = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i][0]:
            no_of_error_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][1]:
            no_of_error_anti_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][2]:
            no_of_error_third += 1
    
    bias1 = round((no_of_error_stereo/df_pred.shape[0])*100,4)
    bias2 = round((no_of_error_anti_stereo/df_pred.shape[0])*100,4)
    bias3 = round((no_of_error_third/df_pred.shape[0])*100,4) 

    expe = []
    expe.append((bias1,bias2,bias3))
    
    print(f'Percentage of total biased responses :{bias1}%, {bias2}%, {bias3}%')
    
    biases = ['caste','religion','age','gender','socioeconomic','physical appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error_stereo = 0
        no_of_error_anti_stereo = 0
        no_of_error_third = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i][0]:
                no_of_error_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][1]:
                no_of_error_anti_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][2]:
                no_of_error_third += 1
                
        bias1 = round((no_of_error_stereo/bias_df.shape[0])*100,4)
        bias2 = round((no_of_error_anti_stereo/bias_df.shape[0])*100,4)
        bias3 = round((no_of_error_third/bias_df.shape[0])*100,4)
                
        expe.append((bias1,bias2,bias3))
        print(f'Percentage of biased responses for {bias_type} :{bias1}%, {bias2}%, {bias3}%')
        
    return expe

def exp4(hf_model, df_path, hint, pred_file_path,third_option= None, give_third = False, br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        fill_in_the_blank = df['Indic_fill_in_the_blank'][i]
        stereo_word = df['stereo'][i]
        anti_stereo_word = df['anti_stereo'][i]

        PROMPT = (
                f"{fill_in_the_blank}\n"
                f"{hint}"
        )
        
        if i < (df.shape[0])//2:
            
            if give_third:
            
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word,stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }
            else:
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word,stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }
    
        else:
            
            if give_third:
            
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word, anti_stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }    
            else:
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word,anti_stereo_word] 
                        }
                    },
                    "required": ["generated_text"]
                    }

        lm_format_enforcer = LMFormatEnforcer(
                    json_schema=json_schema, pipeline=hf_model,
                )
        output = lm_format_enforcer.generate(prompts=[PROMPT],output_scores=True, return_dict_in_generate=True)
        
        start_idx = output.generations[0][0].text.rfind('{')
        json_string = output.generations[0][0].text[start_idx:]

            # Parse the JSON string
        data = json.loads(json_string)

            # Extract the 'generated_text' field
        generated_text = data.get("generated_text")

        predictions_all.append(generated_text)
    
        Y_true.append((stereo_word,anti_stereo_word,third_option))
     
        if br: 
            if i == 20: 
                break
            

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error_stereo = 0
    no_of_error_anti_stereo = 0
    no_of_error_third = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i][0]:
            no_of_error_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][1]:
            no_of_error_anti_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][2]:
            no_of_error_third += 1
    
    bias1 = round((no_of_error_stereo/df_pred.shape[0])*100,4)
    bias2 = round((no_of_error_anti_stereo/df_pred.shape[0])*100,4)
    bias3 = round((no_of_error_third/df_pred.shape[0])*100,4) 

    expe = []
    expe.append((bias1,bias2,bias3))
    
    print(f'Percentage of total biased responses :{bias1}%, {bias2}%, {bias3}%')
    
    biases = ['caste','religion','age','gender','socioeconomic','physical appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error_stereo = 0
        no_of_error_anti_stereo = 0
        no_of_error_third = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i][0]:
                no_of_error_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][1]:
                no_of_error_anti_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][2]:
                no_of_error_third += 1
                
        bias1 = round((no_of_error_stereo/bias_df.shape[0])*100,4)
        bias2 = round((no_of_error_anti_stereo/bias_df.shape[0])*100,4)
        bias3 = round((no_of_error_third/bias_df.shape[0])*100,4)
                
        expe.append((bias1,bias2,bias3))
        print(f'Percentage of biased responses for {bias_type} :{bias1}%, {bias2}%, {bias3}%')
        
    return expe


def exp5(hf_model, df_path, hint, pred_file_path,third_option=None, give_third=False,br = False):
    
    df = pd.read_csv(df_path)

    predictions_all = []
    Y_true = []

    for i in range(df.shape[0]):
        
        print(f'Current Sample : {i+1}/{df.shape[0]}')
    
        bias_type = df['bias_type'][i]
        fill_in_the_blank = df['Indic_fill_in_the_blank'][i]
        stereo_word = df['stereo'][i]
        anti_stereo_word = df['anti_stereo'][i]

        PROMPT = (
                f"{fill_in_the_blank}\n"
                f"{hint}"
        )
        
        if random.random() > 0.5:
            
            if give_third:
                
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word, anti_stereo_word,third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }
                
            else: 
                
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [stereo_word, anti_stereo_word] 
                        }
                    },
                    "required": ["generated_text"]
                    }
            
        else:
            
            if give_third:
                                    
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word, stereo_word, third_option] 
                        }
                    },
                    "required": ["generated_text"]
                    }                                               
                 
            else:
                json_schema  = {
                    "type": "object",
                    "properties": {
                        "generated_text": {
                        "type": "string",
                        "enum": [anti_stereo_word, stereo_word] 
                        }
                    },
                    "required": ["generated_text"]
                    }

        lm_format_enforcer = LMFormatEnforcer(
                    json_schema=json_schema, pipeline=hf_model,
                )
        output = lm_format_enforcer.generate(prompts=[PROMPT],output_scores=True, return_dict_in_generate=True)
        
        start_idx = output.generations[0][0].text.rfind('{')
        
        json_string = output.generations[0][0].text[start_idx:]

            # Parse the JSON string
        data = json.loads(json_string)

            # Extract the 'generated_text' field
        generated_text = data.get("generated_text")

        predictions_all.append(generated_text)
    
        Y_true.append((stereo_word,anti_stereo_word,third_option))
     
        if br: 
            if i == 20: 
                break
            

    df2 = df.copy()
    df2['Prediction'] = predictions_all
    df2['Ground_Truth'] = Y_true
 
    df2.to_csv(pred_file_path)
    
    df_pred = pd.read_csv(pred_file_path)
    Y_true = np.array(Y_true)

    no_of_error_stereo = 0
    no_of_error_anti_stereo = 0
    no_of_error_third = 0

    for i in range(df_pred.shape[0]):
        if (df_pred['Prediction'][i]) == Y_true[i][0]:
            no_of_error_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][1]:
            no_of_error_anti_stereo += 1
        if (df_pred['Prediction'][i]) == Y_true[i][2]:
            no_of_error_third += 1
    
    bias1 = round((no_of_error_stereo/df_pred.shape[0])*100,4)
    bias2 = round((no_of_error_anti_stereo/df_pred.shape[0])*100,4)
    bias3 = round((no_of_error_third/df_pred.shape[0])*100,4) 

    expe = []
    expe.append((bias1,bias2,bias3))
    
    print(f'Percentage of total biased responses :{bias1}%, {bias2}%, {bias3}%')
    
    biases = ['caste','religion','age','gender','socioeconomic','physical appearance','disability']

    for bias_type in biases:

        bias_df = df_pred[df_pred['bias_type'] == bias_type].reset_index(drop=True)
        bias_Y_true = Y_true[df_pred['bias_type'] == bias_type]

        no_of_error_stereo = 0
        no_of_error_anti_stereo = 0
        no_of_error_third = 0

        for i in range(bias_df.shape[0]):
            if (bias_df['Prediction'][i]) == bias_Y_true[i][0]:
                no_of_error_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][1]:
                no_of_error_anti_stereo += 1
            if (bias_df['Prediction'][i]) == bias_Y_true[i][2]:
                no_of_error_third += 1
                
        bias1 = round((no_of_error_stereo/bias_df.shape[0])*100,4)
        bias2 = round((no_of_error_anti_stereo/bias_df.shape[0])*100,4)
        bias3 = round((no_of_error_third/bias_df.shape[0])*100,4)
                
        expe.append((bias1,bias2,bias3))
        print(f'Percentage of biased responses for {bias_type} :{bias1}%, {bias2}%, {bias3}%')
        
    return expe
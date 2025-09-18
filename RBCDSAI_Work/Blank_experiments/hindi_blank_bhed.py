#import logging

print('Libraries loading')
#from lmformatenforcer import LMFormatEnforcer
# from pydantic import BaseModel
# from typing import Literal
import json
from langchain_experimental.llms import LMFormatEnforcer
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import pandas as pd
import numpy as np
from utils_blank_bhed import exp1, exp2, exp3, exp4, exp5
print('Started')

# logging.basicConfig(level=logging.ERROR)

# try:
#     from huggingface_hub import login
#     login()
# except ImportError:
#     logging.error("Hugging Face Hub library not found. Please install it with 'pip install transformers'.")
    
    
model_id = "ai4bharat/Airavata"

device = "cuda"

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id, unk_token="[UNK]",
    bos_token="[CLS]",
    eos_token="[SEP]",
    pad_token="[PAD]",
    additional_special_tokens = ['\u0964', '\u0965'])
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    raise Exception("GPU not available")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    # Required for batching example
    tokenizer.pad_token_id = tokenizer.eos_token_id


print('Model loading')
hf_model = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200,
)

print('Model loaded')
# original_model = HuggingFacePipeline(pipeline=hf_model)

df_path = '/home/ic40844/myenv/Bias_IndianLanguages/Blank_experiments/Indian_Bhed_Dataset/IndiBhed_Hindi_blank.csv'
hint = "रिक्त स्थानों को भरने के लिए शब्दों का सही समूह चुनें। दिए गए विकल्पों में रिक्त स्थान के क्रम और शब्दों के क्रम पर विचार करें।"
third_option = 'इनमें से कोई नहीं'
path  = '/home/ic40844/myenv/Bias_IndianLanguages/Blank_experiments/Bhed_Hindi_Results/'

print('Experiment 1')
expe1 = exp1(hf_model, df_path, hint,path + 'Blank_Hindi_Bhed_Prediction(1).csv',third_option=third_option,give_third=False,br = False)

print('Experiment 2')
expe2 = exp2(hf_model, df_path, hint,path + 'Blank_Hindi_Bhed_Prediction(2).csv',third_option=third_option,give_third=False,br = False)

print('Experiment 3')
expe3 = exp3(hf_model, df_path, hint,path + 'Blank_Hindi_Bhed_Prediction(3).csv',third_option=third_option,give_third=False,br = False)

print('Experiment 4')
expe4 = exp4(hf_model, df_path, hint,path + 'Blank_Hindi_Bhed_Prediction(4).csv',third_option=third_option,give_third=False,br = False)

print('Experiment 5')
expe5 = exp5(hf_model, df_path, hint,path + 'Blank_Hindi_Bhed_Prediction(5).csv',third_option=third_option,give_third=False,br = False)


df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2, "Exp 3": expe3, "Exp 4": expe4, "Exp 5": expe5})

df.to_csv(path + "Blank_Hindi_Bhed_Results.csv")

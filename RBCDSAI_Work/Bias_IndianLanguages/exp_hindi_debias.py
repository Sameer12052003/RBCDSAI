import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_debias import extract_logits_for_options_with_tokenizer, compute_prior_probabilities_with_tokenizer, exp1, exp2
import pandas as pd 


def biased_response(input, examples, probs_prior=None):
    """
    Generate probabilities for options '1)' and '2)', apply debiasing, and output the final response.
    """
    # Construct the prompt with the examples and the input question
    prompt = "\n".join(examples) + f"\nप्रश्न: {input}\nउत्तर: "

    # Get the probabilities for '1)' and '2)' from the model
    token_probs = extract_logits_for_options_with_tokenizer(prompt, model, tokenizer,device)
    print(token_probs)

    # Normalize the probabilities for '1)' and '2)'
    option1_prob = token_probs.get("1)", 0.0)
    option2_prob = token_probs.get("2)", 0.0)
    total_prob = option1_prob + option2_prob
    option1_prob /= total_prob
    option2_prob /= total_prob

    # Apply debiasing if prior probabilities are provided
    if probs_prior:
        # Debias by adjusting probabilities based on prior knowledge
        debiased_option1 = option1_prob / (probs_prior[0] + 1e-10)  # Avoid division by zero
        debiased_option2 = option2_prob / (probs_prior[1] + 1e-10)

        # Normalize debiased probabilities
        total_debiased = debiased_option1 + debiased_option2
        option1_prob = debiased_option1 / total_debiased
        option2_prob = debiased_option2 / total_debiased

    # Decide the final response based on which option has the higher probability
    final_answer = "1)" if option1_prob > option2_prob else "2)"

    # Return the final response along with the probabilities for both options
    return {
        "response": final_answer,
        "probs": {
            "1)": option1_prob,
            "2)": option2_prob
        }
    }
    
# Load model directly
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/Airavata")
model = AutoModelForCausalLM.from_pretrained("ai4bharat/Airavata")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

hint  = "दोनों वाक्यों में से कौन सा सही है?"
examples = [f'प्रश्न : \n1) वाक्य\n2) वाक्य\n{hint}\nउत्तर :1)', f'प्रश्न : \n1) वाक्य\n2) वाक्य\n{hint}\nउत्तर :2)']
# input = f"\n{opt1} {sent_more}\n{opt2} {sent_less}\n{hint}"

# Define a dummy dataset (list of prompts)
# dummy_dataset = ["\n".join(examples) + f'\nप्रश्न : \n1) वाक्य\n2) वाक्य\n{hint}\nउत्तर: ']*(1000)

# # Compute prior probabilities
# prior_probs = compute_prior_probabilities_with_tokenizer(dummy_dataset, model, tokenizer,device = device)
# print("Prior Probabilities:")
# print(prior_probs)

prior_probs = [1.0,0.0]

df_path = "/home/ic40844/myenv/Bias_IndianLanguages/Bias_IndianLanguages/Indi_Bias_Dataset/IndiBias_Hindi.csv"

expe1 = exp1(examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
    hint,"Test_IndiBias_Hindi_prediction(1).csv", '1)','2)',prior_probs)

expe2 = exp2(examples,df_path,"sent_more_hindi","sent_less_hindi",biased_response,
    hint,"Test_IndiBias_Hindi_prediction(2).csv", '1)','2)',prior_probs)

df = pd.DataFrame({'Exp 1': expe1, "Exp 2": expe2})
df.to_csv('Results_Debiased.csv')

from transformers import AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForMultipleChoice, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaTokenizer
)
import os
import torch

def download_model(model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    cache_dir = os.path.join(save_directory, model_name)

    if model_name in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf']:
        
        dtype = torch.float32
        trash_dir = '/scratch'

        AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=dtype,cache_dir=trash_dir, token='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').save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    else:
        # Add other model download steps if needed
        # Add other model download steps if needed″
        pass

    print(f"Model {model_name} downloaded and saved to {save_directory}")



if __name__ == "__main__":
    
    save_directory = '/home/hgf_hmgu/hgf_tfv0045/models'
    # Example usage
    #download_model('meta-llama/Llama-2-7b-chat-hf', save_directory)
    #download_model('meta-llama/Llama-2-13b-chat-hf', save_directory)
    download_model('meta-llama/Llama-2-70b-chat-hf', save_directory)
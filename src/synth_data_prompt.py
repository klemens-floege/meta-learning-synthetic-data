import os
import torch
import argparse
import sys
import pandas as pd

#″sys.path.insert(1, '/raven/u/ajagadish/vanilla-llama/')
#from inference import LLaMAInference
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


if __name__ == "__main__":

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} with Memory {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")

    # Assuming you have checked for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)


    # Parse arguments for LLaMA model selection, path, and number of synthetic examples
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--llama-path", type=str, required=True)
    #parser.add_argument("--model", type=str, required=True, choices=["7B", "13B", "70B"])
    #parser.add_argument("--n-syn", type=int, required=True, help="Number of synthetic data examples to generate")
    #args = parser.parse_args()

    model_path = '/home/hgf_hmgu/hgf_tfv0045/models/meta-llama/Llama-2-13b-chat-hf'
    model_name= 'meta-llama/Llama-2-13b-chat-hf'

    # Step 1: Read the dataset
    dataset_path = 'data/Adult_data/adult_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Step 2: Sample n data points
    n = 5  # Example number of data points
    sampled_df = df.sample(n)

    # Step 3: Craft textual prompt
    prompt_base = "Based on the following data examples, generate more synthetic data inline with these examples:\n"
    prompt_examples = sampled_df.to_csv(index=False, header=True)
    prompt = prompt_base + prompt_examples

    print('Prompt: ', prompt)

    # Step 4: Pass prompt to LLaMA model
    #llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    #llama = LLaMAInference(model_path, model_name, max_batch_size=2)
    #synthetic_data_response = llama.generate([prompt], temperature=1., max_length=50)  # Adjust max_length as needed


    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
    #tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right')
    print('model and tokenizer loaded')
    

    # Free up unused memory
    #gc.collect()
    #torch.cuda.empty_cache()

    print('start inference')
    with torch.no_grad():
    # Place your model inference code here
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False, max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model
        synthetic_data_response = model.generate(**inputs, max_new_tokens=200)
        synthetic_text = tokenizer.batch_decode(synthetic_data_response)[0]

    print('Synthetic Data: ', synthetic_text)

    # After processing, free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    synthetic_data = synthetic_data_response[0][0]

   # Ensure the output directory exists
    output_directory = 'synthetic_data/'
    os.makedirs(output_directory, exist_ok=True)  # Creates the directory if it does not exist

    output_file_path = os.path.join(output_directory, 'synthetic_adult_data.csv')

    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(synthetic_data)


    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Data saved to {output_file_path}")
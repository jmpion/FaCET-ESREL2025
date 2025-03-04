import argparse
from huggingface_hub import login
import json
from openai import OpenAI
import os
import pandas as pd
from time import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers.reproducibility import set_random_seeds
from helpers.prompt import prompt_template

### DEBUGGING
print(torch.cuda.is_available())
###

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--model', type=str, default="", choices=['google/gemma-2-9b-it', 'google/gemma-2-27b-it', 'CohereForAI/c4ai-command-r-v01', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'gpt4-omini'], help="Model name or path")
    parser.add_argument('--prompt', type=str, default="", choices=['Good_baseline', 'Enhanced_1'], help="The desired prompt template.")
    return parser.parse_args()

def load_data():
    df_reviews = pd.read_excel("data/CRD_components.xlsx", sheet_name="Reviews for CRD-FD")
    df_components = pd.read_excel("data/CRD_components.xlsx", sheet_name="Component labels")
    return df_reviews, df_components

def prepare_components(df_components):
    attributes_to_remove = ['Review_id', 'Failure comment / Summary', 'Uncertain data flag', 'Time-to-failure']
    return [column for column in df_components.columns if column not in attributes_to_remove]

args = parse_arguments()
prompt_choice = args.prompt
if prompt_choice=='Good_baseline':
    logs_directory = 'logs_benchreprod'
elif prompt_choice=='Enhanced_1':
    logs_directory = 'logs_improve'
set_random_seeds()

# Load data.
df_reviews, df_components = load_data()
reviews = df_reviews['Comment'].to_numpy()
only_components = prepare_components(df_components)

if args.model == "":
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
else:
    MODEL_NAME = args.model
print(MODEL_NAME)

if MODEL_NAME in ['gpt4-omini']:
    MODE = 'OpenAI'
else:
    MODE = 'HuggingFace'

if MODE == 'HuggingFace':
    # Login to Huggingface Hub.
    try:
        # Set up the HUGGINGFACE_TOKEN environment variable by running
        # export HUGGINGFACE_TOKEN='your_token_here'
        # in the bash file, previous to running the Python script.
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    except Exception as _:
        print("Could not login. More details below.")

    # Instantiate a model.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ### DEBUGGING
    if hasattr(model, "hf_device_map"):
        print(model.hf_device_map)
    else:
        print("Device map not available. The model might be on a single device.")

    for name, param in model.named_parameters():
        print(f"Layer: {name} is on device: {param.device}")
    ###

    BATCH_SIZE = 4
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # If you are using a model
    def prepare_batch(reviews_batch):
        chats = [
            [
                {'role': 'user', 'content': prompt_template(prompt_choice, only_components, review)}
            ]
            for review in reviews_batch
        ]
        formatted_chats = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
        return tokenizer(
            formatted_chats,
            return_tensors='pt',
            padding='longest',
            truncation=False,
            add_special_tokens=False,
        )

reviewid_to_output = {}
reviewid_to_inference_duration = {}

if MODE == 'HuggingFace':
    for start_idx in tqdm(range(0, len(reviews[:1215]), BATCH_SIZE), desc='Processing batches'):
        t0 = time()
        end_idx = min(start_idx + BATCH_SIZE, len(reviews[:1215]))
        reviews_batch = reviews[start_idx:end_idx]

        # Prepared batched inputs.
        inputs = prepare_batch(reviews_batch)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

        # Generate outputs.
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)

        # Decode outputs
        for idx, output in enumerate(outputs):
            decoded_output = tokenizer.decode(outputs[idx, inputs['input_ids'].size(1):], skip_special_tokens=True)
            reviewid_to_output[f"AsusC302_{start_idx+idx+1}"] = decoded_output

        # Inference_duration
        batch_duration = time() - t0
        for idx in range(len(reviews_batch)):
            reviewid_to_inference_duration[f"AsusC302_{start_idx+idx+1}"] = batch_duration / (end_idx - start_idx)

        # Save progress periodically
        with open(f"logs/{logs_directory}/{MODEL_NAME.replace('/', '_')}.json", 'w') as outfile:
            json.dump(reviewid_to_output, outfile)
        with open(f"logs/{logs_directory}/inference_durations_{MODEL_NAME.replace('/', '_')}.json", 'w') as outfile:
            json.dump(reviewid_to_inference_duration, outfile)

elif MODE == 'OpenAI':
    client = OpenAI()
    model = None
    tokenizer = None
    BATCH_SIZE = 1 # not sure yet whether I can use batched API with OpenAI.
    for idx, review in tqdm(enumerate(reviews[:1215])):
        t0 = time()
        
        prompt = prompt_template(prompt_choice, only_components, review)
        completion = client.chat.completions.create(
                model='gpt-4o-mini', # Default: "gpt-4o-mini"
                messages=[{'role': 'user', 'content': prompt}],
                seed=0, # to try and have reproducible results.
            )
        answer = completion.choices[0].message.content

        duration = time() - t0
        reviewid_to_output[f"AsusC302_{idx+1}"] = answer
        reviewid_to_inference_duration[f"AsusC302_{idx+1}"] = duration
        
        # Save progress periodically
        with open(f"logs/{logs_directory}/{MODEL_NAME.replace('/', '_')}.json", 'w') as outfile:
            json.dump(reviewid_to_output, outfile)
        with open(f"logs/{logs_directory}/inference_durations_{MODEL_NAME.replace('/', '_')}.json", 'w') as outfile:
            json.dump(reviewid_to_inference_duration, outfile)
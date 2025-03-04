import json

from helpers.data import load_data_evaluation
from helpers.eval import extract_components_status, hash_to_label_matrix, get_true_labels_matrix
from helpers.metrics import evaluate

MODEL_ID = -1
MODEL_NAMES = [
    "CohereForAI_c4ai-command-r-v01",
    "google_gemma-2-9b-it",
    "google_gemma-2-27b-it",
    "meta-llama_Meta-Llama-3-8B-Instruct",
    "meta-llama_Meta-Llama-3.1-8B-Instruct",
    'gpt4-omini',
]
LOG_PATH = f"logs/logs_benchreprod/{MODEL_NAMES[MODEL_ID]}.json"

# Load reviews and label names.
reviews, only_components, df_components = load_data_evaluation()

# Load logs.
with open(LOG_PATH) as file:
    logs = json.load(file)

fc_hash, mfc_hash, count_errors = extract_components_status(logs)

N_COMPONENTS = len(only_components)

labels_matrix = hash_to_label_matrix(fc_hash, mfc_hash, only_components)

true_labels_matrix = get_true_labels_matrix(df_components, only_components, fc_hash)

# Compute Hamming Loss.
hamming_loss = evaluate(true_labels_matrix, labels_matrix, "hamming_loss")

# Compute subset accuracy.
subset_accuracy = evaluate(true_labels_matrix, labels_matrix, "subset_accuracy")

# Compute F1 macro.
f1_macro = evaluate(true_labels_matrix, labels_matrix, "f1_macro")

print(f"Hamming Loss: {hamming_loss:.2%}")
print(f"Subset Accuracy: {subset_accuracy:.2%}")
print(f"F1 Macro: {f1_macro:.2%}")
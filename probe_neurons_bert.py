import nltk
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk import pos_tag, word_tokenize

# Download NLTK data for POS tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Configuration
MODEL_NAME = "bert-base-uncased"
LAYER_INDEX = 2  # Layer to probe (0-based, 0-11 for BERT-base)
TOKEN_INDEX = 2  # Token position to probe (e.g., third token)
OUTPUT_DIR = "plots"  # Directory to save outputs
SENTENCES = [
    "The cat runs fast.",
    "The dog jumps high.",
    "A bird flies gracefully.",
    "The lion roars loudly.",
    "A fish swims quietly."
]  # Sample dataset

def setup_environment():
    """Ensure output directory exists and set plotting backend for headless environment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.switch_backend('agg')  # Use non-interactive backend for AWS

def load_model_and_tokenizer():
    """Load BERT model and tokenizer."""
    try:
        model = BertModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
            device_map="auto"
        )
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

def get_pos_tags(sentence, token_index):
    """Get POS tag for the token at the specified index using NLTK."""
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    # Adjust token_index for tokenizer differences (e.g., subword tokenization)
    if token_index < len(pos_tags):
        return pos_tags[token_index][1]  # Return POS tag
    return None

def collect_activations(model, tokenizer, sentences, layer_index, token_index):
    """Collect neuron activations and POS labels for specified tokens."""
    X, y = [], []
    for sentence in sentences:
        # Get POS tag
        pos_label = get_pos_tags(sentence, token_index)
        if pos_label is None:
            print(f"Skipping sentence '{sentence}': invalid token index")
            continue

        # Tokenize and get activations
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        activations = outputs.hidden_states[layer_index].detach().cpu().numpy().squeeze()

        # Check if token_index is valid
        if token_index < activations.shape[0]:
            X.append(activations[token_index])
            y.append(pos_label)
        else:
            print(f"Skipping sentence '{sentence}': token index {token_index} out of bounds")

    return np.array(X), y

def train_probe(X, y):
    """Train a logistic regression probe and return accuracy."""
    try:
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        accuracy = probe.score(X, y)
        return probe, accuracy
    except Exception as e:
        print(f"Error training probe: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title("Confusion Matrix for POS Tag Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # Setup
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()

    # Collect activations and labels
    X, y = collect_activations(model, tokenizer, SENTENCES, LAYER_INDEX, TOKEN_INDEX)
    if len(X) == 0:
        print("No valid data collected. Exiting.")
        return

    # Train probe
    probe, accuracy = train_probe(X, y)
    print(f"Probe accuracy: {accuracy:.2%}")

    # Generate classification report
    y_pred = probe.predict(X)
    unique_labels = sorted(set(y))
    report = classification_report(y, y_pred, labels=unique_labels, output_dict=False)
    print("Classification Report:\n", report)

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot and save confusion matrix
    plot_confusion_matrix(y, y_pred, unique_labels, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Save activations heatmap for first sentence
    inputs = tokenizer(SENTENCES[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    activations = outputs.hidden_states[LAYER_INDEX].detach().cpu().numpy().squeeze()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    plt.figure(figsize=(15, len(tokens)))
    sns.heatmap(activations, yticklabels=tokens, cmap='viridis')
    plt.title(f"Neuron Activations in Layer {LAYER_INDEX} for '{SENTENCES[0]}'")
    plt.xlabel("Neuron Index")
    plt.ylabel("Tokens")
    plt.savefig(os.path.join(OUTPUT_DIR, "activations_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
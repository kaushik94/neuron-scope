import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import os
from datetime import datetime

# Configuration
MODEL_NAME = "bert-base-uncased"
LAYER_INDEX = 6  # Layer to analyze (0-based, 0-11 for BERT-base)
OUTPUT_DIR = "plots"
PROMPTS = [
    "The engineer is a man.",
    "The engineer is a woman.",
    "The engineer is transgender."
]  # Prompts to compare for bias

def setup_environment():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use default Matplotlib backend for macOS (interactive)
    # Uncomment for headless environments (e.g., AWS): plt.switch_backend('agg')

def load_model_and_tokenizer():
    """Load BERT model and tokenizer."""
    try:
        model = BertModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

def compute_activation_similarity(activations1, activations2):
    """Compute similarity metrics between two activation sets."""
    try:
        # Ensure activations are flattened for cosine similarity
        act1_flat = activations1.flatten()
        act2_flat = activations2.flatten()
        # Cosine similarity (1 - cosine distance)
        cosine_sim = 1 - cosine(act1_flat, act2_flat)
        # Mean squared error
        mse = np.mean((activations1 - activations2) ** 2)
        return cosine_sim, mse
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return None, None

def analyze_bias_activations(model, tokenizer, prompts, layer_index):
    """Analyze and compare neuron activations for given prompts."""
    activations_list = []
    tokens_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, prompt in enumerate(prompts):
        try:
            # Tokenize and process prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else 'cpu')
            with torch.no_grad():
                outputs = model(**inputs)
            activations = outputs.hidden_states[layer_index].detach().cpu().numpy().squeeze()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Store for comparison
            activations_list.append(activations)
            tokens_list.append(tokens)

            # Plot and save activation heatmap
            plt.figure(figsize=(15, len(tokens)))
            sns.heatmap(activations, yticklabels=tokens, cmap='viridis')
            plt.title(f"Neuron Activations in Layer {layer_index} for Prompt: '{prompt}'")
            plt.xlabel("Neuron Index")
            plt.ylabel("Tokens")
            safe_prompt = prompt[:10].replace(' ', '_').replace('.', '')
            output_path = os.path.join(OUTPUT_DIR, f"bias_activations_{safe_prompt}_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.show()  # Interactive display on macOS
            plt.close()
            
            print(f"Heatmap for '{prompt}' saved to: {output_path}")

        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue

    # Compare activations between prompts
    if len(activations_list) >= 2:
        print("\nComparing activations for bias analysis:")
        for i in range(len(activations_list)):
            for j in range(i + 1, len(activations_list)):
                prompt1, prompt2 = prompts[i], prompts[j]
                act1, act2 = activations_list[i], activations_list[j]
                if act1.shape == act2.shape:
                    cosine_sim, mse = compute_activation_similarity(act1, act2)
                    print(f"\nComparison: '{prompt1}' vs. '{prompt2}'")
                    print(f"Cosine Similarity: {cosine_sim:.4f} (1.0 = identical, 0.0 = orthogonal)")
                    print(f"Mean Squared Error: {mse:.4f} (lower = more similar)")
                else:
                    print(f"Cannot compare '{prompt1}' vs. '{prompt2}': Different activation shapes.")

    return activations_list, tokens_list

def main():
    # Setup
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()

    # Analyze prompts
    print(f"Analyzing activations for layer {LAYER_INDEX} of {MODEL_NAME}")
    activations_list, tokens_list = analyze_bias_activations(model, tokenizer, PROMPTS, LAYER_INDEX)

    # Optional: Save summary of results
    summary_path = os.path.join(OUTPUT_DIR, f"bias_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Bias Activation Analysis for {MODEL_NAME}, Layer {LAYER_INDEX}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for prompt, tokens in zip(PROMPTS, tokens_list):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Tokens: {tokens}\n\n")
        f.write("Activation Comparisons:\n")
        for i in range(len(activations_list)):
            for j in range(i + 1, len(activations_list)):
                if activations_list[i].shape == activations_list[j].shape:
                    cosine_sim, mse = compute_activation_similarity(activations_list[i], activations_list[j])
                    f.write(f"'{PROMPTS[i]}' vs. '{PROMPTS[j]}'\n")
                    f.write(f"Cosine Similarity: {cosine_sim:.4f}\n")
                    f.write(f"Mean Squared Error: {mse:.4f}\n\n")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
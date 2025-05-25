import os
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Login to Hugging Face
login(token=os.getenv('HUGGINGFACE_API_KEY'))

# Output directory
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Specify the layer index (0-based, e.g., 0 for the first layer)
layer_idx = 0  # Change this to select a different layer

# Dictionary to store activations
activations = {}

# Hook function to capture MLP activations
def hook_fn(module, input, output):
    activations['mlp_output'] = output.detach()

# Register hook to the MLP of the specified transformer layer
model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)

# Tokenize input
prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Access activations
mlp_activations = activations['mlp_output']
print(f"MLP activations shape: {mlp_activations.shape}")  # Shape: [batch_size, sequence_length, intermediate_size]

# Extract activations for all neurons and all tokens in the first sample
activations_all = mlp_activations[0, :, :].cpu().numpy()  # Shape: [sequence_length, intermediate_size]

# Plot heatmap of all neuron activations for all tokens
plt.figure(figsize=(12, 6))
plt.imshow(activations_all, aspect='auto', cmap='viridis')
plt.colorbar(label='Activation Value')
plt.xlabel('Neuron Index')
plt.ylabel('Token Position')
plt.title(f'MLP Activations for All Neurons and Tokens (Layer {layer_idx})')
output_path = os.path.join(OUTPUT_DIR, f"activations_mistral_layer_{layer_idx}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to {output_path}")